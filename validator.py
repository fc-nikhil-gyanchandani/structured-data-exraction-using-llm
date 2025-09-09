# validator_llm.py
from __future__ import annotations
import json, argparse, re, sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from openai import OpenAI
from pydantic import ValidationError, BaseModel, Field, field_validator
import instructor

# ---- local imports (same as agent) ----
from data_dict_compiler import (
    load_dictionary,
    compile_pydantic,
    check_constraints,
    ModelSpec, FieldSpec,
)
from prompt_builder import PromptBuilder

# --------------- PYDANTIC MODELS FOR INSTRUCTOR VALIDATION ---------------

class ValidationFieldEvidence(BaseModel):
    """Evidence for a field validation with chunk reference and snippet."""
    chunk_id: str = Field(..., description="Chunk ID containing the evidence")
    quote: str = Field(..., max_length=300, description="Relevant text snippet")

class ValidationResult(BaseModel):
    """Validation result for a single field."""
    ok: bool = Field(..., description="Whether the field is valid")
    why: str = Field(..., description="Explanation of validation result")
    evidence: List[ValidationFieldEvidence] = Field(default_factory=list, description="Supporting evidence")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")

class RecordValidation(BaseModel):
    """Complete record validation result."""
    binary: int = Field(..., ge=0, le=1, description="Overall validation result (0/1)")
    overall_why: str = Field(..., description="Overall validation explanation")
    fields: Dict[str, ValidationResult] = Field(..., description="Field-level validation results")

# --------------- IO / indexing ---------------

def _norm_path(p: Optional[str]) -> str:
    if not p: return ""
    # normalize separators and case; don't resolve on disk
    return str(p).replace("\\", "/").strip().lower()

def load_chunks_index(chunks_jsonl: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """
    Returns:
      id_map: chunk_id -> chunk
      by_source: normalized source_path -> [chunks...]
    """
    id_map: Dict[str, Dict[str, Any]] = {}
    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(chunks_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            cid = d.get("chunk_id") or d.get("id")
            id_map[cid] = d
            sp = _norm_path(d.get("source_path"))
            by_source[sp].append(d)
    return id_map, by_source

def load_records(recs_jsonl: str, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(recs_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if (model_name is None) or (d.get("model") == model_name):
                out.append(d)
    return out

# --------------- helpers ---------------

def try_parse_number(v: Any) -> Optional[float]:
    if v is None: return None
    if isinstance(v, (int,float)): return float(v)
    s = str(v).strip()
    if s.endswith("%"):
        try: return float(s[:-1]) / 100.0
        except: return None
    s = s.replace(",", "")
    try: return float(s)
    except: return None

def normalize_row(row: Dict[str, Any], spec: ModelSpec) -> Dict[str, Any]:
    out = dict(row)
    for f in spec.fields:
        v = out.get(f.name)
        if f.normalize == "upper" and isinstance(v, str):
            out[f.name] = v.upper().strip()
        elif f.normalize == "percent_to_fraction":
            out[f.name] = try_parse_number(v)
        elif f.normalize == "collapse_whitespace" and isinstance(v, str):
            out[f.name] = " ".join(v.split())
        elif f.normalize == "strip_leading_zeros" and isinstance(v, str):
            out[f.name] = v.lstrip("0") or "0"
        elif f.normalize == "title" and isinstance(v, str):
            out[f.name] = v.title()

    # treat sentinel "no upper limit"
    if "value_max" in out and out.get("value_max") in (0, "0", 0.0, "0.0"):
        out["value_max"] = None
    num = try_parse_number(out.get("value_max"))
    if num is not None and num >= 999_999:
        out["value_max"] = None
    return out

def pick_field_evidence_chunks(
    field: FieldSpec,
    value: Any,
    chunk_ids: List[str],
    id_map: Dict[str, Dict[str, Any]],
    k: int = 8
) -> List[Dict[str, Any]]:
    def score_chunk(ch: Dict[str, Any]) -> float:
        text = (ch.get("text") or "")
        t = text.lower()
        score = 0.0
        for h in (field.hints or []):
            if h and h.lower() in t: score += 1.0
        if isinstance(value, (int, float)):
            s1 = f"{value}".rstrip("0").rstrip(".")
            s2 = f"{value:.2f}".rstrip("0").rstrip(".")
            if s1 and s1 in text: score += 1.0
            if s2 and s2 in text: score += 1.0
        elif isinstance(value, str) and value and value.lower() in t:
            score += 1.2
        for w in ("threshold","limit","minimum","maximum","up to","including","duty","tax"):
            if w in t: score += 0.2
        return score

    ranked: List[Tuple[float, Dict[str, Any]]] = []
    seen = set()
    for cid in chunk_ids:
        ch = id_map.get(cid)
        if not ch or cid in seen: continue
        seen.add(cid)
        ranked.append((score_chunk(ch), ch))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [ch for sc, ch in ranked[:k] if sc > 0]

# --------------- LLM schema/messages ---------------

def build_validation_schema(spec: ModelSpec) -> Dict[str, Any]:
    field_map: Dict[str, Any] = {}
    for f in spec.fields:
        field_map[f.name] = {
            "type":"object",
            "properties":{
                "ok":{"type":"boolean"},
                "why":{"type":"string"},
                "evidence":{
                    "type":"array",
                    "items":{
                        "type":"object",
                        "properties":{
                            "chunk_id":{"type":"string"},
                            "quote":{"type":"string","maxLength":300}
                        },
                        "required":["chunk_id","quote"],
                        "additionalProperties":False
                    }
                },
                "confidence":{"type":"number","minimum":0,"maximum":1}
            },
            "required":["ok","why","evidence","confidence"],
            "additionalProperties":False
        }
    return {
        "type":"object",
        "properties":{
            "binary":{"type":"integer","enum":[0,1]},
            "overall_why":{"type":"string"},
            "fields":{"type":"object","properties":field_map,"required":list(field_map.keys()),"additionalProperties":False}
        },
        "required":["binary","overall_why","fields"],
        "additionalProperties":False
    }

def build_validation_messages(
    spec: ModelSpec,
    record: Dict[str, Any],
    evid_pack: Dict[str, List[Dict[str, Any]]],
    machine_checks: Dict[str, Any],
    defaults: Dict[str, Any]
) -> List[Dict[str,str]]:
    """
    Build validation messages using ValidationPromptBuilder.
    """
    # Convert ModelSpec to data dictionary format for ValidationPromptBuilder
    data_dict = {
        "models": {
            spec.name: {
                "description": getattr(spec, 'description', 'N/A'),
                "primary_key": getattr(spec, 'primary_key', []),
                "fields": [
                    {
                        "name": f.name,
                        "dtype": f.dtype,
                        "required": f.required,
                        "hints": f.hints or [],
                        "enum": f.enum or [],
                        "range": f.range or [],
                        "regex": f.regex or "",
                        "normalize": f.normalize or ""
                    }
                    for f in spec.fields
                ],
                "constraints": getattr(spec, 'constraints', []),
                "evidence_rules": getattr(spec, 'evidence_rules', {})
            }
        },
        "defaults": defaults or {}
    }
    
    # Create PromptBuilder instance for validation
    builder = PromptBuilder(spec.name, data_dict, task_type="validation")
    
    # Add validation-specific rules
    builder.add_rule("Use only provided text; do not infer beyond it.")
    builder.add_rule("Numeric bounds: treat 'up to and including X' as closed at X; next band begins at X+0.01 (currency).")
    builder.add_rule("Open-ended ranges are represented by null (no upper limit).")
    builder.add_rule("If multiple chunks conflict, prefer the most specific/official statement; otherwise mark incorrect.")
    builder.add_rule("Output must strictly follow the JSON schema.")
    
    # Set validation data
    builder.set_record(record)
    builder.set_evidence_pack(evid_pack)
    builder.set_machine_checks(machine_checks)
    
    # Build and return the validation messages
    return builder.build_prompt()

def call_llm_validate_with_instructor(
    client: OpenAI,
    model_name: str,
    spec: ModelSpec,
    record: Dict[str, Any],
    evid_pack: Dict[str, List[Dict[str, Any]]],
    machine_checks: Dict[str, Any],
    defaults: Dict[str, Any]
) -> RecordValidation:
    """Validate records using Instructor with Pydantic models and automatic validation."""
    
    print(f"\n>>> CALLING INSTRUCTOR FOR VALIDATION:")
    print(f"   Model: {model_name}")
    print(f"   Schema: {spec.name}")
    print(f"   Fields to validate: {len(record)}")
    sys.stdout.flush()
    
    # Create Instructor client
    instructor_client = instructor.from_openai(client)
    
    # Build messages using existing PromptBuilder (includes all your text prompts)
    messages = build_validation_messages(spec, record, evid_pack, machine_checks, defaults)
    
    # Use the existing messages as-is - they already contain all your prompts!
    all_messages = messages
    
    print(f"   Number of messages: {len(all_messages)}")
    sys.stdout.flush()
    
    try:
        # Validate using Instructor with automatic validation and retries
        result = instructor_client.chat.completions.create(
            model=model_name,
            messages=all_messages,
            response_model=RecordValidation,
            max_retries=3,
            temperature=0.0
        )
        
        print(f"   ✅ Instructor validation successful!")
        print(f"   Overall result: {'PASS' if result.binary else 'FAIL'}")
        print(f"   Fields validated: {len(result.fields)}")
        sys.stdout.flush()
        
        return result
        
    except Exception as e:
        print(f"   ❌ Instructor validation failed: {e}")
        sys.stdout.flush()
        raise

def convert_instructor_validation_to_legacy(validation: RecordValidation) -> Dict[str, Any]:
    """Convert Instructor validation to legacy format for backward compatibility."""
    legacy_result = {
        "binary": validation.binary,
        "overall_why": validation.overall_why,
        "fields": {}
    }
    
    for field_name, field_result in validation.fields.items():
        legacy_result["fields"][field_name] = {
            "ok": field_result.ok,
            "why": field_result.why,
            "evidence": [
                {
                    "chunk_id": evidence.chunk_id,
                    "quote": evidence.quote
                }
                for evidence in field_result.evidence
            ],
            "confidence": field_result.confidence
        }
    
    return legacy_result

def call_llm_validate(
    client: OpenAI,
    model_name: str,
    spec: ModelSpec,
    record: Dict[str, Any],
    evid_pack: Dict[str, List[Dict[str, Any]]],
    machine_checks: Dict[str, Any],
    defaults: Dict[str, Any]
) -> Dict[str, Any]:
    schema = build_validation_schema(spec)
    messages = build_validation_messages(spec, record, evid_pack, machine_checks, defaults)
    
    # Print validation prompt and messages
    print("\n" + "="*80)
    print("VALIDATION PROMPT:")
    print("="*80)
    for i, msg in enumerate(messages):
        print(f"Message {i+1} ({msg['role']}):")
        print("-" * 60)
        if msg['role'] == 'user' and isinstance(msg['content'], dict):
            # Pretty print JSON user message
            try:
                user_data = json.loads(json.dumps(msg['content'])) if isinstance(msg['content'], dict) else msg['content']
                print(json.dumps(user_data, indent=2, ensure_ascii=False))
            except:
                print(msg['content'])
        else:
            print(msg['content'])
        print("-" * 60)
    print("="*80)
    print("END OF VALIDATION PROMPT")
    print("="*80 + "\n")
    
    print(f"\n>>> CALLING OPENAI API FOR VALIDATION:")
    print(f"   Model: {model_name}")
    print(f"   Temperature: 0")
    print(f"   Response Format: JSON Schema (strict)")
    print(f"   Schema Name: {spec.name}_per_record_validation")
    print(f"   Number of messages: {len(messages)}")
    sys.stdout.flush()  # Ensure output appears immediately
    
    resp = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=messages,
        response_format={
            "type":"json_schema",
            "json_schema":{
                "name": f"{spec.name}_per_record_validation",
                "schema": schema,
                "strict": True
            }
        }
    )
    content = resp.choices[0].message.content or "{}"
    
    # Print the LLM validation response
    print("\n" + "="*80)
    print("VALIDATION RESPONSE RECEIVED:")
    print("="*80)
    try:
        response_data = json.loads(content)
        print(json.dumps(response_data, indent=2, ensure_ascii=False))
    except:
        print(content)
    print("="*80)
    print("END OF VALIDATION RESPONSE")
    print("="*80 + "\n")
    
    try:
        return json.loads(content)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", content)
        if m:
            return json.loads(m.group(0))
        raise

# --------------- deterministic checks ---------------

def deterministic_checks(spec: ModelSpec, record: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    Row = compile_pydantic(spec)
    errs: List[Dict[str, Any]] = []
    try:
        obj = Row(**normalize_row(record, spec))
        row = obj.model_dump()
        cfails = check_constraints(row, spec)
        if cfails:
            errs.append({"type":"constraints_failed","details": cfails})
            return False, errs
        return True, []
    except ValidationError as ve:
        errs.append({"type":"pydantic_error","details": json.loads(ve.json())})
        return False, errs

# --------------- validation core ---------------

def validate_records(
    client: OpenAI,
    dd_path: str,
    model_key: str,
    chunks_path: str,
    recs_path: str,
    out_path: str,
    out_summary: str,
    llm_model: str = "gpt-4o-mini",
    restrict_chunks_to_file: bool = True,
    use_instructor: bool = False,
) -> None:
    dd = load_dictionary(dd_path)
    if model_key not in dd.models:
        raise SystemExit(f"Model '{model_key}' not in dictionary")
    spec = dd.models[model_key]
    defaults = getattr(dd, "defaults", {}) or {}

    id_map, by_source = load_chunks_index(chunks_path)
    recs = load_records(recs_path, model_key)

    results: List[Dict[str, Any]] = []

    for idx, r in enumerate(recs):
        row = r.get("record", {})
        # prefer explicit 'input_file_path' if present; else 'file'
        file_path = r.get("input_file_path") or r.get("file") or ""
        norm_file = _norm_path(file_path)

        # start from extractor-provided chunk_ids if any (and exist)
        chunk_ids: List[str] = [cid for cid in (r.get("chunk_ids") or []) if cid in id_map]

        # restrict to same source file if requested
        if restrict_chunks_to_file:
            if chunk_ids:
                chunk_ids = [cid for cid in chunk_ids if _norm_path(id_map[cid].get("source_path")) == norm_file]
            # if empty after restriction (or none provided), fall back to all chunks from the same file
            if not chunk_ids and norm_file in by_source:
                chunk_ids = [ch.get("chunk_id") or ch.get("id") for ch in by_source[norm_file]]

        # if still empty: no evidence available; proceed but LLM will likely mark incorrect
        # Build per-field evidence packs from the (restricted) chunk set
        evid_pack: Dict[str, List[Dict[str, Any]]] = {}
        for f in spec.fields:
            evid_pack[f.name] = pick_field_evidence_chunks(f, row.get(f.name), chunk_ids, id_map, k=6)

        # Deterministic checks
        ok_det, det_errs = deterministic_checks(spec, row)
        machine = {"deterministic_ok": ok_det, "errors": det_errs}

        # LLM validation
        if use_instructor:
            try:
                # Use Instructor for validation
                print(f"\n[INSTRUCTOR] Validating record {idx+1} with Instructor...")
                instructor_validation = call_llm_validate_with_instructor(
                    client, llm_model, spec, row, evid_pack, machine, defaults
                )
                
                # Convert to legacy format
                llm_out = convert_instructor_validation_to_legacy(instructor_validation)
                print(f"[SUCCESS] Instructor validated record {idx+1}")
                
            except Exception as e:
                print(f"[ERROR] Instructor validation failed for record {idx+1}: {e}")
                print("[FALLBACK] Using legacy validation...")
                
                # Fallback to legacy validation
                llm_out = call_llm_validate(
                    client, llm_model, spec, row, evid_pack, machine, defaults
                )
        else:
            # Use legacy validation
            llm_out = call_llm_validate(
                client, llm_model, spec, row, evid_pack, machine, defaults
            )

        final_binary = 1 if (ok_det and int(llm_out.get("binary",0)) == 1) else 0

        result_item = {
            "index": idx,
            "model": model_key,
            "file": file_path,
            "record": row,
            "binary": final_binary,
            "llm_binary": llm_out.get("binary", 0),
            "deterministic_ok": ok_det,
            "overall_why": llm_out.get("overall_why"),
            "fields": llm_out.get("fields", {}),
            "deterministic_errors": det_errs,
            "used_chunk_ids": chunk_ids
        }
        results.append(result_item)
        print(f"[{idx}] {'PASS' if final_binary==1 else 'FAIL'}  — source={file_path}  chunks={len(chunk_ids)}")

    # Write per-record results
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r["binary"] == 1)
    failed = total - passed

    common_field_fail: Dict[str,int] = defaultdict(int)
    for r in results:
        if r["binary"] == 1: continue
        for fname, fr in (r.get("fields") or {}).items():
            if isinstance(fr, dict) and (fr.get("ok") is False):
                common_field_fail[fname] += 1

    summary = {
        "model": model_key,
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": (passed/total if total else 0.0),
        "common_field_failures": sorted(
            [{"field":k,"count":v} for k,v in common_field_fail.items()],
            key=lambda x: x["count"], reverse=True
        )
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n== SUMMARY ==")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

# --------------- CLI ---------------

def main():
    ap = argparse.ArgumentParser(description="Validator LLM: per-record binary verdict + evidence (file-scoped chunks)")
    ap.add_argument("--dict", required=True, help="Path to data_dictionary.yaml")
    ap.add_argument("--model", required=True, help="Model key (e.g., deminimis_rules_stg)")
    ap.add_argument("--chunks", default="output/all_chunks.jsonl", help="Chunks JSONL (default: output/all_chunks.jsonl)")
    ap.add_argument("--records", default="output/records.jsonl", help="Records JSONL (default: output/records.jsonl)")
    ap.add_argument("--out", default="output/validation.jsonl", help="Per-record validation results")
    ap.add_argument("--out-summary", default="output/validation_summary.json", help="Summary JSON")
    ap.add_argument("--llm", default="gpt-4o-mini", help="LLM model")
    ap.add_argument("--no-restrict-chunks-to-file", action="store_true", help="Disable restricting evidence to the record's source file")
    ap.add_argument("--use-instructor", action="store_true", help="Use Instructor with Pydantic models for validation")
    args = ap.parse_args()

    client = OpenAI()  # reads OPENAI_API_KEY
    validate_records(
        client=client,
        dd_path=args.dict,
        model_key=args.model,
        chunks_path=args.chunks,
        recs_path=args.records,
        out_path=args.out,
        out_summary=args.out_summary,
        llm_model=args.llm,
        restrict_chunks_to_file=not args.no_restrict_chunks_to_file,
        use_instructor=args.use_instructor,
    )

if __name__ == "__main__":
    main()
