# extractor.py  — universal-prompt, file-union evidence, country-agnostic
from __future__ import annotations

import os, json, argparse, re, pathlib, sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict

from pydantic import ValidationError, BaseModel, Field, field_validator
from openai import OpenAI
import instructor

# ---- local compiler utils (unchanged contracts) ----
from data_dict_compiler import (
    load_dictionary,
    compile_json_schema as compile_row_schema,   # row-level schema (for validation hint)
    compile_pydantic,
    check_constraints,
    ModelSpec, FieldSpec
)
from prompt_builder import PromptBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('extractor.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ---------------- PYDANTIC MODELS FOR INSTRUCTOR ----------------

class FieldEvidence(BaseModel):
    """Evidence for a field with chunk reference and snippet."""
    chunk_id: str = Field(..., description="Chunk ID containing the evidence")
    snippet: str = Field(..., max_length=300, description="Relevant text snippet")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score for this evidence")

class DeminimisRecord(BaseModel):
    """Single de minimis threshold record with enhanced validation."""
    destination_country_iso: str = Field(..., pattern=r'^[A-Z]{2}$', description="ISO2 country code")
    origin_band_code: str = Field(..., description="Origin band code")
    value_min: float = Field(..., gt=0, description="Minimum value in CAD")
    value_max: Optional[float] = Field(None, gt=0, description="Maximum value in CAD, null for open-ended")
    currency_code: str = Field(default="CAD", description="Currency code")
    duty_applicable: bool = Field(..., description="Whether customs duties apply")
    tax_applicable: bool = Field(..., description="Whether taxes apply")
    note: Optional[str] = Field(None, max_length=500, description="Additional notes or rule details")
    
    # Evidence tracking
    evidence: List[FieldEvidence] = Field(default_factory=list, description="Evidence supporting this record")
    
    @field_validator('value_max')
    @classmethod
    def validate_value_range(cls, v, info):
        """Ensure value_max > value_min when both are provided."""
        if v is not None and 'value_min' in info.data:
            if v <= info.data['value_min']:
                raise ValueError('value_max must be greater than value_min')
        return v

class PolicyTier(BaseModel):
    """Policy tier definition with evidence tracking."""
    lower: float = Field(..., ge=0, description="Lower bound of the tier")
    upper: Optional[float] = Field(None, ge=0, description="Upper bound, null for open-ended")
    duty: bool = Field(..., description="Duty applicable in this tier")
    tax: bool = Field(..., description="Tax applicable in this tier")
    evidence_chunk_id: str = Field(..., description="Chunk ID containing evidence for this tier")

class PolicySignature(BaseModel):
    """Policy signature with tier definitions."""
    scope_label: str = Field(..., description="Label describing the policy scope")
    tiers: List[PolicyTier] = Field(..., min_items=1, description="List of policy tiers")

class DeminimisExtraction(BaseModel):
    """Complete de minimis extraction result with enhanced validation."""
    model: str = Field(..., description="Model name")
    records: List[DeminimisRecord] = Field(..., min_items=1, description="Extracted records")
    policy_signature: PolicySignature = Field(..., description="Policy signature with tiers")
    notes: List[str] = Field(default_factory=list, description="Additional notes")

# ---------------- IO ----------------

def load_chunks(jsonl_path: str) -> List[Dict[str, Any]]:
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks

def group_by_file(chunks: List[Dict[str, Any]], spec: ModelSpec, override_source_filter: Optional[List[str]]=None, files_list: Optional[List[str]]=None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group chunks by source_path. Apply files_list if provided, else override_source_filter, else no filter.
    """
    g: Dict[str, List[Dict[str, Any]]] = {}
    
    # Priority: files_list > override_source_filter > no filter
    source_filter = None
    if files_list:
        source_filter = files_list
    elif override_source_filter is not None:
        source_filter = override_source_filter

    for ch in chunks:
        sp = ch.get("source_path") or ""
        if source_filter:
            if not any(pat in sp for pat in source_filter):
                continue
        g.setdefault(sp, []).append(ch)
    return g

# ---------------- Batching for token limits ----------------

def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars/token"""
    return max(1, len(text) // 4)

def create_chunk_batches(chunks: List[Dict[str, Any]], max_tokens_per_batch: int = 100_000) -> List[List[Dict[str, Any]]]:
    batches: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    t_used = 0
    for ch in chunks:
        t = estimate_tokens(ch.get("text",""))
        if current and (t_used + t > max_tokens_per_batch):
            batches.append(current)
            current = [ch]
            t_used = t
        else:
            current.append(ch)
            t_used += t
    if current:
        batches.append(current)
    return batches

# ---------------- helpers ----------------

def try_parse_number(v: Any) -> Optional[float]:
    if v is None: return None
    if isinstance(v, (int, float)): return float(v)
    s = str(v).strip()
    if not s:
        return None
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except:
            return None
    s = s.replace(",", "")
    try:
        return float(s)
    except:
        return None

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
    return out

def validate_rows(rows: List[Dict[str, Any]], spec: ModelSpec) -> List[Dict[str, Any]]:
    RowModel = compile_pydantic(spec)
    valid: List[Dict[str, Any]] = []
    for r in rows:
        r = normalize_row(r, spec)
        try:
            obj = RowModel(**r)
            row = obj.model_dump()
            cfails = check_constraints(row, spec)
            if not cfails:
                valid.append(row)
        except ValidationError:
            pass
    return valid

# ---------------- Structured Output schema (LLM response) ----------------

def value_type_for_json_schema(dtype: str) -> Dict[str, Any]:
    if dtype in ("number","decimal"):
        return {"type": ["number","string"]}
    if dtype == "integer":
        return {"type": ["integer","number","string"]}
    if dtype == "boolean":
        return {"type": ["boolean","string"]}
    if dtype in ("date","datetime"):
        return {"type": "string"}
    return {"type": "string"}

def build_extractor_response_schema(spec: ModelSpec) -> Dict[str, Any]:
    field_objs: Dict[str, Any] = {}
    for f in spec.fields:
        value_schema = value_type_for_json_schema(f.dtype)
        if f.enum:
            value_schema = {"anyOf": [value_schema, {"type":"string","enum": f.enum}]}
        field_objs[f.name] = {
            "type": ["object","null"],
            "properties": {
                "value": value_schema,
                "confidence": {"type":"number","minimum":0,"maximum":1},
                "evidence": {
                    "type":"array",
                    "items": {
                        "type":"object",
                        "properties": {
                            "chunk_id": {"type":"string"},
                            "snippet": {"type":"string", "maxLength": 300}
                        },
                        "required": ["chunk_id","snippet"],
                        "additionalProperties": False
                    }
                },
                "notes": {"type":["string","null"]}
            },
            "required": ["value","confidence","evidence","notes"],
            "additionalProperties": False
        }

    # Optional band signature helper (non-blocking)
    policy_signature = {
        "type":"object",
        "properties":{
            "scope_label":{"type":"string"},
            "tiers":{
                "type":"array",
                "items":{
                    "type":"object",
                    "properties":{
                        "lower":{"type":["number","null"]},
                        "upper":{"type":["number","null"]},
                        "duty":{"type":"boolean"},
                        "tax":{"type":"boolean"},
                        "evidence_chunk_id":{"type":"string"}
                    },
                    "required":["lower","upper","duty","tax","evidence_chunk_id"],
                    "additionalProperties": False
                }
            }
        },
        "required":["tiers","scope_label"],
        "additionalProperties": False
    }

    return {
        "type":"object",
        "properties":{
            "model":{"type":"string"},
            "records":{
                "type":"array",
                "items":{
                    "type":"object",
                    "properties": field_objs,
                    "additionalProperties": False,
                    "required": [f.name for f in spec.fields]
                }
            },
            "policy_signature": policy_signature,
            "notes":{"type":"array","items":{"type":"string"}}
        },
        "required":["model","records","policy_signature","notes"],
        "additionalProperties": False
    }

# ---------------- Prompt building ----------------

def trim(s: str, max_chars: int) -> str:
    return s if len(s) <= max_chars else s[:max_chars]

def _has_field(spec: ModelSpec, name: str) -> bool:
    return any(f.name == name for f in spec.fields)

def build_messages(
    spec: ModelSpec,
    chunks: List[Dict[str, Any]],
    row_json_schema: Dict[str, Any],
    defaults: Dict[str, Any]
) -> List[Dict[str,str]]:
    """
    Universal, country-agnostic prompt using PromptBuilder. Works for any source files.
    """
    # Convert ModelSpec to data dictionary format for PromptBuilder
    data_dict = {
        "models": {
            spec.name: {
                "description": getattr(spec, 'description', 'N/A'),
                "primary_key": getattr(spec, 'primary_key', []),
                "key_fields": getattr(spec, 'key_fields', []),
                "domain_context": getattr(spec, 'domain_context', ''),
                "extraction_perspective": getattr(spec, 'extraction_perspective', ''),
                "fields": [
                    {
                        "name": f.name,
                        "dtype": f.dtype,
                        "description": getattr(f, 'description', ''),
                        "required": f.required,
                        "hints": f.hints or [],
                        "extraction_rules": f.extraction_rules or [],
                        "examples": f.examples or [],
                        "normalize": f.normalize or "",
                        "regex": f.regex or "",
                        "enum": f.enum or [],
                        "range": f.range or [],
                        "units": f.units or "",
                        "notes": f.notes or ""
                    }
                    for f in spec.fields
                ],
                "business_rules": getattr(spec, 'business_rules', []),
                "record_grouping_logic": getattr(spec, 'record_grouping_logic', ''),
                "document_context": getattr(spec, 'document_context', {}),
                "extraction_focus": getattr(spec, 'extraction_focus', []),
                "source_path_filter": getattr(spec, 'source_path_filter', []),
                "constraints": getattr(spec, 'constraints', [])
            }
        },
        "defaults": defaults or {}
    }
    
    # Create PromptBuilder instance for extraction
    builder = PromptBuilder(spec.name, data_dict, task_type="extraction")
    
    # Add comprehensive extraction rules
    builder.add_rule("DOMAIN AGNOSTIC RULES:")
    builder.add_rule("Extract ONLY information explicitly stated or clearly implied in the provided chunks")
    builder.add_rule("Never use external knowledge beyond what's in the chunks")
    builder.add_rule("If the data dictionary specifies a business perspective, extract from that viewpoint consistently")
    builder.add_rule("Create one record for each distinct entity/rule/threshold/item as defined by the primary key")
    
    builder.add_rule("FIELD EXTRACTION:")
    builder.add_rule("For each field, follow the extraction_rules provided in the data dictionary")
    builder.add_rule("Use the hints array to identify field values in natural language text")
    builder.add_rule("Apply validation_patterns if specified")
    builder.add_rule("When text is ambiguous, prefer the interpretation that aligns with the business_rules")
    
    builder.add_rule("RECORD CREATION LOGIC:")
    builder.add_rule("Follow the record_grouping_logic from data dictionary to determine record boundaries")
    builder.add_rule("Ensure primary key uniqueness - each record must have a unique combination of primary key fields")
    builder.add_rule("If business_rules specify relationships, maintain those constraints across records")
    
    builder.add_rule("CONFIDENCE AND EVIDENCE:")
    builder.add_rule("Higher confidence (0.95) for explicitly stated facts")
    builder.add_rule("Medium confidence (0.80) for clearly implied information")
    builder.add_rule("Lower confidence (0.55) for information requiring interpretation")
    builder.add_rule("Always provide evidence snippets that directly support each extracted value")
    
    builder.add_rule("HANDLING AMBIGUITY:")
    builder.add_rule("When multiple interpretations are possible, choose the one most consistent with the document_context")
    builder.add_rule("If extraction_focus specifies priorities, follow those guidelines")
    builder.add_rule("When in doubt, prefer null values with explanatory notes rather than guessing")
    
    builder.add_rule("OUTPUT: Strictly follow the JSON Schema below.")
    
    # Add adaptive banding logic
    builder.add_rule("ADAPTIVE BANDING LOGIC:")
    builder.add_rule("IF the data dictionary indicates threshold/range-based data:")
    builder.add_rule("- Identify ALL distinct thresholds mentioned in chunks")
    builder.add_rule("- Create separate records for each range with different rule applications")
    builder.add_rule("- Ensure no gaps in coverage unless explicitly stated")
    builder.add_rule("- Handle open-ended ranges as specified in field extraction_rules")
    
    builder.add_rule("IF the data dictionary indicates categorical/enumerated data:")
    builder.add_rule("- Extract each distinct category/item as a separate record")
    builder.add_rule("- Group related items according to the record_grouping_logic")
    
    builder.add_rule("IF the data dictionary indicates relational data:")
    builder.add_rule("- Maintain parent-child or hierarchical relationships as specified")
    builder.add_rule("- Ensure referential integrity across related records")
    
    # Add quality assurance rules
    builder.add_rule("QUALITY ASSURANCE:")
    builder.add_rule("Validate each record against the business_rules from data dictionary")
    builder.add_rule("Check that all required fields are populated or explicitly marked null")
    builder.add_rule("Ensure extracted values match validation_patterns where specified")
    builder.add_rule("Verify that the extraction_perspective is consistently applied")
    builder.add_rule("Flag any conflicts between chunks in the notes field")
    
    # Add specific banding hints if schema looks like thresholds
    if _has_field(spec, "value_min") and _has_field(spec, "value_max") and _has_field(spec, "duty_applicable") and _has_field(spec, "tax_applicable"):
        builder.add_rule("SPECIFIC BANDING FOR THRESHOLDS:")
        builder.add_rule("Identify ALL threshold tiers present; never merge tiers with different duty/tax flags.")
        builder.add_rule("Each tier becomes one record.")
        builder.add_rule("If wording says 'up to and including X', the next band begins at X+0.01 for currency.")
        builder.add_rule("Represent open-ended upper bounds as null.")
    
    # Add chunks
    for ch in chunks:
        builder.add_chunk(
            ch.get("chunk_id", ch.get("id", "")), 
            trim(ch.get("text", ""), 1200)
        )
    
    # Set the required schemas for PromptBuilder
    builder.set_json_schema_for_response(build_extractor_response_schema(spec))
    builder.set_row_json_schema_hint(row_json_schema)
    
    # Add additional instructions
    builder.add_instruction("Analyze ALL provided chunks in this batch.")
    builder.add_instruction("Do not copy large text; include small evidence snippets.")
    
    # Build the complete prompt using PromptBuilder
    return builder.build_prompt()

# ---------------- LLM call ----------------

def call_llm_generate_with_instructor(
    client: OpenAI,
    model: str,
    spec: ModelSpec,
    chunks: List[Dict[str, Any]],
    defaults: Dict[str, Any]
) -> DeminimisExtraction:
    """Extract records using Instructor with Pydantic models and automatic validation."""
    
    logger.info(f"CALLING INSTRUCTOR FOR EXTRACTION:")
    logger.info(f"   Model: {model}")
    logger.info(f"   Schema: {spec.name}")
    logger.info(f"   Number of chunks: {len(chunks)}")
    
    # Create Instructor client
    instructor_client = instructor.from_openai(client)
    
    # Build messages using existing PromptBuilder (includes all your text prompts)
    row_json_schema = compile_row_schema(spec)
    messages = build_messages(spec, chunks, row_json_schema, defaults)
    
    # Use the existing messages as-is - they already contain all your prompts!
    all_messages = messages
    
    logger.info(f"   Number of messages: {len(all_messages)}")
    
    try:
        # Extract using Instructor with automatic validation and retries
        result = instructor_client.chat.completions.create(
            model=model,
            messages=all_messages,
            response_model=DeminimisExtraction,
            max_retries=3,
            temperature=0.0
        )
        
        logger.info(f"   ✅ Instructor extraction successful!")
        logger.info(f"   Records extracted: {len(result.records)}")
        logger.info(f"   Policy tiers: {len(result.policy_signature.tiers)}")
        
        return result
        
    except Exception as e:
        logger.error(f"   ❌ Instructor extraction failed: {e}")
        raise

def convert_instructor_to_legacy_format(extraction: DeminimisExtraction, source_file: str) -> List[Dict[str, Any]]:
    """Convert Instructor extraction to legacy format for backward compatibility."""
    output_records = []
    
    for record in extraction.records:
        # Get all chunk IDs from evidence
        chunk_ids = list(set(evidence.chunk_id for evidence in record.evidence))
        
        output_record = {
            "model": extraction.model,
            "file": source_file,
            "record": {
                "destination_country_iso": record.destination_country_iso,
                "origin_band_code": record.origin_band_code,
                "value_min": record.value_min,
                "value_max": record.value_max,
                "currency_code": record.currency_code,
                "duty_applicable": record.duty_applicable,
                "tax_applicable": record.tax_applicable,
                "note": record.note
            },
            "chunk_ids": chunk_ids
        }
        output_records.append(output_record)
    
    return output_records

def call_llm_generate(
    client: OpenAI,
    model: str,
    spec: ModelSpec,
    chunks: List[Dict[str, Any]],
    defaults: Dict[str, Any]
) -> Dict[str, Any]:
    row_json_schema = compile_row_schema(spec)   # advisory only
    extraction_schema = build_extractor_response_schema(spec)
    messages = build_messages(spec, chunks, row_json_schema, defaults)

    # Log the complete prompt being sent to LLM
    logger.info("="*80)
    logger.info("LLM PROMPT BEING SENT:")
    logger.info("="*80)
    for i, msg in enumerate(messages, 1):
        logger.info(f"\n--- MESSAGE {i}: {msg['role'].upper()} ---")
        if msg['role'] == 'user':
            # Pretty print JSON user message
            try:
                user_data = json.loads(msg['content'])
                logger.info(json.dumps(user_data, indent=2, ensure_ascii=False))
            except:
                logger.info(msg['content'])
        else:
            logger.info(msg['content'])
        logger.info("-" * 60)
    logger.info("="*80)
    logger.info("END OF PROMPT")
    logger.info("="*80)

    logger.info(f"CALLING OPENAI API:")
    logger.info(f"   Model: {model}")
    logger.info(f"   Temperature: 0")
    logger.info(f"   Response Format: JSON Schema (strict)")
    logger.info(f"   Schema Name: {spec.name}_extraction")
    logger.info(f"   Number of messages: {len(messages)}")
    
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": f"{spec.name}_extraction",
                "schema": extraction_schema,
                "strict": True
            }
        }
    )
    content = resp.choices[0].message.content or "{}"
    
    # Log the LLM response
    logger.info("="*80)
    logger.info("LLM RESPONSE RECEIVED:")
    logger.info("="*80)
    try:
        response_data = json.loads(content)
        logger.info(json.dumps(response_data, indent=2, ensure_ascii=False))
    except:
        logger.info(content)
    logger.info("="*80)
    logger.info("END OF RESPONSE")
    logger.info("="*80)
    
    try:
        return json.loads(content)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", content)
        if m:
            return json.loads(m.group(0))
        raise

# ---------------- Glue: LLM JSON -> plain rows ----------------

def records_from_llm_json(resp: Dict[str, Any], spec: ModelSpec) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rec in resp.get("records", []):
        out_row: Dict[str, Any] = {}
        for f in spec.fields:
            cell = rec.get(f.name) or {}
            out_row[f.name] = cell.get("value")
        rows.append(out_row)
    return rows

# ---------------- defaults loader ----------------

def load_defaults(args) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    # inline JSON
    if getattr(args, "defaults_json", None):
        d.update(json.loads(args.defaults_json))
    # file (json or yaml)
    if getattr(args, "defaults_file", None):
        p = Path(args.defaults_file)
        if p.suffix.lower() in [".yaml",".yml"]:
            import yaml
            d.update(yaml.safe_load(p.read_text(encoding="utf-8")))
        else:
            d.update(json.loads(p.read_text(encoding="utf-8")))
    # env overrides (handy in CI/CD)
    d.setdefault("country", os.getenv("LLM_DEFAULT_COUNTRY"))
    d.setdefault("currency", os.getenv("LLM_DEFAULT_CURRENCY"))
    # prune Nones
    return {k:v for k,v in d.items() if v is not None}

# ---------------- Main per-file extraction with batching ----------------

def extract_file_with_llm(
    client: OpenAI,
    model: str,
    file_chunks: List[Dict[str, Any]],
    spec: ModelSpec,
    defaults: Dict[str, Any],
    max_tokens_per_batch: int = 100_000,
    chunk_ids_mode: str = "file"   # "file" => union of all batch chunks; "batch" => per-batch
) -> Tuple[List[Dict[str, Any]], List[List[str]]]:
    """
    Returns: (valid_rows, chunk_ids_lists)
    chunk_ids_lists aligns with valid_rows; each is either the union across the file (recommended),
    or the batch chunks if mode='batch'.
    """
    all_valid_rows: List[Dict[str, Any]] = []
    all_chunk_ids_lists: List[List[str]] = []

    batches = create_chunk_batches(file_chunks, max_tokens_per_batch)

    # union for file-level evidence
    file_level_union_chunk_ids: set[str] = set()

    logger.info(f"   [BATCH] {len(file_chunks)} chunks -> {len(batches)} batch(es)")
    logger.info(f"   [TOTAL CHUNKS] Processing ALL {len(file_chunks)} chunks from file in {len(batches)} batches")
    
    for b_ix, batch in enumerate(batches, start=1):
        batch_chunk_ids = [c.get("chunk_id", c.get("id", "")) for c in batch]
        file_level_union_chunk_ids.update(batch_chunk_ids)

        logger.info(f"   [PROCESS] Batch {b_ix}/{len(batches)} with {len(batch)} chunks")
        logger.info(f"   [CHUNKS] Chunk IDs in this batch: {[c.get('chunk_id', c.get('id', '')) for c in batch]}")
        logger.info(f"   [PROGRESS] Processing chunks {len(file_level_union_chunk_ids)}/{len(file_chunks)} total chunks so far")
        
        llm_json = call_llm_generate(client, model, spec, batch, defaults)
        rows_raw = records_from_llm_json(llm_json, spec)
        valid = validate_rows(rows_raw, spec)
        
        logger.info(f"   [BATCH RESULT] Extracted {len(valid)} records from batch {b_ix}")

        if chunk_ids_mode == "batch":
            for _ in valid:
                all_chunk_ids_lists.append(batch_chunk_ids)

        all_valid_rows.extend(valid)
    
    logger.info(f"   [COMPLETE] Processed ALL {len(file_level_union_chunk_ids)} chunks from file, extracted {len(all_valid_rows)} total records")
    
    # Validation: Ensure we processed all chunks
    if len(file_level_union_chunk_ids) != len(file_chunks):
        logger.warning(f"   [WARNING] Chunk count mismatch! Expected {len(file_chunks)} chunks, processed {len(file_level_union_chunk_ids)} chunks")
        missing_chunks = set(c.get("chunk_id", c.get("id", "")) for c in file_chunks) - file_level_union_chunk_ids
        if missing_chunks:
            logger.warning(f"   [MISSING] Unprocessed chunks: {list(missing_chunks)}")
    else:
        logger.info(f"   [VALIDATION] ✅ All {len(file_chunks)} chunks processed successfully")

    if chunk_ids_mode == "file":
        union_sorted = sorted(file_level_union_chunk_ids)
        all_chunk_ids_lists = [union_sorted for _ in all_valid_rows]

    return all_valid_rows, all_chunk_ids_lists

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(
        description="Schema-driven universal extractor (LLM Structured Outputs)"
    )
    ap.add_argument("--dict", required=True, help="Path to data_dictionary.yaml")
    ap.add_argument("--model", required=True, help="Target model name from YAML")
    ap.add_argument("--chunks", default="output/all_chunks.jsonl", help="Input chunks JSONL")
    ap.add_argument("--out-records", default="output/records.jsonl")
    ap.add_argument("--llm", default="gpt-4o-mini", help="LLM model name")
    ap.add_argument("--source-filter", help="Override source path filter (comma-separated patterns)")
    ap.add_argument("--files", nargs="+", help="List of specific files to process (overrides source_path_filter)")
    ap.add_argument("--max-tokens-per-batch", type=int, default=100_000)
    # new: runtime defaults and chunk-id strategy
    ap.add_argument("--defaults-json", help='Inline JSON defaults, e.g. \'{"country":"CA","currency":"CAD"}\'')
    ap.add_argument("--defaults-file", help="Path to a JSON/YAML file with defaults")
    ap.add_argument("--use-instructor", action="store_true", help="Use Instructor with Pydantic models for extraction")
    ap.add_argument("--chunk-ids-mode", choices=["file","batch"], default="file",
                    help="Attach union of all file chunks (file) or only per-batch chunks (batch)")
    args = ap.parse_args()

    client = OpenAI()  # uses OPENAI_API_KEY
    dd = load_dictionary(args.dict)
    if args.model not in dd.models:
        raise SystemExit(f"Model '{args.model}' not found. Available: {list(dd.models.keys())}")
    spec = dd.models[args.model]

    # optional source filter override
    override_source_filter = None
    if args.source_filter:
        override_source_filter = [p.strip() for p in args.source_filter.split(",")]
        logger.info(f"🔍 Using source filter override: {override_source_filter}")

    # files list takes precedence over source filter
    files_list = None
    if args.files:
        files_list = args.files
        logger.info(f"📁 Using files list: {files_list}")

    defaults = load_defaults(args)
    if defaults:
        logger.info(f"⚙️  Defaults in use: {defaults}")

    chunks = load_chunks(args.chunks)
    by_file = group_by_file(chunks, spec, override_source_filter, files_list)

    logger.info(f"Files to process for model '{args.model}': {len(by_file)}")
    for fp, arr in by_file.items():
        logger.info(f"   - {fp}: {len(arr)} chunks")

    all_rows: List[Dict[str, Any]] = []

    for file_path, file_chunks in by_file.items():
        if not file_chunks:
            continue

        logger.info(f"📁 [FILE] Processing {file_path} with {len(file_chunks)} chunks")
        
        if args.use_instructor:
            # Use Instructor for extraction
            logger.info(f"[INSTRUCTOR] Processing {file_path} with Instructor...")
            try:
                extraction = call_llm_generate_with_instructor(
                    client=client,
                    model=args.llm,
                    spec=spec,
                    chunks=file_chunks,
                    defaults=defaults
                )
                
                # Convert to legacy format
                instructor_rows = convert_instructor_to_legacy_format(extraction, file_path)
                all_rows.extend(instructor_rows)
                
                logger.info(f"[SUCCESS] Instructor extracted {len(instructor_rows)} records from {file_path}")
                logger.info(f"📊 [SUMMARY] File {file_path}: {len(file_chunks)} chunks → {len(instructor_rows)} records")
                
            except Exception as e:
                logger.error(f"[ERROR] Instructor extraction failed for {file_path}: {e}")
                logger.info("[FALLBACK] Using legacy extraction...")
                
                # Fallback to legacy extraction
                rows, chunk_ids_lists = extract_file_with_llm(
                    client=client,
                    model=args.llm,
                    file_chunks=file_chunks,
                    spec=spec,
                    defaults=defaults,
                    max_tokens_per_batch=args.max_tokens_per_batch,
                    chunk_ids_mode=args.chunk_ids_mode
                )

                for i, r in enumerate(rows):
                    record_data = {
                        "model": spec.name,
                        "file": file_path,
                        "record": r,
                        "chunk_ids": chunk_ids_lists[i] if i < len(chunk_ids_lists) else []
                    }
                    all_rows.append(record_data)
        else:
            # Use legacy extraction
            rows, chunk_ids_lists = extract_file_with_llm(
                client=client,
                model=args.llm,
                file_chunks=file_chunks,
                spec=spec,
                defaults=defaults,
                max_tokens_per_batch=args.max_tokens_per_batch,
                chunk_ids_mode=args.chunk_ids_mode
            )

            for i, r in enumerate(rows):
                record_data = {
                    "model": spec.name,
                    "file": file_path,
                    "record": r,
                    "chunk_ids": chunk_ids_lists[i] if i < len(chunk_ids_lists) else []
                }
                all_rows.append(record_data)
            
            logger.info(f"📊 [SUMMARY] File {file_path}: {len(file_chunks)} chunks → {len(rows)} records")

    Path(args.out_records).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_records, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"[SUCCESS] Wrote {len(all_rows)} records -> {args.out_records}")

if __name__ == "__main__":
    main()
