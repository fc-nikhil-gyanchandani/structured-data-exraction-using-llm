#!/usr/bin/env python3
"""
Prompt-Pack Generator (meta-prompt) — v1

What this does
--------------
Given:
  • task_brief (what you want the LLM to do)
  • targets (one or more output models as JSON Schemas)
  • field_hints (optional semantic guidance)
  • seed_examples (list of {chunk_text, outputs_by_model} pairs)
  • guardrails/constraints (optional)

It asks OpenAI to PRODUCE a prompt pack:
  {
    "system": "...",
    "developer": "...",
    "user_template": "... {CHUNK_TEXT} ... {TARGETS_JSON} ...",
    "fewshots": [{"user":"...", "assistant":"..."}, ...],
    "notes": "rationale / tuning tips"
  }

You can then use this pack with your generator runner
(e.g., the multi-model extractor we built earlier).

Usage
-----
export OPENAI_API_KEY=...
python prompt_pack_generator.py --demo
OR
python prompt_pack_generator.py --spec-file ./spec.json --out ./prompt_pack.json

spec.json format
----------------
{
  "mode": "multi_model",  // or "single_model"
  "task_brief": "Convert customs text into normalized rows...",
  "targets": { "<model_name>": {<JSON Schema>}, ... },
  "field_hints": ["province_code is 2 letters", "tax_type is one of ..."],
  "constraints": ["Output JSON only", "No hallucinations"],
  "seed_examples": [
    {
      "chunk_text": "raw or csv-ish text ...",
      "outputs_by_model": {
        "<model_name>": [ {row}, {row} ],
        "...": []
      }
    }
  ],
  "style_guidelines": {
    "tone": "precise,minimal",
    "determinism": true,
    "percent_handling": "13% -> 13.0"
  },
  "fewshot_strategy": {
    "synthesize_more": true,     // let LLM create extra fewshots
    "max_fewshots": 2
  }
}
"""

import os
import json
import argparse
import logging
import sys
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('prompt_generator.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# OpenAI SDK
try:
    from openai import OpenAI
except Exception as e:
    raise SystemExit("OpenAI SDK is required. Install: pip install openai") from e


# ---------- Output contract for the prompt pack (so we can validate shape) ----------
PROMPT_PACK_SCHEMA = {
    "type": "object",
    "properties": {
        "system": {"type": "string"},
        "developer": {"type": "string"},
        "user_template": {"type": "string"},
        "fewshots": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "user": {"type": "string"},
                    "assistant": {"type": "string"}
                },
                "required": ["user", "assistant"],
                "additionalProperties": False
            }
        },
        "notes": {"type": "string"}
    },
    "required": ["system", "developer", "user_template", "fewshots"],
    "additionalProperties": False
}


def default_demo_spec() -> Dict[str, Any]:
    """A demo spec aligned with your 'canadian_tax_rates_by_province.csv' case."""
    return {
        "mode": "multi_model",
        "task_brief": "Extract provinces, tax types, and province tax rates from Canadian tax text or tables into normalized rows for data warehousing. Multi-output: provinces_stg, tax_types_stg, province_tax_rates_stg.",
        "targets": {
            "provinces_stg": {
                "type": "object",
                "properties": {
                    "province_code": {"type": "string", "minLength": 2, "maxLength": 2},
                    "name": {"type": "string"}
                },
                "required": ["province_code", "name"],
                "additionalProperties": False
            },
            "tax_types_stg": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": ["GST","HST","PST","QST","RST"]},
                    "tax_code": {"type": "string", "enum": ["GST","HST","PST","QST","RST"]}
                },
                "required": ["name", "tax_code"],
                "additionalProperties": False
            },
            "province_tax_rates_stg": {
                "type": "object",
                "properties": {
                    "province_code": {"type": "string", "minLength": 2, "maxLength": 2},
                    "tax_type": {"type": "string", "enum": ["GST","HST","PST","QST","RST"]},
                    "rate": {"type": "number", "minimum": 0, "maximum": 100}
                },
                "required": ["province_code", "tax_type", "rate"],
                "additionalProperties": False
            }
        },
        "field_hints": [
            "province_code: 2-letter Canadian province/territory code (ON, QC, BC, AB, NS, NB, MB, SK, NL, PE, NT, YT, NU).",
            "tax_type: one of GST, HST, PST, QST, RST.",
            "rate: interpret '13%' as 13.0; where context is percent, '0.13' also means 13.0."
        ],
        "constraints": [
            "No hallucinations. If uncertain, omit row.",
            "Output must be JSON-only when used; for training, include placeholders {CHUNK_TEXT} and {TARGETS_JSON}.",
            "Uppercase province codes and tax types."
        ],
        "style_guidelines": {
            "tone": "precise,minimal",
            "determinism": True,
            "percent_handling": "13% -> 13.0"
        },
        "fewshot_strategy": {
            "synthesize_more": True,
            "max_fewshots": 2
        },
        "seed_examples": [
            {
                "chunk_text": (
                    "Province_Code,Province / Territory,GST_Rate,HST_Rate,PST_Rate\n"
                    "ON,Ontario,,13%,\n"
                    "QC,Quebec,5%,,9.975%\n"
                ),
                "outputs_by_model": {
                    "provinces_stg": [
                        {"province_code": "ON", "name": "Ontario"},
                        {"province_code": "QC", "name": "Quebec"}
                    ],
                    "tax_types_stg": [
                        {"name":"HST","tax_code":"HST"},
                        {"name":"GST","tax_code":"GST"},
                        {"name":"QST","tax_code":"QST"}
                    ],
                    "province_tax_rates_stg": [
                        {"province_code": "ON", "tax_type": "HST", "rate": 13.0},
                        {"province_code": "QC", "tax_type": "GST", "rate": 5.0},
                        {"province_code": "QC", "tax_type": "QST", "rate": 9.975}
                    ]
                }
            },
            {
                "chunk_text": (
                    "British Columbia applies PST at 7% in addition to the federal GST (5%).\n"
                    "Quebec applies QST at 9.975%. HST is not applicable in Quebec.\n"
                ),
                "outputs_by_model": {
                    "provinces_stg": [
                        {"province_code":"BC","name":"British Columbia"},
                        {"province_code":"QC","name":"Quebec"}
                    ],
                    "tax_types_stg": [
                        {"name":"PST","tax_code":"PST"},
                        {"name":"GST","tax_code":"GST"},
                        {"name":"QST","tax_code":"QST"}
                    ],
                    "province_tax_rates_stg": [
                        {"province_code":"BC","tax_type":"PST","rate":7.0},
                        {"province_code":"BC","tax_type":"GST","rate":5.0},
                        {"province_code":"QC","tax_type":"QST","rate":9.975}
                    ]
                }
            }
        ]
    }


# ---------- Meta-prompt templates (we ask the LLM to author the prompt pack) ----------
SYSTEM_META = """\
You are a Prompt Pack Architect. Given a parsing task (targets, examples, and constraints),
produce a high-quality prompt pack for a Generator LLM that extracts structured rows.
Return STRICT JSON for the prompt pack with keys: system, developer, user_template, fewshots, notes.
"""

DEVELOPER_META = """\
Author prompts that:
- Are concise, deterministic, and suitable for chunk-by-chunk extraction.
- Enforce JSON-only outputs using a user_template with placeholders the caller will fill:
  • {CHUNK_TEXT}  - to be replaced at runtime with the raw source chunk.
  • {TARGETS_JSON} - to be replaced at runtime with model_name -> JSON Schema mapping.
- Provide strong guardrails against hallucination and schema drift.
- Few-shots: craft a small number of high-signal examples from provided seeds (and synthesize if asked),
  formatting each as a pair: {"user": "...", "assistant": "..."}.
- Ensure your 'assistant' few-shot outputs are JSON-ONLY and match the target keys/types.
- Do not include code fences in outputs.
- Keep the 'system' and 'developer' messages generalizable across documents of the same task.
- Keep the 'user_template' generic (uses placeholders) and short enough to fit within token budgets.
"""

USER_META_TEMPLATE = """\
TASK BRIEF:
{TASK_BRIEF}

MODE: {MODE}

TARGET MODELS (JSON Schemas):
{TARGETS_JSON}

FIELD HINTS (optional):
{FIELD_HINTS}

CONSTRAINTS (optional):
{CONSTRAINTS}

STYLE GUIDELINES (optional):
{STYLE_GUIDELINES}

FEWSHOT STRATEGY:
- synthesize_more: {SYNTH_MORE}
- max_fewshots: {MAX_FEWSHOTS}

SEED EXAMPLES (each provides a source chunk and the desired outputs_by_model):
{SEED_EXAMPLES}

OUTPUT CONTRACT (STRICT JSON):
{{
  "system": "string",
  "developer": "string",
  "user_template": "string with placeholders {{CHUNK_TEXT}} and {{TARGETS_JSON}}",
  "fewshots": [{{"user":"string","assistant":"string"}}, ...],
  "notes": "optional string"
}}
"""


def build_meta_messages(spec: Dict[str, Any]) -> List[Dict[str, str]]:
    targets_json = json.dumps(spec["targets"], ensure_ascii=False, indent=2)
    field_hints = json.dumps(spec.get("field_hints", []), ensure_ascii=False, indent=2)
    constraints = json.dumps(spec.get("constraints", []), ensure_ascii=False, indent=2)
    style = json.dumps(spec.get("style_guidelines", {}), ensure_ascii=False, indent=2)
    fs = spec.get("fewshot_strategy", {})
    synth_more = fs.get("synthesize_more", True)
    max_fs = fs.get("max_fewshots", 2)

    # Compact seed examples while preserving readability
    seed_examples = json.dumps(spec.get("seed_examples", []), ensure_ascii=False, indent=2)

    user_payload = USER_META_TEMPLATE \
        .replace("{TASK_BRIEF}", spec["task_brief"]) \
        .replace("{MODE}", spec.get("mode", "multi_model")) \
        .replace("{TARGETS_JSON}", targets_json) \
        .replace("{FIELD_HINTS}", field_hints) \
        .replace("{CONSTRAINTS}", constraints) \
        .replace("{STYLE_GUIDELINES}", style) \
        .replace("{SYNTH_MORE}", str(synth_more)) \
        .replace("{MAX_FEWSHOTS}", str(max_fs)) \
        .replace("{SEED_EXAMPLES}", seed_examples)

    return [
        {"role": "system", "content": SYSTEM_META},
        {"role": "developer", "content": DEVELOPER_META},
        {"role": "user", "content": user_payload}
    ]


def generate_prompt_pack(spec: Dict[str, Any], model: str = "gpt-4.1-mini", max_tokens: int = 1800) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key)

    messages = build_meta_messages(spec)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
        max_tokens=max_tokens
    )

    content = resp.choices[0].message.content.strip()
    try:
        pack = json.loads(content)
    except json.JSONDecodeError:
        # fence cleanup fallback
        txt = content.strip().strip("`")
        txt = txt[4:].strip() if txt.lower().startswith("json") else txt
        pack = json.loads(txt)

    # Validate shape (minimal)
    for key in ("system", "developer", "user_template", "fewshots"):
        if key not in pack:
            raise ValueError(f"Prompt pack missing '{key}' field.")
    if not isinstance(pack["fewshots"], list):
        raise ValueError("fewshots must be a list.")
    for fs in pack["fewshots"]:
        if not isinstance(fs, dict) or "user" not in fs or "assistant" not in fs:
            raise ValueError("Each fewshot must have 'user' and 'assistant' strings.")

    # Sanity: placeholders present
    if "{CHUNK_TEXT}" not in pack["user_template"] or "{TARGETS_JSON}" not in pack["user_template"]:
        raise ValueError("user_template must contain {CHUNK_TEXT} and {TARGETS_JSON} placeholders.")

    return pack


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="Generate a Prompt Pack (system/developer/user/fewshots) via meta-prompting.")
    ap.add_argument("--spec-file", type=str, help="Path to JSON spec with task_brief, targets, seed_examples, etc.")
    ap.add_argument("--model", type=str, default="gpt-4.1-mini")
    ap.add_argument("--out", type=str, help="Where to write the generated pack (JSON). Default: stdout")
    ap.add_argument("--demo", action="store_true", help="Use built-in Canadian tax demo spec.")
    ap.add_argument("--dry-run", action="store_true", help="Show the meta-prompt that would be sent to OpenAI without calling the API.")
    args = ap.parse_args()

    if args.demo:
        spec = default_demo_spec()
    elif args.spec_file:
        spec = load_json(args.spec_file)
    else:
        ap.error("Provide --spec-file or use --demo")

    if args.dry_run:
        logger.info("=== META-PROMPT THAT WOULD BE SENT TO OPENAI ===")
        messages = build_meta_messages(spec)
        for i, msg in enumerate(messages):
            logger.info(f"--- {msg['role'].upper()} MESSAGE {i+1} ---")
            logger.info(msg['content'])
            logger.info("")
        logger.info("=== END META-PROMPT ===")
        return

    pack = generate_prompt_pack(spec, model=args.model)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(pack, f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote prompt pack → {args.out}")
    else:
        logger.info(json.dumps(pack, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
