import json
from typing import List, Dict, Any, Optional


class PromptBuilder:
    """
    A unified prompt builder for schema-driven extraction and validation tasks.
    Generates OpenAI-compatible messages from a YAML-based data dictionary,
    supporting both extraction and validation use cases.

    Key upgrades:
      - System prompt includes normalization, banding, confidence ladder,
        evidence object shape, null-but-required guidance, and sorting.
      - Evidence now uses [{chunk_id, snippet}] (not plain IDs).
      - USER payload is a single JSON object with:
          data_dictionary, json_schema_for_response, row_json_schema_hint,
          defaults, all_chunks, instructions.
      - Validation mode expects evidence objects and can carry machine checks.
    """

    def __init__(
        self,
        model_name: str,
        data_dict: Dict[str, Any],
        task_type: str = "extraction",
    ):
        self.model_name = model_name
        # Pull the model block once
        self.model_spec: Dict[str, Any] = data_dict.get("models", {}).get(model_name, {})
        self.fields: List[Dict[str, Any]] = self.model_spec.get("fields", [])
        self.constraints: List[Dict[str, Any]] = self.model_spec.get("constraints", [])
        self.rules: List[str] = []          # extra, caller-provided rules
        self.examples: List[str] = []       # optional few-shot examples (strings)
        self.chunks: List[Dict[str, str]] = []
        self.defaults: Dict[str, Any] = data_dict.get("defaults", {})
        self.task_type = task_type

        # New: externally-supplied schemas & instructions
        self.json_schema_for_response: Optional[Dict[str, Any]] = None
        self.row_json_schema_hint: Optional[Dict[str, Any]] = None
        self.extra_instructions: List[str] = []

        # Validation-specific attributes
        self.record: Dict[str, Any] = {}
        self.evidence_pack: Dict[str, List[Dict[str, Any]]] = {}
        self.machine_checks: Dict[str, Any] = {}

    # ---------------------------
    # Configuration helpers
    # ---------------------------
    def add_rule(self, rule: str):
        self.rules.append(rule)

    def add_example(self, example: str):
        self.examples.append(example)

    def add_chunk(self, chunk_id: str, text: str):
        self.chunks.append({"chunk_id": chunk_id, "text": text})

    def set_json_schema_for_response(self, schema: Dict[str, Any]):
        self.json_schema_for_response = schema

    def set_row_json_schema_hint(self, row_schema_hint: Dict[str, Any]):
        self.row_json_schema_hint = row_schema_hint

    def add_instruction(self, instruction: str):
        self.extra_instructions.append(instruction)

    # Validation-specific methods
    def set_record(self, record: Dict[str, Any]):
        self.record = record

    def set_evidence_pack(self, evidence_pack: Dict[str, List[Dict[str, Any]]]):
        """
        evidence_pack shape:
          {
            "field_name": [
              {"chunk_id": "id", "text": "full evidence text"},
              ...
            ],
            ...
          }
        """
        self.evidence_pack = evidence_pack

    def set_machine_checks(self, machine_checks: Dict[str, Any]):
        self.machine_checks = machine_checks

    # ---------------------------
    # Internal formatters
    # ---------------------------
    def _format_field_description(self) -> str:
        """Human-readable field list for the data_dictionary block."""
        return "\n".join([
            f"- `{f['name']}` ({f['dtype']}){' [REQUIRED]' if f.get('required') else ''} | "
            f"Hints: {', '.join(f.get('hints', []))}"
            for f in self.fields
        ])

    def _data_dictionary_block(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "description": self.model_spec.get("description", "N/A"),
            "primary_key": self.model_spec.get("primary_key", []),
            "fields": {
                f["name"]: {
                    "type": f.get("dtype", "string"),
                    "hints": f.get("hints", []),
                    "required": bool(f.get("required", False)),
                } for f in self.fields
            },
        }

    # ---------------------------
    # Public build API
    # ---------------------------
    def build_prompt(self) -> List[Dict[str, str]]:
        if self.task_type == "validation":
            return self.build_validation_prompt()
        return self.build_extraction_prompt()

    # ---------------------------
    # Extraction
    # ---------------------------
    def build_extraction_prompt(self) -> List[Dict[str, str]]:
        # --- SYSTEM: single source of truth ---
        system_msg = (
            "You are a schema-driven extraction model. Extract ONLY from provided chunks. "
            "Never invent or use outside knowledge. If a field is not explicitly supported "
            "by evidence, set that field to null (but include the field key). Output strict JSON only "
            "(no markdown, no commentary).\n\n"
            "Quality contracts:\n"
            "- No hallucinations.\n"
            "- Obey the provided JSON Schema exactly.\n"
            "- For every non-null field include 1–3 evidence objects with the shape "
            "{chunk_id, snippet} where snippet is a ≤120 char direct quote.\n"
            "- If constraints appear violated, still return the best structured result and add a note in top-level 'notes'.\n\n"
            "Normalization:\n"
            "- Currency codes uppercase ISO-4217 (e.g., 'CAD'); numeric values are numbers (strip $ and commas).\n"
            "- Range edges: if text says 'up to and including X', the next band starts at X + 0.01 (minor unit).\n"
            "- Open upper bounds: use null.\n"
            "- Sort records by value_min ascending.\n\n"
            "Confidence ladder:\n"
            "- 0.95: stated verbatim in one chunk.\n"
            "- 0.80: clearly implied across adjacent lines/chunks.\n"
            "- 0.55: simple normalization (strip symbols) but wording unambiguous.\n\n"
            "Banding rules:\n"
            "- Identify ALL threshold tiers. Do not merge tiers with different duty/tax flags.\n"
            "- One record per tier.\n"
            "- If scope (e.g., courier vs postal) is mixed, include tiers for the primary scope and explain in 'notes' if you include both."
        )

        # --- USER payload: one JSON object the model must follow ---
        if self.json_schema_for_response is None:
            raise ValueError("json_schema_for_response must be set before building the extraction prompt.")
        if self.row_json_schema_hint is None:
            self.row_json_schema_hint = {}

        user_payload: Dict[str, Any] = {
            "data_dictionary": self._data_dictionary_block(),
            "json_schema_for_response": self.json_schema_for_response,
            "row_json_schema_hint": self.row_json_schema_hint,
            "defaults": self.defaults or {},
            "all_chunks": self.chunks,  # [{chunk_id, text}]
            "instructions": (
                self.extra_instructions
                if self.extra_instructions
                else [
                    "Analyze ALL provided chunks in this batch.",
                    "Include small evidence snippets; avoid quoting large text.",
                ]
            ),
        }

        # Helpful human-readable section (kept inside the user JSON, optional for your logs/parsing)
        user_payload["_human_friendly_overview"] = {
            "model": self.model_name,
            "description": self.model_spec.get("description", "N/A"),
            "primary_key": self.model_spec.get("primary_key", []),
            "fields_readable": self._format_field_description(),
            "constraints": self.constraints or [],
            "rules_extra": self.rules or [],
            "examples": self.examples or [],
        }

        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

    # ---------------------------
    # Validation
    # ---------------------------
    def build_validation_prompt(self) -> List[Dict[str, str]]:
        system_msg = (
            "You are a validation model that checks whether extracted records follow the rules and schema "
            "for a given data model. Use ONLY the provided chunks for evidence. Do NOT infer or fabricate values. "
            "Return strict JSON only."
        )

        # Developer guidance aligns w/ evidence objects + null-but-required fields
        dev_msg = (
            "Validation rules:\n"
            "1) Each non-null field must contain:\n"
            "   - value: the extracted value (type per schema)\n"
            "   - confidence: number in [0,1]\n"
            "   - evidence: list of objects [{chunk_id, snippet<=120}]\n"
            "   - notes: null or string\n"
            "2) Mark a field status as:\n"
            "   - VALID: value supported by evidence and within constraints\n"
            "   - INVALID: contradicts constraints or unsupported by evidence\n"
            "   - NULL: field key exists but set to null (allowed if not evidenced)\n"
            "3) Use hints/enums/ranges/required flags from the model to validate.\n"
            "4) Open-ended numeric ranges must use null for max (per extraction rules).\n"
            "5) If conflicting evidence exists, prefer the most specific and authoritative; otherwise flag INVALID.\n"
            "6) Always return a JSON object matching the expected validation schema for your validator."
        )

        # Build a compact evidence map for the model
        evidence_by_field = {
            fname: [
                {
                    "chunk_id": ch.get("chunk_id") or ch.get("id"),
                    "snippet": (ch.get("text") or "")[:300]
                }
                for ch in chunks_list
            ]
            for fname, chunks_list in (self.evidence_pack or {}).items()
        }

        user_payload = {
            "model": self.model_name,
            "data_dictionary": self._data_dictionary_block(),
            "constraints": self.constraints or [],
            "record": self.record,
            "evidence_by_field": evidence_by_field,
            "machine_checks": self.machine_checks or {},
        }

        return [
            {"role": "system", "content": system_msg},
            {"role": "developer", "content": dev_msg},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]
