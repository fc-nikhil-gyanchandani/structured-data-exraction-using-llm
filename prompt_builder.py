import json
import os
import re
from typing import List, Dict, Any, Optional


class PromptBuilder:
    """
    A unified prompt builder for schema-driven extraction and validation tasks.
    Generates OpenAI-compatible messages from a YAML-based data dictionary,
    supporting both extraction and validation use cases.

    Key features:
      - Generic domain-agnostic extraction based on data dictionary specifications
      - System prompt includes normalization, banding, confidence ladder, and evidence requirements
      - Evidence uses [{chunk_id, snippet}] format with comprehensive collection
      - Flexible template system supporting both extraction and validation
      - Robust schema validation and error handling
    """

    def __init__(
        self,
        model_name: str,
        data_dict: Dict[str, Any],
        task_type: str = "extraction",
        templates_dir: str = "prompt_templates",
        evidence_mode: str = "comprehensive"
    ):
        self.model_name = model_name
        # Pull the model block once
        self.model_spec: Dict[str, Any] = data_dict.get("models", {}).get(model_name, {})
        self.fields: List[Dict[str, Any]] = self.model_spec.get("fields", [])
        self.constraints: List[Dict[str, Any]] = self.model_spec.get("constraints", [])
        self.rules: List[str] = []          # extra, caller-provided rules
        self.examples: List[str] = []       # optional few-shot examples (strings)
        self.chunks: List[Dict[str, Any]] = []
        self.defaults: Dict[str, Any] = data_dict.get("defaults", {})
        self.task_type = task_type
        self.templates_dir = templates_dir
        self.evidence_mode = evidence_mode  # "comprehensive" or "limited"

        # New: externally-supplied schemas & instructions
        self.json_schema_for_response: Optional[Dict[str, Any]] = None
        self.row_json_schema_hint: Optional[Dict[str, Any]] = None
        self.extra_instructions: List[str] = []

        # Validation-specific attributes
        self.record: Dict[str, Any] = {}
        self.evidence_pack: Dict[str, List[Dict[str, Any]]] = {}
        self.machine_checks: Dict[str, Any] = {}

        # Load templates
        self.templates = self._load_templates()

    # ---------------------------
    # Template and file management
    # ---------------------------
    def _load_templates(self) -> Dict[str, str]:
        """Load prompt templates from the templates directory."""
        templates = {}
        
        # Load extraction templates
        extraction_file = os.path.join(self.templates_dir, 'extraction.txt')
        if os.path.exists(extraction_file):
            try:
                with open(extraction_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse extraction templates using markdown headers
                if '## System Prompt' in content and '## User Prompt Template' in content:
                    system_start = content.find('## System Prompt') + len('## System Prompt')
                    user_start = content.find('## User Prompt Template')
                    
                    if user_start != -1:
                        templates['system_extraction'] = content[system_start:user_start].strip()
                        user_content = content[user_start + len('## User Prompt Template'):].strip()
                        templates['user_extraction'] = user_content
                    else:
                        templates['system_extraction'] = content[system_start:].strip()
                else:
                    # Fallback: try to split by sections
                    sections = re.split(r'## (.*?)\n', content)
                    for i in range(1, len(sections), 2):
                        if i + 1 < len(sections):
                            section_name = sections[i].strip().lower().replace(' ', '_')
                            template_content = sections[i + 1].strip()
                            
                            if 'system' in section_name and 'prompt' in section_name:
                                templates['system_extraction'] = template_content
                            elif 'user' in section_name and 'prompt' in section_name:
                                templates['user_extraction'] = template_content
            except Exception as e:
                print(f"Warning: Could not parse extraction template: {e}")
        
        # Load validation templates
        validation_file = os.path.join(self.templates_dir, 'validation.txt')
        if os.path.exists(validation_file):
            try:
                with open(validation_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse validation templates
                sections = re.split(r'## (.*?)\n', content)
                for i in range(1, len(sections), 2):
                    if i + 1 < len(sections):
                        section_name = sections[i].strip().lower().replace(' ', '_')
                        template_content = sections[i + 1].strip()
                        
                        if 'system' in section_name:
                            templates['system_validation'] = template_content
                        elif 'developer' in section_name:
                            templates['developer_validation'] = template_content
                        elif 'user' in section_name:
                            templates['user_validation'] = template_content
            except Exception as e:
                print(f"Warning: Could not parse validation template: {e}")
        
        # If no templates found, use fallback
        if not templates:
            return self._get_fallback_templates()
        
        return templates
    
    def _get_fallback_templates(self) -> Dict[str, str]:
        """Fallback templates if the templates file is not found."""
        return {
            'system_extraction': """You are a schema-driven extraction model. Extract ONLY from provided chunks. 
Never invent or use outside knowledge. If a field is not explicitly supported 
by evidence, set that field to null (but include the field key). Output strict JSON only 
(no markdown, no commentary).

Quality contracts:
- No hallucinations.
- Obey the provided JSON Schema exactly.
- For every non-null field include relevant evidence objects with the shape 
{chunk_id, snippet} where snippet is a ≤120 char direct quote.
- If constraints appear violated, still return the best structured result and add a note in top-level 'notes'.""",
            
            'system_validation': """You are a validation model that checks whether extracted records follow the rules and schema.
Analyze the provided record against the data dictionary and evidence, then return validation results.""",
            
            'developer_validation': """Validation rules:
1) Each non-null field must contain valid evidence supporting the extraction
2) Check field types, constraints, and business rules
3) Verify evidence quality and relevance""",
            
            'user_extraction': """{
  "data_dictionary": {DATA_DICTIONARY_PLACEHOLDER},
  "json_schema_for_response": {JSON_SCHEMA_PLACEHOLDER},
  "row_json_schema_hint": {ROW_JSON_SCHEMA_HINT_PLACEHOLDER},
  "defaults": {DEFAULTS_PLACEHOLDER},
  "all_chunks": {CHUNKS_PLACEHOLDER},
  "instructions": {INSTRUCTIONS_PLACEHOLDER},
  "_human_friendly_overview": {
    "model": {MODEL_NAME_PLACEHOLDER},
    "description": {DESCRIPTION_PLACEHOLDER},
    "primary_key": {PRIMARY_KEY_PLACEHOLDER},
    "fields_readable": {FIELDS_READABLE_PLACEHOLDER},
    "constraints": {CONSTRAINTS_PLACEHOLDER},
    "rules_extra": {RULES_EXTRA_PLACEHOLDER},
    "examples": {EXAMPLES_PLACEHOLDER}
  }
}""",
            
            'user_validation': """{
  "model": {MODEL_NAME_PLACEHOLDER},
  "data_dictionary": {DATA_DICTIONARY_PLACEHOLDER},
  "constraints": {CONSTRAINTS_PLACEHOLDER},
  "record": {RECORD_PLACEHOLDER},
  "evidence_by_field": {EVIDENCE_BY_FIELD_PLACEHOLDER},
  "machine_checks": {MACHINE_CHECKS_PLACEHOLDER}
}"""
        }
    
    def save_prompt_to_file(self, output_dir: str = "output", filename_prefix: str = "prompt") -> str:
        """Save the current prompt as separate txt files."""
        os.makedirs(output_dir, exist_ok=True)
        
        prompt = self.build_prompt()
        timestamp = str(int(__import__('time').time()))
        
        # Save system prompt
        system_file = os.path.join(output_dir, f"{filename_prefix}_system_{timestamp}.txt")
        with open(system_file, 'w', encoding='utf-8') as f:
            f.write(prompt[0]['content'])
        
        # Save user prompt (or developer + user for validation)
        if len(prompt) > 2:  # validation case: system, developer, user
            dev_file = os.path.join(output_dir, f"{filename_prefix}_developer_{timestamp}.txt")
            with open(dev_file, 'w', encoding='utf-8') as f:
                f.write(prompt[1]['content'])
            
            user_file = os.path.join(output_dir, f"{filename_prefix}_user_{timestamp}.txt")
            with open(user_file, 'w', encoding='utf-8') as f:
                f.write(prompt[2]['content'])
        else:  # extraction case: system, user
            user_file = os.path.join(output_dir, f"{filename_prefix}_user_{timestamp}.txt")
            with open(user_file, 'w', encoding='utf-8') as f:
                f.write(prompt[1]['content'])
        
        return system_file

    # ---------------------------
    # Configuration helpers
    # ---------------------------
    def add_rule(self, rule: str):
        """Add a custom extraction rule."""
        self.rules.append(rule)

    def add_example(self, example: str):
        """Add a few-shot example."""
        self.examples.append(example)

    def add_chunk(self, chunk_id: str, text: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a text chunk with optional metadata."""
        chunk = {"chunk_id": chunk_id, "text": text}
        if metadata:
            chunk["metadata"] = metadata
        self.chunks.append(chunk)

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Add multiple chunks at once."""
        for chunk in chunks:
            if 'chunk_id' in chunk and 'text' in chunk:
                self.chunks.append(chunk)
            else:
                raise ValueError("Each chunk must have 'chunk_id' and 'text' keys")

    def set_json_schema_for_response(self, schema: Dict[str, Any]):
        """Set the JSON schema that the model should follow for response."""
        self.json_schema_for_response = schema

    def set_row_json_schema_hint(self, row_schema_hint: Dict[str, Any]):
        """Set hints about the expected row structure."""
        self.row_json_schema_hint = row_schema_hint

    def add_instruction(self, instruction: str):
        """Add a custom extraction instruction."""
        self.extra_instructions.append(instruction)

    def set_evidence_mode(self, mode: str):
        """Set evidence collection mode: 'comprehensive' or 'limited'."""
        if mode not in ['comprehensive', 'limited']:
            raise ValueError("Evidence mode must be 'comprehensive' or 'limited'")
        self.evidence_mode = mode

    # Validation-specific methods
    def set_record(self, record: Dict[str, Any]):
        """Set the record to be validated."""
        self.record = record

    def set_evidence_pack(self, evidence_pack: Dict[str, List[Dict[str, Any]]]):
        """
        Set evidence pack for validation.
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
        """Set machine validation checks results."""
        self.machine_checks = machine_checks

    # ---------------------------
    # Internal formatters
    # ---------------------------
    def _format_field_description(self) -> str:
        """Human-readable field list for the data_dictionary block."""
        field_descriptions = []
        for f in self.fields:
            desc = f"- `{f['name']}` ({f.get('dtype', 'string')})"
            if f.get('required'):
                desc += " [REQUIRED]"
            
            # Add hints
            if f.get('hints'):
                desc += f" | Hints: {', '.join(f.get('hints', []))}"
            
            # Add extraction rules
            if f.get('extraction_rules'):
                rules = '; '.join(f.get('extraction_rules', []))
                desc += f" | Rules: {rules}"
            
            # Add examples
            if f.get('examples'):
                examples = ', '.join([str(ex) for ex in f.get('examples', [])[:3]])  # Limit to first 3 examples
                desc += f" | Examples: {examples}"
            
            # Add constraints
            constraints = []
            if f.get('regex'):
                constraints.append(f"regex: {f['regex']}")
            if f.get('enum'):
                constraints.append(f"enum: {f['enum']}")
            if f.get('range'):
                constraints.append(f"range: {f['range']}")
            if f.get('normalize'):
                constraints.append(f"normalize: {f['normalize']}")
            if f.get('units'):
                constraints.append(f"units: {f['units']}")
            
            if constraints:
                desc += f" | Constraints: {'; '.join(constraints)}"
            
            field_descriptions.append(desc)
        
        return "\n".join(field_descriptions)

    def _data_dictionary_block(self) -> Dict[str, Any]:
        """Build the complete data dictionary block."""
        return {
            "model": self.model_name,
            "description": self.model_spec.get("description", "N/A"),
            "domain_context": self.model_spec.get("domain_context", ""),
            "extraction_perspective": self.model_spec.get("extraction_perspective", ""),
            "primary_key": self.model_spec.get("primary_key", []),
            "key_fields": self.model_spec.get("key_fields", []),
            "fields": {
                f["name"]: {
                    "type": f.get("dtype", "string"),
                    "description": f.get("description", ""),
                    "hints": f.get("hints", []),
                    "required": bool(f.get("required", False)),
                    "extraction_rules": f.get("extraction_rules", []),
                    "validation_patterns": f.get("regex", ""),
                    "examples": f.get("examples", []),
                    "normalize": f.get("normalize", ""),
                    "enum": f.get("enum", []),
                    "range": f.get("range", []),
                    "units": f.get("units", ""),
                    "notes": f.get("notes", "")
                } for f in self.fields
            },
            "business_rules": self.model_spec.get("business_rules", []),
            "record_grouping_logic": self.model_spec.get("record_grouping_logic", ""),
            "document_context": self.model_spec.get("document_context", {}),
            "extraction_focus": self.model_spec.get("extraction_focus", []),
            "source_path_filter": self.model_spec.get("source_path_filter", [])
        }

    def _validate_extraction_requirements(self):
        """Validate that all required components are present for extraction."""
        errors = []
        
        if not self.json_schema_for_response:
            errors.append("json_schema_for_response must be set before building prompt")
        if not self.chunks:
            errors.append("At least one chunk must be added before building prompt")
        if not self.fields:
            errors.append("Data dictionary must contain fields")
        if not self.model_name:
            errors.append("Model name must be specified")
            
        if errors:
            raise ValueError("Extraction requirements not met:\n" + "\n".join(f"- {e}" for e in errors))

    def _validate_validation_requirements(self):
        """Validate that all required components are present for validation."""
        errors = []
        
        if not self.record:
            errors.append("Record must be set for validation")
        if not self.evidence_pack and not self.machine_checks:
            errors.append("Either evidence_pack or machine_checks must be provided for validation")
        if not self.fields:
            errors.append("Data dictionary must contain fields")
            
        if errors:
            raise ValueError("Validation requirements not met:\n" + "\n".join(f"- {e}" for e in errors))

    def _replace_template_placeholders(self, template: str) -> str:
        """Replace all placeholders in the template with actual values."""
        replacements = {
            '{DATA_DICTIONARY_PLACEHOLDER}': json.dumps(self._data_dictionary_block(), ensure_ascii=False),
            '{JSON_SCHEMA_PLACEHOLDER}': json.dumps(self.json_schema_for_response or {}, ensure_ascii=False),
            '{ROW_JSON_SCHEMA_HINT_PLACEHOLDER}': json.dumps(self.row_json_schema_hint or {}, ensure_ascii=False),
            '{DEFAULTS_PLACEHOLDER}': json.dumps(self.defaults or {}, ensure_ascii=False),
            '{CHUNKS_PLACEHOLDER}': json.dumps(self.chunks, ensure_ascii=False),
            '{INSTRUCTIONS_PLACEHOLDER}': json.dumps(self.extra_instructions or [
                "Analyze ALL provided chunks in this batch.",
                f"Include {'comprehensive' if self.evidence_mode == 'comprehensive' else 'minimal'} evidence snippets.",
            ], ensure_ascii=False),
            '{MODEL_NAME_PLACEHOLDER}': json.dumps(self.model_name, ensure_ascii=False),
            '{DESCRIPTION_PLACEHOLDER}': json.dumps(self.model_spec.get("description", "N/A"), ensure_ascii=False),
            '{PRIMARY_KEY_PLACEHOLDER}': json.dumps(self.model_spec.get("primary_key", []), ensure_ascii=False),
            '{FIELDS_READABLE_PLACEHOLDER}': json.dumps(self._format_field_description(), ensure_ascii=False),
            '{CONSTRAINTS_PLACEHOLDER}': json.dumps(self.constraints or [], ensure_ascii=False),
            '{RULES_EXTRA_PLACEHOLDER}': json.dumps(self.rules or [], ensure_ascii=False),
            '{EXAMPLES_PLACEHOLDER}': json.dumps(self.examples or [], ensure_ascii=False),
            '{DOCUMENT_CONTEXT_PLACEHOLDER}': json.dumps(self.model_spec.get("document_context", {}), ensure_ascii=False),
            '{EXTRACTION_FOCUS_PLACEHOLDER}': json.dumps(self.model_spec.get("extraction_focus", []), ensure_ascii=False),
            # Validation-specific placeholders
            '{RECORD_PLACEHOLDER}': json.dumps(self.record, ensure_ascii=False),
            '{EVIDENCE_BY_FIELD_PLACEHOLDER}': json.dumps(self._build_evidence_by_field(), ensure_ascii=False),
            '{MACHINE_CHECKS_PLACEHOLDER}': json.dumps(self.machine_checks or {}, ensure_ascii=False)
        }
        
        result = template
        for placeholder, value in replacements.items():
            result = result.replace(placeholder, value)
        
        return result

    def _build_evidence_by_field(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build evidence mapping for validation."""
        return {
            fname: [
                {
                    "chunk_id": ch.get("chunk_id") or ch.get("id", "unknown"),
                    "snippet": (ch.get("text") or "")[:300]
                }
                for ch in chunks_list
            ]
            for fname, chunks_list in (self.evidence_pack or {}).items()
        }

    # ---------------------------
    # Public build API
    # ---------------------------
    def build_prompt(self) -> List[Dict[str, str]]:
        """Build the appropriate prompt based on task type."""
        if self.task_type == "validation":
            return self.build_validation_prompt()
        return self.build_extraction_prompt()

    # ---------------------------
    # Extraction
    # ---------------------------
    def build_extraction_prompt(self) -> List[Dict[str, str]]:
        """Build extraction prompt with system and user messages."""
        self._validate_extraction_requirements()
        
        # Get system message
        system_msg = self.templates.get('system_extraction', 
            self._get_fallback_templates()['system_extraction'])

        # Get user template and replace placeholders
        user_template = self.templates.get('user_extraction', 
            self._get_fallback_templates()['user_extraction'])
        
        user_content = self._replace_template_placeholders(user_template)

        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]

    # ---------------------------
    # Validation
    # ---------------------------
    def build_validation_prompt(self) -> List[Dict[str, str]]:
        """Build validation prompt with system, developer, and user messages."""
        self._validate_validation_requirements()
        
        # Get templates
        system_msg = self.templates.get('system_validation', 
            self._get_fallback_templates()['system_validation'])
        dev_msg = self.templates.get('developer_validation', 
            self._get_fallback_templates()['developer_validation'])
        user_template = self.templates.get('user_validation', 
            self._get_fallback_templates()['user_validation'])
        
        # Replace placeholders in user template
        user_content = self._replace_template_placeholders(user_template)

        return [
            {"role": "system", "content": system_msg},
            {"role": "developer", "content": dev_msg},
            {"role": "user", "content": user_content},
        ]

    # ---------------------------
    # Utility methods
    # ---------------------------
    def get_field_names(self) -> List[str]:
        """Get list of all field names."""
        return [f["name"] for f in self.fields]

    def get_required_fields(self) -> List[str]:
        """Get list of required field names."""
        return [f["name"] for f in self.fields if f.get("required", False)]

    def get_field_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get field specification by name."""
        for field in self.fields:
            if field["name"] == name:
                return field
        return None

    def validate_chunk_structure(self, chunk: Dict[str, Any]) -> bool:
        """Validate that a chunk has the required structure."""
        required_keys = ["chunk_id", "text"]
        return all(key in chunk for key in required_keys)

    def clear_chunks(self):
        """Clear all chunks."""
        self.chunks = []

    def clear_rules(self):
        """Clear all custom rules."""
        self.rules = []

    def clear_examples(self):
        """Clear all examples."""
        self.examples = []

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the current configuration."""
        return {
            "model_name": self.model_name,
            "task_type": self.task_type,
            "field_count": len(self.fields),
            "required_field_count": len(self.get_required_fields()),
            "chunk_count": len(self.chunks),
            "rule_count": len(self.rules),
            "example_count": len(self.examples),
            "constraint_count": len(self.constraints),
            "evidence_mode": self.evidence_mode,
            "has_json_schema": self.json_schema_for_response is not None,
            "has_row_schema": self.row_json_schema_hint is not None
        }