"""
Prompt Builder for creating LLM prompts with data dictionary integration
Reusable across different agents
"""

import json
import logging
import yaml
import tiktoken
from typing import Dict, Any, List, Optional
from pathlib import Path
from langchain.prompts import PromptTemplate
from langchain.schema import Document

logger = logging.getLogger(__name__)

class PromptBuilder:
    """Builds prompts for different agents using a data dictionary"""
    
    def __init__(self, country: Optional[str] = None, defaults: Optional[Dict[str, Any]] = None):
        self.country = country
        self.defaults = defaults or {}
        self._load_data_dictionary()
        self._tokenizer = None
        self._current_template = None
    
    def _get_tokenizer(self):
        """Get the tokenizer for the current model."""
        if self._tokenizer is None:
            try:
                # Encoding for gpt-4o and similar models
                self._tokenizer = tiktoken.get_encoding("o200k_base")
            except ValueError:
                # Fallback for other models like gpt-4
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
        return self._tokenizer

    def _estimate_tokens(self, text: str) -> int:
        """Roughly estimate the number of tokens for a given string."""
        tokenizer = self._get_tokenizer()
        return len(tokenizer.encode(text))

    def _load_data_dictionary(self):
        """Load data dictionary from YAML file"""
        try:
            # Look for data_dictionary.yaml in common locations
            yaml_paths = [
                Path("data_dictionary.yaml"),
                Path("..") / "data_dictionary.yaml",
                Path("../..") / "data_dictionary.yaml"
            ]
            
            yaml_path = None
            for path in yaml_paths:
                if path.exists():
                    yaml_path = path
                    break
            
            if not yaml_path:
                logger.warning("Data dictionary not found, using empty dictionary")
                self.data_dictionary = {}
                return
            
            with open(yaml_path, 'r', encoding='utf-8') as f:
                self.data_dictionary = yaml.safe_load(f)
                
        except Exception as e:
            logger.error(f"Error loading data dictionary: {e}")
            self.data_dictionary = {}

    def _get_template(self, template_name: str) -> str:
        """Loads a prompt template from a file."""
        try:
            template_paths = [
                Path("prompt_templates") / f"{template_name}.txt",
                Path("..") / "prompt_templates" / f"{template_name}.txt",
                Path("../..") / "prompt_templates" / f"{template_name}.txt"
            ]
            
            for template_path in template_paths:
                if template_path.exists():
                    with open(template_path, 'r', encoding='utf-8') as f:
                        return f.read()
            
            raise FileNotFoundError(f"Template not found: {template_name}")
        except Exception as e:
            logger.error(f"Failed to load template {template_name}: {e}")
            raise

    def build_mapper_prompt(self, file_chunks: List[Dict[str, Any]], max_tokens: int = 100000) -> str:
        """Builds the prompt for the classifier/mapper agent with truncation."""
        
        template_str = self._get_template("mapper")
        
        # Set template type for placeholder replacement
        self._current_template = "mapper"
        
        # For mapper/classifier, include complete data dictionary for model selection
        dictionary_str = json.dumps(self.data_dictionary, indent=2)
        
        # First, format the template with the dictionary to estimate tokens accurately
        base_prompt = template_str.format(
            DATA_DICTIONARY=dictionary_str,
            FILE_CHUNK="{FILE_CHUNK}"  # Keep placeholder for later replacement
        )
        
        # Calculate available tokens for chunks
        base_tokens = self._estimate_tokens(base_prompt)
        available_tokens_for_chunks = max_tokens - base_tokens
        
        # Add chunks until token limit is reached
        final_chunks_str = ""
        included_chunks_count = 0
        for chunk in file_chunks:
            # Use simplified format: just id and text
            chunk_id = chunk.get('chunk_id', chunk.get('id', 'N/A'))
            chunk_text = chunk.get('text', '')
            chunk_str = f"--- Chunk ID: {chunk_id} ---\n{chunk_text}\n\n"
            chunk_tokens = self._estimate_tokens(chunk_str)
            
            if available_tokens_for_chunks - chunk_tokens > 0:
                final_chunks_str += chunk_str
                available_tokens_for_chunks -= chunk_tokens
                included_chunks_count += 1
            else:
                logger.warning(
                    f"Truncating chunks for mapper prompt. "
                    f"Included {included_chunks_count}/{len(file_chunks)} chunks to fit within {max_tokens} tokens."
                )
                break
        
        return base_prompt.replace("{FILE_CHUNK}", final_chunks_str)

    def build_extraction_prompt(self, file_chunks: List[Dict[str, Any]], model_spec: Dict[str, Any], max_tokens: int = 100000) -> str:
        """Builds the prompt for the extraction agent with truncation."""
        
        template_str = self._get_template("extraction")
        model_name = model_spec.get("model_name", "unknown")
        
        # Set template type for placeholder replacement
        self._current_template = "extraction"
        
        # Base prompt without chunks
        base_prompt = template_str.replace("{{chunks}}", "")

        base_tokens = self._estimate_tokens(base_prompt)
        available_tokens_for_chunks = max_tokens - base_tokens
        
        # Add chunks until token limit is reached
        final_chunks_str = ""
        included_chunks_count = 0
        for chunk in file_chunks:
            # Use simplified format: just id and text
            chunk_id = chunk.get('chunk_id', chunk.get('id', 'N/A'))
            chunk_text = chunk.get('text', '')
            chunk_str = f"--- Chunk ID: {chunk_id} ---\n{chunk_text}\n\n"
            chunk_tokens = self._estimate_tokens(chunk_str)
            
            if available_tokens_for_chunks - chunk_tokens > 0:
                final_chunks_str += chunk_str
                available_tokens_for_chunks -= chunk_tokens
                included_chunks_count += 1
            else:
                logger.warning(
                    f"Truncating chunks for extraction prompt (model: {model_name}). "
                    f"Included {included_chunks_count}/{len(file_chunks)} chunks to fit within {max_tokens} tokens."
                )
                break
        
        # For extraction, include only the specific model's data dictionary
        model_data_dict = {
            "model": model_name,
            "description": model_spec.get("description", ""),
            "domain_context": model_spec.get("domain_context", ""),
            "extraction_perspective": model_spec.get("extraction_perspective", ""),
            "primary_key": model_spec.get("primary_key", []),
            "key_fields": model_spec.get("key_fields", []),
            "fields": model_spec.get("fields", []),
            "business_rules": model_spec.get("business_rules", []),
            "record_grouping_logic": model_spec.get("record_grouping_logic", ""),
            "document_context": model_spec.get("document_context", {}),
            "extraction_focus": model_spec.get("extraction_focus", []),
            "source_path_filter": model_spec.get("source_path_filter", [])
        }
        
        # Add model_name to model_spec for compatibility
        model_spec["model_name"] = model_name
        
        # Use comprehensive placeholder replacement
        prompt = self._replace_template_placeholders(template_str, model_spec, file_chunks)
        
        # Replace chunks placeholder with the actual chunks string
        prompt = prompt.replace("{CHUNKS_PLACEHOLDER}", final_chunks_str)
        
        return prompt

    def _replace_template_placeholders(self, template: str, model_spec: Dict[str, Any] = None, chunks: List[Dict[str, Any]] = None) -> str:
        """Replace all placeholders in the template with actual values using comprehensive replacement."""
        if model_spec is None:
            model_spec = {}
        if chunks is None:
            chunks = []
            
        # Build data dictionary block
        data_dict_block = self._data_dictionary_block(model_spec)
        
        # Build field descriptions
        field_descriptions = self._format_field_description(model_spec.get("fields", []))
        
        replacements = {
            '{DATA_DICTIONARY_PLACEHOLDER}': json.dumps(data_dict_block, ensure_ascii=False, indent=2),
            '{JSON_SCHEMA_PLACEHOLDER}': json.dumps(model_spec.get("json_schema", {}), ensure_ascii=False, indent=2),
            '{ROW_JSON_SCHEMA_HINT_PLACEHOLDER}': json.dumps(model_spec.get("row_json_schema_hint", {}), ensure_ascii=False, indent=2),
            '{DEFAULTS_PLACEHOLDER}': json.dumps(self.defaults, ensure_ascii=False, indent=2),
            '{CHUNKS_PLACEHOLDER}': json.dumps([{"id": chunk.get("chunk_id", chunk.get("id", "unknown")), "text": chunk.get("text", "")} for chunk in chunks], ensure_ascii=False, indent=2),
            '{CHUNKS_JSON}': json.dumps([{"id": chunk.get("chunk_id", chunk.get("id", "unknown")), "text": chunk.get("text", "")} for chunk in chunks], ensure_ascii=False, indent=2),
            '{RECORDS_JSON}': json.dumps([], ensure_ascii=False, indent=2),  # Will be replaced later
            '{INSTRUCTIONS_PLACEHOLDER}': json.dumps(model_spec.get("extraction_instructions", []), ensure_ascii=False, indent=2),
            '{MODEL_NAME_PLACEHOLDER}': json.dumps(model_spec.get("model_name", "unknown"), ensure_ascii=False),
            '{MODEL_NAME}': model_spec.get("model_name", "unknown"),
            '{DESCRIPTION_PLACEHOLDER}': json.dumps(model_spec.get("description", "N/A"), ensure_ascii=False),
            '{PRIMARY_KEY_PLACEHOLDER}': json.dumps(model_spec.get("primary_key", []), ensure_ascii=False, indent=2),
            '{FIELDS_READABLE_PLACEHOLDER}': json.dumps(field_descriptions, ensure_ascii=False),
            '{CONSTRAINTS_PLACEHOLDER}': json.dumps(model_spec.get("constraints", []), ensure_ascii=False, indent=2),
            '{RULES_EXTRA_PLACEHOLDER}': json.dumps(model_spec.get("business_rules", []), ensure_ascii=False, indent=2),
            '{EXAMPLES_PLACEHOLDER}': json.dumps(model_spec.get("examples", []), ensure_ascii=False, indent=2),
            '{DOCUMENT_CONTEXT_PLACEHOLDER}': json.dumps(model_spec.get("document_context", {}), ensure_ascii=False, indent=2),
            '{EXTRACTION_FOCUS_PLACEHOLDER}': json.dumps(model_spec.get("extraction_focus", []), ensure_ascii=False, indent=2),
            # Validation-specific placeholders
            '{RECORD_PLACEHOLDER}': json.dumps({}, ensure_ascii=False, indent=2),
            '{EVIDENCE_BY_FIELD_PLACEHOLDER}': json.dumps({}, ensure_ascii=False, indent=2),
            '{MACHINE_CHECKS_PLACEHOLDER}': json.dumps({}, ensure_ascii=False, indent=2)
        }
        
        # For validation templates, don't replace RECORDS_JSON and CHUNKS_JSON placeholders
        # as they will be replaced later with actual data
        template_name = getattr(self, '_current_template', '')
        if 'validation' in template_name.lower():
            # Remove these placeholders from replacement so they remain as placeholders
            if '{RECORDS_JSON}' in replacements:
                del replacements['{RECORDS_JSON}']
            if '{CHUNKS_JSON}' in replacements:
                del replacements['{CHUNKS_JSON}']
        
        result = template
        for placeholder, value in replacements.items():
            result = result.replace(placeholder, value)
        
        return result

    def _data_dictionary_block(self, model_spec: Dict[str, Any] = None) -> Dict[str, Any]:
        """Build the complete data dictionary block."""
        if model_spec is None:
            model_spec = {}
            
        fields = model_spec.get("fields", [])
        
        return {
            "model": model_spec.get("model_name", "unknown"),
            "description": model_spec.get("description", ""),
            "domain_context": model_spec.get("domain_context", ""),
            "extraction_perspective": model_spec.get("extraction_perspective", ""),
            "primary_key": model_spec.get("primary_key", []),
            "key_fields": model_spec.get("key_fields", []),
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
                } for f in fields
            },
            "business_rules": model_spec.get("business_rules", []),
            "record_grouping_logic": model_spec.get("record_grouping_logic", ""),
            "document_context": model_spec.get("document_context", {}),
            "extraction_focus": model_spec.get("extraction_focus", []),
            "source_path_filter": model_spec.get("source_path_filter", [])
        }
    
    def _format_field_description(self, fields: List[Dict[str, Any]]) -> str:
        """Human-readable field list for the data_dictionary block."""
        field_descriptions = []
        for f in fields:
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

    def build_validation_prompt(self, model_name: str, model_spec: Dict[str, Any], 
                                records_data: List[Dict[str, Any]], 
                                evidence_chunks: List[Dict[str, Any]], 
                                max_tokens: int = 100000) -> str:
        """Builds the prompt for the validation agent with truncation."""
        template_str = self._get_template("validation")

        # Add model_name to model_spec for compatibility
        model_spec["model_name"] = model_name
        
        # Set template type for placeholder replacement
        self._current_template = "validation"
        
        # Use comprehensive placeholder replacement (same as extraction)
        # For validation, don't include evidence chunks in base prompt - they'll be added later
        prompt = self._replace_template_placeholders(template_str, model_spec, [])

        base_tokens = self._estimate_tokens(prompt)
        available_tokens = max_tokens - base_tokens
        
        # Add records first, as they are the primary subject of validation
        final_records_str = ""
        included_records_count = 0
        records_to_include = []
        for record in records_data:
            record_str = json.dumps(record, indent=2) + "\n"
            record_tokens = self._estimate_tokens(record_str)
            
            if available_tokens - record_tokens > 0:
                records_to_include.append(record)
                available_tokens -= record_tokens
                included_records_count += 1
            else:
                logger.warning(
                    f"Truncating records for validation prompt (model: {model_name}). "
                    f"Included {included_records_count}/{len(records_data)} records."
                )
                break
        final_records_str = json.dumps(records_to_include, indent=2)

        # Add as many evidence chunks as can fit in the remaining space
        final_chunks_str = ""
        included_chunks_count = 0
        for chunk in evidence_chunks:
            # Use simplified format: just id and text
            chunk_id = chunk.get('chunk_id', chunk.get('id', 'N/A'))
            chunk_text = chunk.get('text', '')
            chunk_str = f"--- Chunk ID: {chunk_id} ---\n{chunk_text}\n\n"
            chunk_tokens = self._estimate_tokens(chunk_str)

            if available_tokens - chunk_tokens > 0:
                final_chunks_str += chunk_str
                available_tokens -= chunk_tokens
                included_chunks_count += 1
            else:
                logger.warning(
                    f"Truncating evidence chunks for validation prompt (model: {model_name}). "
                    f"Included {included_chunks_count}/{len(evidence_chunks)} chunks."
                )
                break
        
        # Replace the placeholders with actual data
        prompt = prompt.replace("{RECORDS_JSON}", final_records_str)
        prompt = prompt.replace("{CHUNKS_JSON}", final_chunks_str)
        
        return prompt
    
    def _truncate_json_array(self, json_str: str, max_tokens: int) -> str:
        """Truncate a JSON array to fit within token limit"""
        try:
            data = json.loads(json_str)
            if not isinstance(data, list):
                return json_str
            
            # Start with empty array and add items until token limit
            result = []
            current_tokens = 2  # For "[]"
            
            for item in data:
                item_str = json.dumps(item)
                item_tokens = self._estimate_tokens(item_str)
                
                if current_tokens + item_tokens + 1 > max_tokens:  # +1 for comma
                    break
                
                result.append(item)
                current_tokens += item_tokens + 1
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to truncate JSON array: {e}")
            return "[]"
    
    def build_classifier_prompt(self, file_chunks: List[Dict[str, Any]], max_tokens: int = 100000) -> str:
        """
        Build prompt for classifier/mapper agent.
        Includes complete data dictionary for model selection.
        """
        return self.build_mapper_prompt(file_chunks, max_tokens)
    
    def build_extraction_prompt_for_model(self, model_name: str, file_chunks: List[Dict[str, Any]], 
                                        max_tokens: int = 100000) -> str:
        """
        Build prompt for extraction agent for a specific model.
        Includes only the specific model's data dictionary.
        """
        # Get the specific model's specification
        model_spec = self.data_dictionary.get("models", {}).get(model_name, {})
        if not model_spec:
            raise ValueError(f"Model '{model_name}' not found in data dictionary")
        
        # Add model_name to the spec for compatibility
        model_spec["model_name"] = model_name
        
        return self.build_extraction_prompt(file_chunks, model_spec, max_tokens)
    
    def build_validation_prompt_for_model(self, model_name: str, records_data: List[Dict[str, Any]], 
                                        evidence_chunks: List[Dict[str, Any]], 
                                        max_tokens: int = 100000) -> str:
        """
        Build prompt for validation agent for a specific model.
        Includes only the specific model's data dictionary.
        """
        # Get the specific model's specification
        model_spec = self.data_dictionary.get("models", {}).get(model_name, {})
        if not model_spec:
            raise ValueError(f"Model '{model_name}' not found in data dictionary")
        
        return self.build_validation_prompt(model_name, model_spec, records_data, evidence_chunks, max_tokens)
    
    def build_custom_prompt(self, template_name: str, **kwargs) -> str:
        """
        Build custom prompt with any template and variables
        
        Args:
            template_name: Name of template file
            **kwargs: Variables to substitute in template
            
        Returns:
            Complete prompt string
        """
        try:
            # Load template
            template = self._get_template(template_name)
            
            # Create prompt with provided variables
            prompt = template.format(**kwargs)
            
            return prompt
            
        except Exception as e:
            logger.error(f"Failed to build custom prompt {template_name}: {e}")
            raise
    
    def _generate_json_schema(self, model_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Generate JSON schema based on model specification"""
        try:
            # Get fields from model spec
            fields = model_spec.get("fields", {})
            if isinstance(fields, dict):
                # Convert dict format to list format if needed
                field_list = []
                for field_name, field_spec in fields.items():
                    if isinstance(field_spec, dict):
                        field_spec["name"] = field_name
                        field_list.append(field_spec)
                fields = field_list
            
            # Build properties for each field
            record_properties = {}
            required_fields = []
            
            for field in fields:
                field_name = field.get("name", "")
                field_type = self._map_field_type(field.get("dtype", "string"))
                is_required = field.get("required", False)
                
                if field_name:
                    record_properties[field_name] = {
                        "type": field_type
                    }
                    
                    if is_required:
                        required_fields.append(field_name)
            
            # Build the complete schema
            schema = {
                "type": "object",
                "properties": {
                    "records": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": record_properties,
                            "required": required_fields
                        }
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string"},
                            "total_records": {"type": "integer"},
                            "confidence": {"type": "number"}
                        }
                    }
                },
                "required": ["records"]
            }
            
            return schema
            
        except Exception as e:
            logger.error(f"Failed to generate JSON schema: {e}")
            # Return a basic schema as fallback
            return {
                "type": "object",
                "properties": {
                    "records": {
                        "type": "array",
                        "items": {"type": "object"}
                    }
                },
                "required": ["records"]
            }
    
    def _map_field_type(self, dtype: str) -> str:
        """Map data dictionary field types to JSON schema types"""
        type_mapping = {
            "string": "string",
            "str": "string",
            "integer": "integer",
            "int": "integer",
            "number": "number",
            "float": "number",
            "boolean": "boolean",
            "bool": "boolean",
            "array": "array",
            "object": "object"
        }
        return type_mapping.get(dtype.lower(), "string")
    
    def _combine_chunks_with_token_limit(self, chunks: List[Dict[str, Any]], max_tokens: int) -> str:
        """
        Combine chunks while respecting token limit
        
        Args:
            chunks: List of chunk dictionaries
            max_tokens: Maximum tokens allowed
            
        Returns:
            Combined text string
        """
        combined_text = ""
        current_tokens = 0
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get("text", "")
            chunk_tokens = len(chunk_text.split())  # Rough token estimate
            
            # Check if adding this chunk would exceed limit
            if current_tokens + chunk_tokens > max_tokens:
                logger.warning(f"Token limit reached, stopping at {current_tokens} tokens")
                break
            
            combined_text += f"\n\nCHUNK {i+1}:\n{chunk_text}"
            current_tokens += chunk_tokens
        
        return combined_text
    
    def get_data_dictionary(self) -> Dict[str, Any]:
        """Get the loaded data dictionary"""
        return self.data_dictionary
    
    def get_available_models(self) -> List[str]:
        """Returns a list of available model names"""
        models = self.data_dictionary.get("models", {})
        return list(models.keys())
    
    def get_model_spec(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get specification for a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model specification or None if not found
        """
        models = self.data_dictionary.get("models", {})
        return models.get(model_name)
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        tokenizer = self._get_tokenizer()
        return len(tokenizer.encode(text))
    
    def validate_template_variables(self, template_name: str, **kwargs) -> bool:
        """
        Validate that all required variables are provided for a template
        
        Args:
            template_name: Name of template file
            **kwargs: Variables to check
            
        Returns:
            True if all required variables are provided
        """
        try:
            template = self._get_template(template_name)
            
            # Try to format with provided variables
            template.format(**kwargs)
            return True
            
        except KeyError as e:
            logger.error(f"Missing required variable for template {template_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Template validation failed: {e}")
            return False
