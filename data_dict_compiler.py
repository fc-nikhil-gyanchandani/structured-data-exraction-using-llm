"""
Data Dictionary Compiler - Provides schema compilation and validation functions
"""
import yaml
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from pydantic import BaseModel, create_model
import re
import json


@dataclass
class FieldSpec:
    """Field specification from data dictionary"""
    name: str
    dtype: str
    required: bool = False
    enum: Optional[List[str]] = None
    range: Optional[List[Union[int, float]]] = None
    regex: Optional[str] = None
    normalize: Optional[str] = None
    hints: Optional[List[str]] = None
    examples: Optional[List[str]] = None
    extraction_rules: Optional[List[str]] = None
    units: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class ModelSpec:
    """Model specification from data dictionary"""
    name: str
    description: str
    primary_key: List[str]
    key_fields: List[str]
    fields: List[FieldSpec]
    domain_context: Optional[str] = None
    extraction_perspective: Optional[str] = None
    business_rules: Optional[List[str]] = None
    record_grouping_logic: Optional[str] = None
    constraints: Optional[List[Dict[str, Any]]] = None
    ingestion_gate: Optional[Dict[str, Any]] = None
    evidence_rules: Optional[Dict[str, Any]] = None
    source_path_filter: Optional[List[str]] = None


@dataclass
class DataDictionary:
    """Complete data dictionary with models"""
    version: str
    defaults: Dict[str, Any]
    models: Dict[str, ModelSpec]


def load_dictionary(yaml_path: str) -> DataDictionary:
    """Load data dictionary from YAML file"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    # Convert to our dataclass structure
    models = {}
    for model_name, model_data in data.get('models', {}).items():
        fields = []
        for field_data in model_data.get('fields', []):
            field = FieldSpec(**field_data)
            fields.append(field)
        
        model = ModelSpec(
            name=model_name,
            description=model_data.get('description', ''),
            primary_key=model_data.get('primary_key', []),
            key_fields=model_data.get('key_fields', []),
            fields=fields,
            domain_context=model_data.get('domain_context', ''),
            extraction_perspective=model_data.get('extraction_perspective', ''),
            business_rules=model_data.get('business_rules', []),
            record_grouping_logic=model_data.get('record_grouping_logic', ''),
            constraints=model_data.get('constraints', []),
            ingestion_gate=model_data.get('ingestion_gate', {}),
            evidence_rules=model_data.get('evidence_rules', {}),
            source_path_filter=model_data.get('source_path_filter', [])
        )
        models[model_name] = model
    
    return DataDictionary(
        version=data.get('version', '0.1'),
        defaults=data.get('defaults', {}),
        models=models
    )


def compile_json_schema(spec: ModelSpec) -> Dict[str, Any]:
    """Compile JSON schema for row-level validation"""
    properties = {}
    required = []
    
    for field in spec.fields:
        # Map dtype to JSON schema type
        if field.dtype == "string":
            schema_type = "string"
        elif field.dtype in ("number", "decimal"):
            schema_type = "number"
        elif field.dtype == "integer":
            schema_type = "integer"
        elif field.dtype == "boolean":
            schema_type = "boolean"
        elif field.dtype in ("date", "datetime"):
            schema_type = "string"
        else:
            schema_type = "string"
        
        field_schema = {"type": schema_type}
        
        # Add enum if specified
        if field.enum:
            field_schema["enum"] = field.enum
        
        # Add range if specified
        if field.range and len(field.range) >= 2:
            if field.range[0] is not None:
                field_schema["minimum"] = field.range[0]
            if field.range[1] is not None:
                field_schema["maximum"] = field.range[1]
        
        # Add regex if specified
        if field.regex:
            field_schema["pattern"] = field.regex
        
        properties[field.name] = field_schema
        
        if field.required:
            required.append(field.name)
    
    return {
        "type": "object",
        "properties": properties,
        "required": required
    }


def compile_pydantic(spec: ModelSpec) -> type:
    """Compile Pydantic model from specification"""
    field_definitions = {}
    
    for field in spec.fields:
        # Map dtype to Python type
        if field.dtype == "string":
            field_type = str
        elif field.dtype in ("number", "decimal"):
            field_type = float
        elif field.dtype == "integer":
            field_type = int
        elif field.dtype == "boolean":
            field_type = bool
        elif field.dtype in ("date", "datetime"):
            field_type = str
        else:
            field_type = str
        
        # Handle required vs optional
        if field.required:
            field_definitions[field.name] = (field_type, ...)
        else:
            field_definitions[field.name] = (Optional[field_type], None)
    
    return create_model(f"Dynamic{spec.name.title()}", **field_definitions)


def check_constraints(row: Dict[str, Any], spec: ModelSpec) -> List[str]:
    """Check constraints for a row (simplified implementation)"""
    violations = []
    
    if not spec.constraints:
        return violations
    
    for constraint in spec.constraints:
        constraint_name = constraint.get('name', 'unknown')
        expression = constraint.get('expression', '')
        
        # Simple constraint checking - in a real implementation this would be more sophisticated
        if 'is not null' in expression.lower():
            field_name = expression.split()[0]
            if row.get(field_name) is None:
                violations.append(f"{constraint_name}: {field_name} cannot be null")
        
        elif 'length(trim(' in expression.lower():
            # Extract field name from length(trim(field)) > 0
            match = re.search(r'length\(trim\(([^)]+)\)\)\s*>\s*0', expression)
            if match:
                field_name = match.group(1)
                value = row.get(field_name, '')
                if not str(value).strip():
                    violations.append(f"{constraint_name}: {field_name} cannot be empty")
    
    return violations 