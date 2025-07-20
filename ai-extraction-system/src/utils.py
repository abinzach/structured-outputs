"""Utility functions for the extraction system."""
import tiktoken
import json
from typing import Any, Dict, List, Tuple
from jsonschema import validate, ValidationError


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text for a given model."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        # Fallback to cl100k_base encoding if model not found
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))


def estimate_schema_tokens(schema: dict, model: str = "gpt-4") -> int:
    """Estimate token count for a JSON schema."""
    schema_str = json.dumps(schema, indent=2)
    return count_tokens(schema_str, model)


def validate_json_against_schema(data: dict, schema: dict) -> Tuple[bool, str]:
    """Validate JSON data against a schema."""
    try:
        validate(instance=data, schema=schema)
        return True, ""
    except ValidationError as e:
        return False, str(e)


def get_nested_value(data: dict, path: str, default=None):
    """Get nested value from dictionary using dot notation."""
    keys = path.split('.')
    current = data
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current


def set_nested_value(data: dict, path: str, value: Any):
    """Set nested value in dictionary using dot notation."""
    keys = path.split('.')
    current = data
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value


def flatten_schema(schema: dict, prefix: str = "") -> Dict[str, dict]:
    """Flatten a nested schema into a flat dictionary with dot notation keys."""
    flattened = {}
    
    if 'properties' in schema:
        for key, value in schema['properties'].items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if 'properties' in value:
                # Nested object
                flattened.update(flatten_schema(value, full_key))
            else:
                # Leaf field
                flattened[full_key] = value
    
    return flattened


def calculate_schema_depth(schema: dict, current_depth: int = 0) -> int:
    """Calculate the maximum depth of a JSON schema."""
    max_depth = current_depth
    
    if 'properties' in schema:
        for value in schema['properties'].values():
            if 'properties' in value:
                depth = calculate_schema_depth(value, current_depth + 1)
                max_depth = max(max_depth, depth)
    
    return max_depth


def count_schema_objects(schema: dict) -> int:
    """Count the number of object types in a schema."""
    count = 0
    
    if 'properties' in schema:
        for value in schema['properties'].values():
            if value.get('type') == 'object' or 'properties' in value:
                count += 1
                count += count_schema_objects(value)
    
    return count


def count_enum_values(schema: dict) -> int:
    """Count total enum values in a schema."""
    count = 0
    
    if 'enum' in schema:
        count += len(schema['enum'])
    
    if 'properties' in schema:
        for value in schema['properties'].values():
            count += count_enum_values(value)
    
    if 'items' in schema:
        count += count_enum_values(schema['items'])
    
    return count


def merge_dicts_deep(dict1: dict, dict2: dict) -> dict:
    """Deep merge two dictionaries."""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts_deep(result[key], value)
        else:
            result[key] = value
    
    return result
