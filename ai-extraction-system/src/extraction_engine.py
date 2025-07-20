"""Extraction Engine for AI-powered text-to-JSON conversion."""
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI
try:
    from schema_analyzer import SchemaAnalyzer
    from utils import count_tokens, validate_json_against_schema, merge_dicts_deep
except ImportError:
    from .schema_analyzer import SchemaAnalyzer
    from .utils import count_tokens, validate_json_against_schema, merge_dicts_deep


class ExtractionEngine:
    """Main extraction engine using OpenAI API."""
    
    def __init__(self, model: str = "gpt-4.1", api_key: str = None):
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.schema_analyzer = SchemaAnalyzer()
        self.max_tokens_per_request = int(os.getenv('MAX_TOKENS_PER_REQUEST', 1000000))
    
    def extract(self, text: str, schema: dict) -> dict:
        """
        Main extraction method that automatically chooses the best strategy.
        
        Args:
            text: Input text to extract from
            schema: JSON schema defining the expected output structure
            
        Returns:
            dict: Extracted data with confidence scores
        """
        # Analyze schema complexity
        analysis = self.schema_analyzer.analyze_complexity(schema)
        strategy = analysis['strategy']
        
        if strategy == 'single_pass':
            return self.extract_simple(text, schema)
        elif strategy == 'multi_pass_hierarchical':
            return self.extract_hierarchical(text, schema)
        else:  # multi_pass_chunked
            schema_chunks = self.schema_analyzer.chunk_schema(schema)
            return self.extract_complex(text, schema_chunks)
    
    def extract_simple(self, text: str, schema: dict) -> dict:
        """
        Single-pass extraction for simple schemas.
        
        Args:
            text: Input text
            schema: JSON schema
            
        Returns:
            dict: Extraction result with confidence scores
        """
        prompt = self._build_extraction_prompt(text, schema)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert data extraction system. Extract structured data from text according to the provided JSON schema. Return only valid JSON that matches the schema exactly."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            confidence = self.calculate_confidence(result, schema, text)
            
            return {
                'data': result,
                'confidence': confidence,
                'strategy': 'single_pass',
                'token_usage': response.usage.total_tokens if response.usage else 0
            }
            
        except Exception as e:
            return {
                'data': {},
                'confidence': {'overall': 0.0, 'fields': {}},
                'error': str(e),
                'strategy': 'single_pass'
            }
    
    def extract_hierarchical(self, text: str, schema: dict) -> dict:
        """
        Multi-pass hierarchical extraction for complex nested schemas.
        
        Args:
            text: Input text
            schema: JSON schema
            
        Returns:
            dict: Extraction result with confidence scores
        """
        # Extract in layers: top-level first, then nested objects
        extraction_order = self.schema_analyzer.get_extraction_order(schema)
        
        result = {}
        confidence_scores = {}
        total_tokens = 0
        
        # Group fields by hierarchy level
        levels = self._group_fields_by_level(extraction_order)
        
        for level, fields in levels.items():
            level_schema = self._build_level_schema(schema, fields)
            level_prompt = self._build_extraction_prompt(text, level_schema, result)
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are extracting data at hierarchy level {level}. Use any previously extracted data as context."
                        },
                        {
                            "role": "user",
                            "content": level_prompt
                        }
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                
                level_result = json.loads(response.choices[0].message.content)
                result = merge_dicts_deep(result, level_result)
                
                level_confidence = self.calculate_confidence(level_result, level_schema, text)
                confidence_scores.update(level_confidence['fields'])
                
                total_tokens += response.usage.total_tokens if response.usage else 0
                
            except Exception as e:
                print(f"Error in hierarchical extraction level {level}: {e}")
                continue
        
        overall_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0.0
        
        return {
            'data': result,
            'confidence': {
                'overall': overall_confidence,
                'fields': confidence_scores
            },
            'strategy': 'multi_pass_hierarchical',
            'token_usage': total_tokens
        }
    
    def extract_complex(self, text: str, schema_chunks: List[dict]) -> dict:
        """
        Multi-pass extraction for complex schemas using chunking.
        
        Args:
            text: Input text
            schema_chunks: List of schema chunks from SchemaAnalyzer
            
        Returns:
            dict: Merged extraction results with confidence scores
        """
        results = []
        total_tokens = 0
        
        # Sort chunks by priority (high priority first)
        sorted_chunks = sorted(schema_chunks, 
                             key=lambda x: {'high': 0, 'medium': 1, 'low': 2}[x['priority']])
        
        previous_results = {}
        
        for chunk_info in sorted_chunks:
            chunk_schema = chunk_info['schema']
            chunk_id = chunk_info['chunk_id']
            
            # Build context from previous results
            context_prompt = self._build_chunk_context(previous_results, chunk_info['dependencies'])
            prompt = self._build_extraction_prompt(text, chunk_schema, context=context_prompt)
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are extracting chunk {chunk_id + 1} of {chunk_info['total_chunks']}. Use provided context from previous extractions."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                
                chunk_result = json.loads(response.choices[0].message.content)
                confidence = self.calculate_confidence(chunk_result, chunk_schema, text)
                
                results.append({
                    'data': chunk_result,
                    'confidence': confidence,
                    'chunk_id': chunk_id,
                    'priority': chunk_info['priority']
                })
                
                # Update previous results for context
                previous_results = merge_dicts_deep(previous_results, chunk_result)
                total_tokens += response.usage.total_tokens if response.usage else 0
                
            except Exception as e:
                print(f"Error in chunk {chunk_id}: {e}")
                results.append({
                    'data': {},
                    'confidence': {'overall': 0.0, 'fields': {}},
                    'chunk_id': chunk_id,
                    'error': str(e)
                })
        
        # Merge all results
        merged_data = {}
        all_field_confidences = {}
        
        for result in results:
            merged_data = merge_dicts_deep(merged_data, result['data'])
            all_field_confidences.update(result['confidence']['fields'])
        
        overall_confidence = sum(all_field_confidences.values()) / len(all_field_confidences) if all_field_confidences else 0.0
        
        return {
            'data': merged_data,
            'confidence': {
                'overall': overall_confidence,
                'fields': all_field_confidences
            },
            'strategy': 'multi_pass_chunked',
            'token_usage': total_tokens,
            'chunk_results': results
        }
    
    def calculate_confidence(self, result: dict, schema: dict, original_text: str = "") -> dict:
        """
        Calculate confidence scores for extraction results.
        
        Args:
            result: Extracted data
            schema: Original schema
            original_text: Source text for context
            
        Returns:
            dict: Confidence scores
        """
        field_confidences = {}
        
        # Validate against schema
        is_valid, validation_error = validate_json_against_schema(result, schema)
        base_confidence = 0.8 if is_valid else 0.3
        
        # Calculate field-level confidence
        try:
            from utils import flatten_schema
        except ImportError:
            from .utils import flatten_schema
        flattened_schema = flatten_schema(schema)
        
        for field_path, field_schema in flattened_schema.items():
            field_confidence = self._calculate_field_confidence(
                result, field_path, field_schema, original_text
            )
            field_confidences[field_path] = field_confidence
        
        overall_confidence = sum(field_confidences.values()) / len(field_confidences) if field_confidences else base_confidence
        
        return {
            'overall': overall_confidence,
            'fields': field_confidences,
            'schema_valid': is_valid,
            'validation_error': validation_error if not is_valid else None
        }
    
    def _build_extraction_prompt(self, text: str, schema: dict, context: Any = None) -> str:
        """Build extraction prompt for OpenAI API."""
        prompt_parts = [
            "Extract structured data from the following text according to the JSON schema provided.",
            "",
            "TEXT TO EXTRACT FROM:",
            text,  # GPT-4.1 can handle up to 1M tokens, no need to limit
            "",
            "JSON SCHEMA:",
            json.dumps(schema, indent=2),
            ""
        ]
        
        if context:
            if isinstance(context, dict):
                prompt_parts.extend([
                    "CONTEXT FROM PREVIOUS EXTRACTIONS:",
                    json.dumps(context, indent=2),
                    ""
                ])
            elif isinstance(context, str):
                prompt_parts.extend([
                    "CONTEXT:",
                    context,
                    ""
                ])
        
        prompt_parts.extend([
            "INSTRUCTIONS:",
            "1. Extract data that matches the schema exactly",
            "2. Use null for missing required fields",
            "3. Ensure all enum values match exactly",
            "4. Maintain proper data types (string, number, boolean, array, object)",
            "5. Return only valid JSON that conforms to the schema",
            "",
            "EXTRACTED DATA:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _calculate_field_confidence(self, result: dict, field_path: str, field_schema: dict, original_text: str) -> float:
        """Calculate confidence for a specific field."""
        try:
            from utils import get_nested_value
        except ImportError:
            from .utils import get_nested_value
        
        field_value = get_nested_value(result, field_path)
        confidence = 1.0
        
        # Check if field exists
        if field_value is None:
            if field_schema.get('required', False):
                confidence = 0.0
            else:
                confidence = 0.5
            return confidence
        
        # Check data type
        expected_type = field_schema.get('type')
        if expected_type:
            if not self._validate_type(field_value, expected_type):
                confidence *= 0.6
        
        # Check enum values
        if 'enum' in field_schema:
            if field_value not in field_schema['enum']:
                confidence *= 0.4
        
        # Check string patterns
        if expected_type == 'string' and 'pattern' in field_schema:
            import re
            if not re.match(field_schema['pattern'], str(field_value)):
                confidence *= 0.7
        
        # Check numeric ranges
        if expected_type in ['number', 'integer']:
            if 'minimum' in field_schema and field_value < field_schema['minimum']:
                confidence *= 0.5
            if 'maximum' in field_schema and field_value > field_schema['maximum']:
                confidence *= 0.5
        
        return confidence
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate value against expected JSON schema type."""
        type_mapping = {
            'string': str,
            'number': (int, float),
            'integer': int,
            'boolean': bool,
            'array': list,
            'object': dict,
            'null': type(None)
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True
    
    def _group_fields_by_level(self, extraction_order: List[str]) -> Dict[int, List[str]]:
        """Group fields by their hierarchy level."""
        levels = {}
        
        for field in extraction_order:
            level = field.count('.')
            if level not in levels:
                levels[level] = []
            levels[level].append(field)
        
        return levels
    
    def _build_level_schema(self, full_schema: dict, fields: List[str]) -> dict:
        """Build schema for a specific hierarchy level."""
        level_schema = {'type': 'object', 'properties': {}}
        
        for field_path in fields:
            parts = field_path.split('.')
            current_schema = full_schema
            current_level = level_schema
            
            # Navigate to the field in the full schema
            for part in parts[:-1]:
                if 'properties' in current_schema and part in current_schema['properties']:
                    current_schema = current_schema['properties'][part]
                    if 'properties' not in current_level:
                        current_level['properties'] = {}
                    if part not in current_level['properties']:
                        current_level['properties'][part] = {'type': 'object', 'properties': {}}
                    current_level = current_level['properties'][part]
            
            # Add the final field
            if 'properties' in current_schema and parts[-1] in current_schema['properties']:
                if 'properties' not in current_level:
                    current_level['properties'] = {}
                current_level['properties'][parts[-1]] = current_schema['properties'][parts[-1]]
        
        return level_schema
    
    def _build_chunk_context(self, previous_results: dict, dependencies: List[str]) -> str:
        """Build context string from previous extraction results."""
        if not previous_results or not dependencies:
            return ""
        
        context_data = {}
        for dep in dependencies:
            try:
                from utils import get_nested_value
            except ImportError:
                from .utils import get_nested_value
            value = get_nested_value(previous_results, dep)
            if value is not None:
                context_data[dep] = value
        
        if context_data:
            return f"Previously extracted data: {json.dumps(context_data, indent=2)}"
        
        return ""
