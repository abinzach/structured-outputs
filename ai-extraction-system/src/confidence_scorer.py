"""Confidence Scoring System for extraction results."""
import os
from typing import Dict, List, Any, Tuple
try:
    from utils import validate_json_against_schema, flatten_schema, get_nested_value
except ImportError:
    from .utils import validate_json_against_schema, flatten_schema, get_nested_value


class ConfidenceScorer:
    """Calculates confidence scores for extraction results."""
    
    def __init__(self, confidence_threshold: float = None):
        self.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', 0.7))
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
    
    def score_extraction(self, result: dict, schema: dict, original_text: str = "") -> dict:
        """
        Calculate overall extraction confidence metrics.
        
        Args:
            result: Extracted data
            schema: Original JSON schema
            original_text: Source text for context analysis
            
        Returns:
            dict: Comprehensive confidence analysis
        """
        # Validate against schema
        is_valid, validation_error = validate_json_against_schema(result, schema)
        
        # Calculate field-level confidence scores
        field_scores = self._calculate_field_scores(result, schema, original_text)
        
        # Calculate completion metrics
        completion_metrics = self._calculate_completion_metrics(result, schema)
        
        # Calculate consistency scores
        consistency_scores = self._calculate_consistency_scores(result, schema)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            field_scores, completion_metrics, consistency_scores, is_valid
        )
        
        # Identify review candidates
        review_candidates = self.get_review_candidates(field_scores, self.confidence_threshold)
        
        return {
            'overall': overall_confidence,
            'fields': field_scores,
            'completion': completion_metrics,
            'consistency': consistency_scores,
            'schema_valid': is_valid,
            'validation_error': validation_error if not is_valid else None,
            'review_candidates': review_candidates,
            'confidence_threshold': self.confidence_threshold
        }
    
    def score_field(self, field_value: Any, field_schema: dict, extraction_context: dict = None) -> float:
        """
        Calculate confidence score for individual field.
        
        Args:
            field_value: The extracted field value
            field_schema: Schema definition for the field
            extraction_context: Additional context for scoring
            
        Returns:
            float: Confidence score (0.0-1.0)
        """
        if extraction_context is None:
            extraction_context = {}
        
        confidence = 1.0
        
        # Check if field exists
        if field_value is None:
            if field_schema.get('required', False):
                return 0.0
            else:
                return 0.5  # Optional field missing
        
        # Data type validation
        confidence *= self._score_data_type(field_value, field_schema)
        
        # Enum validation
        confidence *= self._score_enum_match(field_value, field_schema)
        
        # Pattern validation
        confidence *= self._score_pattern_match(field_value, field_schema)
        
        # Range validation
        confidence *= self._score_range_validation(field_value, field_schema)
        
        # Length validation
        confidence *= self._score_length_validation(field_value, field_schema)
        
        # Context relevance (if available)
        if 'original_text' in extraction_context:
            confidence *= self._score_context_relevance(
                field_value, extraction_context['original_text']
            )
        
        return max(0.0, min(1.0, confidence))
    
    def get_review_candidates(self, field_scores: dict, threshold: float = None) -> List[dict]:
        """
        Identify fields that need human review.
        
        Args:
            field_scores: Dictionary of field confidence scores
            threshold: Confidence threshold for review (defaults to instance setting)
            
        Returns:
            List of fields needing review with metadata
        """
        if threshold is None:
            threshold = self.confidence_threshold
        
        review_candidates = []
        
        for field_path, score in field_scores.items():
            if score < threshold:
                review_candidates.append({
                    'field': field_path,
                    'confidence': score,
                    'reason': self._get_review_reason(score),
                    'priority': self._get_review_priority(score)
                })
        
        # Sort by priority (lowest confidence first)
        review_candidates.sort(key=lambda x: x['confidence'])
        
        return review_candidates
    
    def _calculate_field_scores(self, result: dict, schema: dict, original_text: str) -> dict:
        """Calculate confidence scores for all fields."""
        field_scores = {}
        flattened_schema = flatten_schema(schema)
        
        for field_path, field_schema in flattened_schema.items():
            field_value = get_nested_value(result, field_path)
            
            extraction_context = {
                'original_text': original_text,
                'schema': field_schema,
                'full_result': result
            }
            
            field_scores[field_path] = self.score_field(
                field_value, field_schema, extraction_context
            )
        
        return field_scores
    
    def _calculate_completion_metrics(self, result: dict, schema: dict) -> dict:
        """Calculate completion metrics for the extraction."""
        flattened_schema = flatten_schema(schema)
        
        total_fields = len(flattened_schema)
        completed_fields = 0
        required_fields = 0
        completed_required = 0
        
        for field_path, field_schema in flattened_schema.items():
            field_value = get_nested_value(result, field_path)
            is_required = field_schema.get('required', False)
            
            if is_required:
                required_fields += 1
                if field_value is not None:
                    completed_required += 1
            
            if field_value is not None:
                completed_fields += 1
        
        completion_rate = completed_fields / total_fields if total_fields > 0 else 0.0
        required_completion_rate = completed_required / required_fields if required_fields > 0 else 1.0
        
        return {
            'total_fields': total_fields,
            'completed_fields': completed_fields,
            'required_fields': required_fields,
            'completed_required': completed_required,
            'completion_rate': completion_rate,
            'required_completion_rate': required_completion_rate
        }
    
    def _calculate_consistency_scores(self, result: dict, schema: dict) -> dict:
        """Calculate consistency scores across the extraction."""
        flattened_result = self._flatten_dict(result)
        flattened_schema = flatten_schema(schema)
        
        # Type consistency
        type_consistency = self._calculate_type_consistency(flattened_result, flattened_schema)
        
        # Enum consistency
        enum_consistency = self._calculate_enum_consistency(flattened_result, flattened_schema)
        
        # Cross-field consistency
        cross_field_consistency = self._calculate_cross_field_consistency(result, schema)
        
        overall_consistency = (type_consistency + enum_consistency + cross_field_consistency) / 3
        
        return {
            'type_consistency': type_consistency,
            'enum_consistency': enum_consistency,
            'cross_field_consistency': cross_field_consistency,
            'overall': overall_consistency
        }
    
    def _calculate_overall_confidence(self, field_scores: dict, completion_metrics: dict, 
                                    consistency_scores: dict, is_schema_valid: bool) -> float:
        """Calculate overall confidence score."""
        if not field_scores:
            return 0.0
        
        # Average field confidence (weighted by importance)
        field_confidence = sum(field_scores.values()) / len(field_scores)
        
        # Completion bonus/penalty
        completion_factor = (
            completion_metrics['completion_rate'] * 0.3 +
            completion_metrics['required_completion_rate'] * 0.7
        )
        
        # Consistency factor
        consistency_factor = consistency_scores['overall']
        
        # Schema validation factor
        schema_factor = 1.0 if is_schema_valid else 0.5
        
        # Weighted combination
        overall = (
            field_confidence * 0.5 +
            completion_factor * 0.2 +
            consistency_factor * 0.2 +
            schema_factor * 0.1
        )
        
        return max(0.0, min(1.0, overall))
    
    def _score_data_type(self, value: Any, schema: dict) -> float:
        """Score data type correctness."""
        expected_type = schema.get('type')
        if not expected_type:
            return 1.0
        
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
        if expected_python_type and isinstance(value, expected_python_type):
            return 1.0
        
        # Partial credit for compatible types
        if expected_type == 'number' and isinstance(value, int):
            return 1.0
        if expected_type == 'string' and isinstance(value, (int, float, bool)):
            return 0.8  # Can be converted to string
        
        return 0.3
    
    def _score_enum_match(self, value: Any, schema: dict) -> float:
        """Score enum value matching."""
        if 'enum' not in schema:
            return 1.0
        
        enum_values = schema['enum']
        if value in enum_values:
            return 1.0
        
        # Check for case-insensitive match for strings
        if isinstance(value, str):
            for enum_val in enum_values:
                if isinstance(enum_val, str) and value.lower() == enum_val.lower():
                    return 0.9
        
        return 0.2
    
    def _score_pattern_match(self, value: Any, schema: dict) -> float:
        """Score pattern matching for strings."""
        if 'pattern' not in schema or not isinstance(value, str):
            return 1.0
        
        import re
        try:
            if re.match(schema['pattern'], value):
                return 1.0
            else:
                return 0.4
        except re.error:
            return 1.0  # Invalid pattern, don't penalize
    
    def _score_range_validation(self, value: Any, schema: dict) -> float:
        """Score numeric range validation."""
        if not isinstance(value, (int, float)):
            return 1.0
        
        score = 1.0
        
        if 'minimum' in schema and value < schema['minimum']:
            score *= 0.3
        if 'maximum' in schema and value > schema['maximum']:
            score *= 0.3
        if 'exclusiveMinimum' in schema and value <= schema['exclusiveMinimum']:
            score *= 0.3
        if 'exclusiveMaximum' in schema and value >= schema['exclusiveMaximum']:
            score *= 0.3
        
        return score
    
    def _score_length_validation(self, value: Any, schema: dict) -> float:
        """Score length validation for strings and arrays."""
        if not isinstance(value, (str, list)):
            return 1.0
        
        length = len(value)
        score = 1.0
        
        if 'minLength' in schema and length < schema['minLength']:
            score *= 0.5
        if 'maxLength' in schema and length > schema['maxLength']:
            score *= 0.5
        
        return score
    
    def _score_context_relevance(self, value: Any, original_text: str) -> float:
        """Score how relevant the extracted value is to the original text."""
        if not isinstance(value, str) or not original_text:
            return 1.0
        
        # Simple relevance check: is the value mentioned in the text?
        value_lower = str(value).lower()
        text_lower = original_text.lower()
        
        if value_lower in text_lower:
            return 1.0
        
        # Check for partial matches
        words = value_lower.split()
        if len(words) > 1:
            matches = sum(1 for word in words if word in text_lower)
            return 0.5 + (matches / len(words)) * 0.5
        
        return 0.7  # Neutral score if no clear relevance
    
    def _calculate_type_consistency(self, result: dict, schema: dict) -> float:
        """Calculate type consistency across fields."""
        if not result:
            return 1.0
        
        consistent_fields = 0
        total_fields = 0
        
        for field_path, value in result.items():
            if field_path in schema:
                field_schema = schema[field_path]
                if self._score_data_type(value, field_schema) >= 0.8:
                    consistent_fields += 1
                total_fields += 1
        
        return consistent_fields / total_fields if total_fields > 0 else 1.0
    
    def _calculate_enum_consistency(self, result: dict, schema: dict) -> float:
        """Calculate enum consistency across fields."""
        if not result:
            return 1.0
        
        consistent_fields = 0
        enum_fields = 0
        
        for field_path, value in result.items():
            if field_path in schema:
                field_schema = schema[field_path]
                if 'enum' in field_schema:
                    enum_fields += 1
                    if self._score_enum_match(value, field_schema) >= 0.8:
                        consistent_fields += 1
        
        return consistent_fields / enum_fields if enum_fields > 0 else 1.0
    
    def _calculate_cross_field_consistency(self, result: dict, schema: dict) -> float:
        """Calculate consistency between related fields."""
        # This is a simplified implementation
        # In practice, you might define specific consistency rules
        
        # For now, return a neutral score
        return 0.8
    
    def _flatten_dict(self, d: dict, parent_key: str = '', sep: str = '.') -> dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _get_review_reason(self, confidence: float) -> str:
        """Get human-readable reason for review."""
        if confidence < 0.3:
            return "Very low confidence - likely extraction error"
        elif confidence < 0.5:
            return "Low confidence - validation failed"
        elif confidence < 0.7:
            return "Medium confidence - may need verification"
        else:
            return "Below threshold - minor issues detected"
    
    def _get_review_priority(self, confidence: float) -> str:
        """Get review priority based on confidence."""
        if confidence < 0.3:
            return "high"
        elif confidence < 0.5:
            return "medium"
        else:
            return "low"
