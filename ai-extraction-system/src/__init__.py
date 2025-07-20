"""AI Text-to-JSON Extraction System."""

from .schema_analyzer import SchemaAnalyzer
from .extraction_engine import ExtractionEngine
from .document_processor import DocumentProcessor
from .confidence_scorer import ConfidenceScorer
from .utils import (
    count_tokens, estimate_schema_tokens, validate_json_against_schema,
    flatten_schema, calculate_schema_depth, count_schema_objects,
    count_enum_values, merge_dicts_deep
)

__version__ = "1.0.0"
__all__ = [
    "SchemaAnalyzer",
    "ExtractionEngine", 
    "DocumentProcessor",
    "ConfidenceScorer",
    "count_tokens",
    "estimate_schema_tokens",
    "validate_json_against_schema",
    "flatten_schema",
    "calculate_schema_depth",
    "count_schema_objects",
    "count_enum_values",
    "merge_dicts_deep"
]
