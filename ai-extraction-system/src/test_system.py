#!/usr/bin/env python3
"""
Test script for the AI Text-to-JSON Extraction System.
This script tests the core functionality without requiring an OpenAI API key.
"""

import json
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from schema_analyzer import SchemaAnalyzer
from document_processor import DocumentProcessor
from confidence_scorer import ConfidenceScorer
from utils import (
    count_tokens, estimate_schema_tokens, validate_json_against_schema,
    flatten_schema, calculate_schema_depth, count_schema_objects,
    count_enum_values
)


def test_schema_analyzer():
    """Test the Schema Analyzer component."""
    print("🔍 Testing Schema Analyzer...")
    
    # Load test schema
    with open('test_cases/simple_schema.json', 'r') as f:
        simple_schema = json.load(f)
    
    with open('test_cases/complex_schema.json', 'r') as f:
        complex_schema = json.load(f)
    
    analyzer = SchemaAnalyzer()
    
    # Test simple schema
    simple_analysis = analyzer.analyze_complexity(simple_schema)
    print(f"  Simple Schema - Complexity Score: {simple_analysis['complexity_score']}")
    print(f"  Simple Schema - Strategy: {simple_analysis['strategy']}")
    print(f"  Simple Schema - Max Depth: {simple_analysis['metrics']['max_depth']}")
    
    # Test complex schema
    complex_analysis = analyzer.analyze_complexity(complex_schema)
    print(f"  Complex Schema - Complexity Score: {complex_analysis['complexity_score']}")
    print(f"  Complex Schema - Strategy: {complex_analysis['strategy']}")
    print(f"  Complex Schema - Max Depth: {complex_analysis['metrics']['max_depth']}")
    print(f"  Complex Schema - Total Fields: {complex_analysis['metrics']['total_fields']}")
    
    # Test chunking
    if complex_analysis['needs_chunking']:
        chunks = analyzer.chunk_schema(complex_schema)
        print(f"  Complex Schema - Chunks: {len(chunks)}")
    
    print("  ✅ Schema Analyzer tests passed!")
    return True


def test_document_processor():
    """Test the Document Processor component."""
    print("\n📄 Testing Document Processor...")
    
    # Load test documents
    with open('test_cases/sample_documents/simple_text.txt', 'r') as f:
        simple_text = f.read()
    
    with open('test_cases/sample_documents/complex_text.txt', 'r') as f:
        complex_text = f.read()
    
    with open('test_cases/simple_schema.json', 'r') as f:
        schema = json.load(f)
    
    processor = DocumentProcessor()
    
    # Test simple document
    simple_doc_info = processor.process_document(simple_text, schema)
    print(f"  Simple Text - Needs Chunking: {simple_doc_info['needs_chunking']}")
    print(f"  Simple Text - Total Tokens: {simple_doc_info['total_tokens']}")
    
    # Test complex document
    complex_doc_info = processor.process_document(complex_text, schema)
    print(f"  Complex Text - Needs Chunking: {complex_doc_info['needs_chunking']}")
    print(f"  Complex Text - Total Tokens: {complex_doc_info['total_tokens']}")
    
    if complex_doc_info['needs_chunking']:
        print(f"  Complex Text - Chunks: {complex_doc_info['total_chunks']}")
    
    print("  ✅ Document Processor tests passed!")
    return True


def test_confidence_scorer():
    """Test the Confidence Scorer component."""
    print("\n🎯 Testing Confidence Scorer...")
    
    # Load test schema
    with open('test_cases/simple_schema.json', 'r') as f:
        schema = json.load(f)
    
    # Mock extraction result
    mock_result = {
        "name": "John Smith",
        "age": 32,
        "email": "john.smith@techcorp.com",
        "occupation": "Software Engineer",
        "skills": ["Python", "JavaScript", "Cloud Computing"],
        "active": True
    }
    
    scorer = ConfidenceScorer()
    
    # Test confidence scoring
    confidence_analysis = scorer.score_extraction(mock_result, schema, "test text")
    print(f"  Overall Confidence: {confidence_analysis['overall']:.2f}")
    print(f"  Schema Valid: {confidence_analysis['schema_valid']}")
    print(f"  Field Scores: {len(confidence_analysis['fields'])} fields analyzed")
    print(f"  Review Candidates: {len(confidence_analysis['review_candidates'])}")
    
    # Test individual field scoring
    field_score = scorer.score_field("john.smith@techcorp.com", schema['properties']['email'])
    print(f"  Email Field Score: {field_score:.2f}")
    
    print("  ✅ Confidence Scorer tests passed!")
    return True


def test_utils():
    """Test utility functions."""
    print("\n🛠️ Testing Utility Functions...")
    
    # Load test schema
    with open('test_cases/complex_schema.json', 'r') as f:
        schema = json.load(f)
    
    # Test token counting
    test_text = "This is a test sentence for token counting."
    token_count = count_tokens(test_text)
    print(f"  Token Count: {token_count} tokens")
    
    # Test schema token estimation
    schema_tokens = estimate_schema_tokens(schema)
    print(f"  Schema Tokens: {schema_tokens} tokens")
    
    # Test schema analysis functions
    depth = calculate_schema_depth(schema)
    objects = count_schema_objects(schema)
    enums = count_enum_values(schema)
    
    print(f"  Schema Depth: {depth}")
    print(f"  Schema Objects: {objects}")
    print(f"  Enum Values: {enums}")
    
    # Test schema flattening
    flattened = flatten_schema(schema)
    print(f"  Flattened Fields: {len(flattened)}")
    
    # Test validation
    test_data = {"company": {"name": "Test Corp", "industry": "Technology"}}
    is_valid, error = validate_json_against_schema(test_data, schema)
    print(f"  Validation Test: {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    print("  ✅ Utility function tests passed!")
    return True


def test_integration():
    """Test integration between components."""
    print("\n🔗 Testing Component Integration...")
    
    # Load test data
    with open('test_cases/simple_schema.json', 'r') as f:
        schema = json.load(f)
    
    with open('test_cases/sample_documents/simple_text.txt', 'r') as f:
        text = f.read()
    
    # Test full pipeline (without OpenAI API)
    analyzer = SchemaAnalyzer()
    processor = DocumentProcessor()
    scorer = ConfidenceScorer()
    
    # Analyze schema
    schema_analysis = analyzer.analyze_complexity(schema)
    print(f"  Schema Analysis: {schema_analysis['strategy']} strategy recommended")
    
    # Process document
    doc_info = processor.process_document(text, schema)
    print(f"  Document Processing: {doc_info['total_chunks']} chunk(s)")
    
    # Mock extraction result for confidence testing
    mock_result = {
        "name": "John Smith",
        "age": 32,
        "email": "john.smith@techcorp.com",
        "occupation": "Software Engineer",
        "skills": ["Python", "JavaScript", "Cloud Computing"],
        "active": True
    }
    
    # Score confidence
    confidence = scorer.score_extraction(mock_result, schema, text)
    print(f"  Confidence Scoring: {confidence['overall']:.2f} overall confidence")
    
    print("  ✅ Integration tests passed!")
    return True


def main():
    """Run all tests."""
    print("🚀 AI Text-to-JSON Extraction System - Test Suite")
    print("=" * 60)
    
    tests = [
        test_utils,
        test_schema_analyzer,
        test_document_processor,
        test_confidence_scorer,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ❌ Test failed with error: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\n💡 Next steps:")
        print("   1. Set your OpenAI API key in the .env file")
        print("   2. Run 'python app.py' to start the web interface")
        print("   3. Open http://localhost:5000 in your browser")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
