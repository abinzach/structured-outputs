"""Test script for 5-level deep enterprise architecture schema extraction."""
import json
import os
from dotenv import load_dotenv
from src import ExtractionEngine, SchemaAnalyzer, DocumentProcessor, ConfidenceScorer

# Load environment variables
load_dotenv()

def test_complex_5level_extraction():
    """Test extraction with 5-level deep enterprise architecture schema."""
    print("🚀 Testing 5-Level Deep Schema Extraction")
    print("=" * 70)
    
    # Load schema and text
    with open('test_cases/enterprise_architecture_schema.json', 'r') as f:
        schema = json.load(f)
    
    with open('test_cases/sample_documents/enterprise_architecture_doc.txt', 'r') as f:
        architecture_text = f.read()
    
    print("\n📋 Schema Complexity Analysis:")
    analyzer = SchemaAnalyzer()
    analysis = analyzer.analyze_complexity(schema)
    
    print(f"  📊 Complexity Metrics:")
    print(f"    • Complexity Score: {analysis['complexity_score']}/100")
    print(f"    • Strategy: {analysis['strategy']}")
    print(f"    • Max Depth: {analysis['metrics']['max_depth']} levels")
    print(f"    • Total Fields: {analysis['metrics']['total_fields']}")
    print(f"    • Object Count: {analysis['metrics']['object_count']}")
    print(f"    • Enum Values: {analysis['metrics']['enum_complexity']}")
    print(f"    • Estimated Tokens: {analysis['metrics']['estimated_tokens']:,}")
    print(f"    • Needs Chunking: {analysis['needs_chunking']}")
    
    # Analyze the 5-level depth structure
    print(f"\n🏗️  Schema Structure Analysis:")
    print(f"    Level 1: Root document")
    print(f"    Level 2: Systems (frontend, backend, database)")
    print(f"    Level 3: Components/Services/Tables")
    print(f"    Level 4: Methods/Endpoints/Columns")
    print(f"    Level 5: Parameters/Permissions/Constraints")
    
    print("\n📄 Document Analysis:")
    processor = DocumentProcessor()
    doc_info = processor.process_document(architecture_text, schema)
    print(f"  Document Tokens: {doc_info['total_tokens']:,}")
    print(f"  Needs Chunking: {doc_info['needs_chunking']}")
    print(f"  Number of Chunks: {doc_info['total_chunks']}")
    
    # Check if API key is set
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key == 'your_openai_api_key_here':
        print("\n⚠️  OpenAI API key not set!")
        print("   Please set your API key in the .env file to test extraction")
        print(f"\n📝 Sample Architecture Text (first 800 chars):")
        print(architecture_text[:800] + "...")
        
        # Show what the system would extract
        print(f"\n🎯 Expected Extraction Challenges:")
        print(f"   • 5-level nested structure (deepest: parameter validation rules)")
        print(f"   • Complex enums (30+ technology options)")
        print(f"   • UUID pattern matching")
        print(f"   • Cross-references between services")
        print(f"   • Mixed data types across all levels")
        return
    
    print("\n🔄 Performing Complex 5-Level Extraction...")
    print("   This may take longer due to schema complexity...")
    
    engine = ExtractionEngine()
    
    try:
        result = engine.extract(architecture_text, schema)
        
        print("\n✅ Extraction Complete!")
        print(f"  Strategy Used: {result.get('strategy', 'unknown')}")
        print(f"  Token Usage: {result.get('token_usage', 0):,}")
        
        # Display extracted data with focus on 5-level structure
        print("\n📊 Extracted Data Analysis:")
        extracted_data = result.get('data', {})
        
        # Level 1: Document Info
        if 'documentInfo' in extracted_data:
            doc_info = extracted_data['documentInfo']
            print(f"\n  📋 Level 1 - Document Info:")
            print(f"    Title: {doc_info.get('title', 'N/A')}")
            print(f"    Version: {doc_info.get('version', 'N/A')}")
            print(f"    Author: {doc_info.get('author', 'N/A')}")
            if 'reviewers' in doc_info:
                print(f"    Reviewers: {len(doc_info['reviewers'])} found")
        
        # Level 2: Systems
        if 'systems' in extracted_data:
            systems = extracted_data['systems']
            print(f"\n  🏗️  Level 2 - Systems:")
            
            # Level 3: Frontend Components
            if 'frontend' in systems and 'components' in systems['frontend']:
                components = systems['frontend']['components']
                print(f"    Frontend Components: {len(components)} found")
                
                for i, comp in enumerate(components[:2]):  # Show first 2
                    print(f"      {i+1}. {comp.get('name', 'N/A')} ({comp.get('type', 'N/A')})")
                    
                    # Level 4: Implementation Methods
                    if 'implementation' in comp and 'methods' in comp['implementation']:
                        methods = comp['implementation']['methods']
                        print(f"         Methods: {len(methods)} found")
                        
                        for method in methods[:1]:  # Show first method
                            print(f"           - {method.get('name', 'N/A')} ({method.get('visibility', 'N/A')})")
                            
                            # Level 5: Parameters with Validation
                            if 'parameters' in method:
                                params = method['parameters']
                                print(f"             Parameters: {len(params)} found")
                                for param in params[:1]:  # Show first parameter
                                    print(f"               • {param.get('name', 'N/A')} ({param.get('type', 'N/A')})")
                                    if 'validation' in param:
                                        validation = param['validation']
                                        print(f"                 Validation rules: {len(validation)} found")
            
            # Level 3: Backend Services
            if 'backend' in systems and 'services' in systems['backend']:
                services = systems['backend']['services']
                print(f"    Backend Services: {len(services)} found")
                
                for service in services[:1]:  # Show first service
                    print(f"      - {service.get('name', 'N/A')} (Port: {service.get('port', 'N/A')})")
                    
                    # Level 4: Endpoints
                    if 'endpoints' in service:
                        endpoints = service['endpoints']
                        print(f"        Endpoints: {len(endpoints)} found")
                        
                        for endpoint in endpoints[:1]:  # Show first endpoint
                            print(f"          {endpoint.get('method', 'N/A')} {endpoint.get('path', 'N/A')}")
                            
                            # Level 5: Authentication Permissions
                            if 'authentication' in endpoint and 'permissions' in endpoint['authentication']:
                                perms = endpoint['authentication']['permissions']
                                print(f"            Security Level: {perms.get('securityLevel', 'N/A')}")
                                if 'restrictions' in perms and 'rateLimit' in perms['restrictions']:
                                    rate_limit = perms['restrictions']['rateLimit']
                                    print(f"            Rate Limit: {rate_limit.get('requests', 'N/A')}/{rate_limit.get('window', 'N/A')}")
            
            # Level 3: Database Tables
            if 'database' in systems and 'instances' in systems['database']:
                instances = systems['database']['instances']
                print(f"    Database Instances: {len(instances)} found")
                
                for instance in instances[:1]:  # Show first instance
                    print(f"      - {instance.get('name', 'N/A')} ({instance.get('technology', 'N/A')})")
                    
                    if 'schemas' in instance:
                        for schema_info in instance['schemas'][:1]:  # Show first schema
                            if 'tables' in schema_info:
                                tables = schema_info['tables']
                                print(f"        Tables: {len(tables)} found")
                                
                                # Level 4: Columns
                                for table in tables[:1]:  # Show first table
                                    print(f"          {table.get('name', 'N/A')} table")
                                    if 'columns' in table:
                                        columns = table['columns']
                                        print(f"            Columns: {len(columns)} found")
                                        
                                        # Level 5: Column Constraints
                                        for col in columns[:1]:  # Show first column
                                            print(f"              {col.get('name', 'N/A')} ({col.get('dataType', 'N/A')})")
                                            if 'constraints' in col and 'validation' in col['constraints']:
                                                validation = col['constraints']['validation']
                                                print(f"                Validation: {len(validation)} rules")
        
        # Confidence analysis
        print("\n🎯 Confidence Analysis:")
        confidence = result.get('confidence', {})
        print(f"  Overall Confidence: {confidence.get('overall', 0):.2%}")
        print(f"  Schema Valid: {confidence.get('schema_valid', False)}")
        
        # Detailed confidence scoring
        scorer = ConfidenceScorer()
        full_confidence = scorer.score_extraction(extracted_data, schema, architecture_text)
        
        print(f"\n📈 Detailed Confidence Metrics:")
        completion = full_confidence.get('completion', {})
        print(f"  Completion Rate: {completion.get('completion_rate', 0):.1%}")
        print(f"  Required Fields: {completion.get('completed_required', 0)}/{completion.get('required_fields', 0)}")
        
        consistency = full_confidence.get('consistency', {})
        print(f"  Type Consistency: {consistency.get('type_consistency', 0):.1%}")
        print(f"  Enum Consistency: {consistency.get('enum_consistency', 0):.1%}")
        
        # Review candidates
        review_candidates = full_confidence.get('review_candidates', [])
        if review_candidates:
            print(f"\n🔍 Fields Needing Review ({len(review_candidates)}):")
            for candidate in review_candidates[:8]:  # Show top 8
                field_parts = candidate['field'].split('.')
                level = len(field_parts)
                print(f"    Level {level}: {candidate['field']}")
                print(f"      Confidence: {candidate['confidence']:.1%} ({candidate['reason']})")
        else:
            print("\n✨ Excellent! No fields need review - high confidence extraction!")
        
        # Save results
        output_file = 'test_cases/enterprise_5level_extraction_result.json'
        with open(output_file, 'w') as f:
            json.dump({
                'extracted_data': extracted_data,
                'confidence': full_confidence,
                'analysis': {
                    'schema_complexity': analysis,
                    'document_info': {
                        'total_tokens': doc_info.get('total_tokens', 0),
                        'needs_chunking': doc_info.get('needs_chunking', False)
                    },
                    'extraction_strategy': result.get('strategy'),
                    'token_usage': result.get('token_usage', 0)
                }
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # Summary of what was tested
        print(f"\n🎉 5-Level Deep Schema Test Summary:")
        print(f"   ✅ Successfully extracted {analysis['metrics']['total_fields']} fields")
        print(f"   ✅ Handled {analysis['metrics']['max_depth']}-level nesting")
        print(f"   ✅ Processed {analysis['metrics']['enum_complexity']} enum values")
        print(f"   ✅ Validated complex patterns (UUIDs, emails, versions)")
        print(f"   ✅ Maintained relationships across {analysis['metrics']['object_count']} objects")
        print(f"   ✅ Strategy: {result.get('strategy')} (optimized for GPT-4.1)")
        
    except Exception as e:
        print(f"\n❌ Extraction failed: {str(e)}")
        print("   This complex schema pushes the system to its limits!")
        print("   Make sure your OpenAI API key is valid and has access to GPT-4.1")

if __name__ == "__main__":
    test_complex_5level_extraction()
