"""Test script for resume extraction using the JSON Resume Schema."""
import json
import os
from dotenv import load_dotenv
from src import ExtractionEngine, SchemaAnalyzer, DocumentProcessor, ConfidenceScorer

# Load environment variables
load_dotenv()

def test_resume_extraction():
    """Test extraction with resume schema and sample text."""
    print("🚀 Testing Resume Extraction System")
    print("=" * 60)
    
    # Load schema and text
    with open('test_cases/resume_schema.json', 'r') as f:
        schema = json.load(f)
    
    with open('test_cases/sample_documents/resume_sample.txt', 'r') as f:
        resume_text = f.read()
    
    print("\n📋 Schema Analysis:")
    analyzer = SchemaAnalyzer()
    analysis = analyzer.analyze_complexity(schema)
    print(f"  Complexity Score: {analysis['complexity_score']}")
    print(f"  Strategy: {analysis['strategy']}")
    print(f"  Max Depth: {analysis['metrics']['max_depth']}")
    print(f"  Total Fields: {analysis['metrics']['total_fields']}")
    print(f"  Estimated Tokens: {analysis['metrics']['estimated_tokens']}")
    
    print("\n📄 Document Analysis:")
    processor = DocumentProcessor()
    doc_info = processor.process_document(resume_text, schema)
    print(f"  Document Tokens: {doc_info['total_tokens']}")
    print(f"  Needs Chunking: {doc_info['needs_chunking']}")
    print(f"  Number of Chunks: {doc_info['total_chunks']}")
    
    # Check if API key is set
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key == 'your_openai_api_key_here':
        print("\n⚠️  OpenAI API key not set!")
        print("   Please set your API key in the .env file to test extraction")
        print("\n📝 Sample Resume Text (first 500 chars):")
        print(resume_text[:500] + "...")
        return
    
    print("\n🔄 Performing Extraction...")
    engine = ExtractionEngine()
    
    try:
        result = engine.extract(resume_text, schema)
        
        print("\n✅ Extraction Complete!")
        print(f"  Strategy Used: {result.get('strategy', 'unknown')}")
        print(f"  Token Usage: {result.get('token_usage', 0)}")
        
        # Display extracted data
        print("\n📊 Extracted Data:")
        extracted_data = result.get('data', {})
        
        # Show key fields
        if 'basics' in extracted_data:
            basics = extracted_data['basics']
            print(f"\n  Basic Information:")
            print(f"    Name: {basics.get('name', 'N/A')}")
            print(f"    Label: {basics.get('label', 'N/A')}")
            print(f"    Email: {basics.get('email', 'N/A')}")
            print(f"    Phone: {basics.get('phone', 'N/A')}")
            
            if 'location' in basics:
                loc = basics['location']
                print(f"    Location: {loc.get('city', '')}, {loc.get('region', '')} {loc.get('countryCode', '')}")
        
        if 'work' in extracted_data and extracted_data['work']:
            print(f"\n  Work Experience: {len(extracted_data['work'])} positions found")
            for i, job in enumerate(extracted_data['work'][:2]):  # Show first 2
                print(f"    {i+1}. {job.get('position', 'N/A')} at {job.get('name', 'N/A')}")
        
        if 'education' in extracted_data and extracted_data['education']:
            print(f"\n  Education: {len(extracted_data['education'])} degrees found")
            for edu in extracted_data['education']:
                print(f"    - {edu.get('studyType', 'N/A')} in {edu.get('area', 'N/A')} from {edu.get('institution', 'N/A')}")
        
        if 'skills' in extracted_data and extracted_data['skills']:
            print(f"\n  Skills: {len(extracted_data['skills'])} skill categories found")
        
        # Confidence analysis
        print("\n🎯 Confidence Analysis:")
        confidence = result.get('confidence', {})
        print(f"  Overall Confidence: {confidence.get('overall', 0):.2%}")
        print(f"  Schema Valid: {confidence.get('schema_valid', False)}")
        
        # Review candidates
        scorer = ConfidenceScorer()
        full_confidence = scorer.score_extraction(extracted_data, schema, resume_text)
        review_candidates = full_confidence.get('review_candidates', [])
        
        if review_candidates:
            print(f"\n  Fields Needing Review ({len(review_candidates)}):")
            for candidate in review_candidates[:5]:  # Show top 5
                print(f"    - {candidate['field']}: {candidate['confidence']:.2%} ({candidate['reason']})")
        else:
            print("  No fields need review - high confidence extraction!")
        
        # Save results
        output_file = 'test_cases/resume_extraction_result.json'
        with open(output_file, 'w') as f:
            json.dump({
                'extracted_data': extracted_data,
                'confidence': confidence,
                'analysis': {
                    'schema_complexity': analysis,
                    'document_info': {
                        'total_tokens': doc_info['total_tokens'],
                        'needs_chunking': doc_info['needs_chunking']
                    }
                }
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Extraction failed: {str(e)}")
        print("   Make sure your OpenAI API key is valid and has access to GPT-4.1")

if __name__ == "__main__":
    test_resume_extraction()
