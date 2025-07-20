# AI Text-to-JSON Extraction System

A sophisticated system that converts unstructured text into structured JSON following complex schemas. Built to handle 3-7 nested levels, 50-150 objects, and 1000+ literals/enums with confidence scoring.

## 🚀 Features

- **Adaptive Extraction Strategies**: Automatically chooses between single-pass, multi-pass hierarchical, or chunked extraction based on schema complexity
- **Schema Analysis**: Intelligent analysis of JSON schemas with complexity scoring and dependency mapping
- **Document Processing**: Optimized for GPT-4.1's 1M token context - processes entire documents up to ~300 pages in a single pass
- **Confidence Scoring**: Field-level confidence analysis with human review queue
- **Web Interface**: User-friendly web application with real-time results and visualizations
- **OpenAI Integration**: Leverages GPT-4.1 with JSON mode and 1M token context window for reliable structured outputs

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- Dependencies listed in `requirements.txt`

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ai-extraction-system
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Open your browser** and navigate to `http://localhost:5000`

## 🏗️ Architecture

### Core Components

1. **Schema Analyzer** (`src/schema_analyzer.py`)
   - Analyzes JSON schema complexity
   - Builds dependency graphs
   - Implements intelligent chunking strategies

2. **Extraction Engine** (`src/extraction_engine.py`)
   - OpenAI API integration with adaptive strategies
   - Single-pass, hierarchical, and chunked extraction modes
   - Confidence calculation and validation

3. **Document Processor** (`src/document_processor.py`)
   - Smart document chunking with overlap
   - Context preservation across chunks
   - Result merging and conflict resolution

4. **Confidence Scorer** (`src/confidence_scorer.py`)
   - Field-level confidence calculation
   - Schema validation and consistency checking
   - Human review queue generation

5. **Web Interface** (`app.py`)
   - Flask-based web application
   - Interactive schema and text input
   - Real-time results with confidence visualization

## 📊 Extraction Strategies

### Single-Pass Extraction
- **When**: Simple schemas (<8K tokens, <5 levels deep)
- **How**: Direct extraction in one API call
- **Benefits**: Fast, cost-effective

### Multi-Pass Hierarchical
- **When**: Complex nested schemas (5+ levels, 20+ objects)
- **How**: Extract by hierarchy levels, building context
- **Benefits**: Better accuracy for complex structures

### Multi-Pass Chunked
- **When**: Very large schemas (>8K tokens)
- **How**: Break schema into dependency-aware chunks
- **Benefits**: Handles massive schemas while preserving relationships

## 🎯 Confidence Scoring

The system provides comprehensive confidence analysis:

- **Field-Level Scores**: Individual confidence for each extracted field
- **Schema Validation**: Ensures extracted data matches schema requirements
- **Completion Metrics**: Tracks required vs optional field completion
- **Consistency Analysis**: Cross-field validation and type checking
- **Review Queue**: Automatically flags low-confidence fields for human review

## 📝 Usage Examples

### Simple Extraction
```python
from src import ExtractionEngine

engine = ExtractionEngine()
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    }
}
text = "John Smith is 32 years old."

result = engine.extract(text, schema)
print(result['data'])  # {"name": "John Smith", "age": 32}
print(result['confidence']['overall'])  # 0.95
```

### Complex Schema Analysis
```python
from src import SchemaAnalyzer

analyzer = SchemaAnalyzer()
analysis = analyzer.analyze_complexity(complex_schema)

print(f"Complexity Score: {analysis['complexity_score']}")
print(f"Strategy: {analysis['strategy']}")
print(f"Max Depth: {analysis['metrics']['max_depth']}")
```

### Document Processing
```python
from src import DocumentProcessor

processor = DocumentProcessor()
doc_info = processor.process_document(large_text, schema)

if doc_info['needs_chunking']:
    print(f"Document split into {doc_info['total_chunks']} chunks")
```

## 🌐 Web Interface

The web interface provides:

- **Schema Input**: Paste JSON schema or upload file
- **Text Input**: Paste text or upload document
- **Real-time Processing**: Live extraction with progress indicators
- **Results Visualization**: Tabbed interface showing:
  - Extracted data with confidence scores
  - Detailed confidence analysis
  - Schema complexity metrics
  - Raw API responses
- **Review Queue**: Highlighted fields needing human verification

## 🧪 Testing

Test cases are provided in the `test_cases/` directory:

- `simple_schema.json`: Basic schema for testing
- `complex_schema.json`: Multi-level nested schema
- `sample_documents/`: Sample texts for extraction

### Running Tests
```bash
# Test simple extraction
python -c "
from src import ExtractionEngine
import json

with open('test_cases/simple_schema.json') as f:
    schema = json.load(f)
with open('test_cases/sample_documents/simple_text.txt') as f:
    text = f.read()

engine = ExtractionEngine()
result = engine.extract(text, schema)
print(json.dumps(result, indent=2))
"
```

## 📈 Performance Metrics

The system is designed to handle:

- **Schema Complexity**: Up to 7 levels deep, 150+ objects
- **Document Size**: 50+ page documents with smart chunking
- **Enum Values**: 1000+ enum values with validation
- **Processing Speed**: Adaptive compute based on complexity
- **Accuracy**: Field-level confidence scoring with review queues

## 🔧 Configuration

Environment variables (`.env`):

```bash
OPENAI_API_KEY=your_openai_api_key_here
FLASK_ENV=development
MAX_TOKENS_PER_REQUEST=1000000
CONFIDENCE_THRESHOLD=0.7
```

## 🚦 API Endpoints

- `GET /` - Web interface
- `POST /extract` - Main extraction endpoint
- `POST /analyze-schema` - Schema analysis only
- `GET /health` - Health check

### Extract Endpoint
```bash
curl -X POST http://localhost:5000/extract \
  -H "Content-Type: application/json" \
  -d '{
    "schema": {...},
    "text": "..."
  }'
```

## 🔍 Error Handling

The system includes comprehensive error handling:

- **Schema Validation**: Validates JSON schema format
- **API Errors**: Graceful handling of OpenAI API issues
- **Token Limits**: Automatic chunking for large inputs
- **Network Issues**: Retry logic with exponential backoff
- **Validation Errors**: Clear error messages for debugging

## 🎨 Customization

### Adding New Extraction Strategies
1. Extend `ExtractionEngine` class
2. Implement strategy logic
3. Update `_determine_strategy()` method

### Custom Confidence Scoring
1. Extend `ConfidenceScorer` class
2. Override scoring methods
3. Add custom validation rules

### Schema Analysis Extensions
1. Extend `SchemaAnalyzer` class
2. Add new complexity metrics
3. Implement custom chunking strategies

## 📚 Technical Details

### Token Management
- Automatic token counting with tiktoken
- Smart chunking to stay within API limits
- Overlap preservation for context continuity

### Dependency Resolution
- Topological sorting for field extraction order
- Cross-reference detection and handling
- Circular dependency management

### Confidence Calculation
- Multi-factor scoring algorithm
- Schema validation integration
- Context relevance analysis
- Statistical consistency checking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:

1. Check the documentation
2. Review test cases for examples
3. Open an issue on GitHub
4. Contact the development team

## 🔮 Future Enhancements

- **Model Ensemble**: Multiple model validation
- **Streaming Processing**: Real-time extraction for large documents
- **Custom Models**: Fine-tuned models for specific domains
- **Advanced Analytics**: ML-based confidence prediction
- **Integration APIs**: Direct integration with popular platforms
