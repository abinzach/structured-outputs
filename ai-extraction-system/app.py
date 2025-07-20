"""Flask Web Application for AI Text-to-JSON Extraction System."""
import os
import json
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from src import ExtractionEngine, DocumentProcessor, ConfidenceScorer, SchemaAnalyzer

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize components
extraction_engine = ExtractionEngine()
document_processor = DocumentProcessor()
confidence_scorer = ConfidenceScorer()
schema_analyzer = SchemaAnalyzer()

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Text-to-JSON Extraction System</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        textarea, input[type="file"] {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-family: monospace;
        }
        textarea {
            min-height: 150px;
            resize: vertical;
        }
        button {
            background-color: #007bff;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin-right: 10px;
        }
        button:hover {
            background-color: #0056b3;
        }
        button:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        .results {
            margin-top: 30px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 5px;
            border-left: 4px solid #007bff;
        }
        .confidence-score {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 3px;
            color: white;
            font-weight: bold;
            margin-left: 10px;
        }
        .confidence-high { background-color: #28a745; }
        .confidence-medium { background-color: #ffc107; color: #000; }
        .confidence-low { background-color: #dc3545; }
        .review-candidates {
            margin-top: 20px;
            padding: 15px;
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
        }
        .error {
            color: #dc3545;
            background-color: #f8d7da;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
        }
        .loading {
            text-align: center;
            color: #007bff;
            font-style: italic;
        }
        .tabs {
            display: flex;
            margin-bottom: 20px;
            border-bottom: 1px solid #ddd;
        }
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border-bottom: 2px solid transparent;
        }
        .tab.active {
            border-bottom-color: #007bff;
            color: #007bff;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        pre {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            white-space: pre-wrap;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }
        .metric-label {
            color: #666;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AI Text-to-JSON Extraction System</h1>
        
        <form id="extractionForm">
            <div class="form-group">
                <label for="schema">JSON Schema:</label>
                <textarea id="schema" name="schema" placeholder="Paste your JSON schema here..." required></textarea>
            </div>
            
            <div class="form-group">
                <label for="text">Text to Extract From:</label>
                <textarea id="text" name="text" placeholder="Paste the text you want to extract data from..." required></textarea>
            </div>
            
            <div class="form-group">
                <label for="schemaFile">Or upload schema file:</label>
                <input type="file" id="schemaFile" accept=".json">
            </div>
            
            <div class="form-group">
                <label for="textFile">Or upload text file:</label>
                <input type="file" id="textFile" accept=".txt,.md">
            </div>
            
            <button type="submit" id="extractBtn">Extract Data</button>
            <button type="button" id="clearBtn">Clear All</button>
        </form>
        
        <div id="results" class="results" style="display: none;">
            <div class="tabs">
                <div class="tab active" onclick="showTab('extracted')">Extracted Data</div>
                <div class="tab" onclick="showTab('confidence')">Confidence Analysis</div>
                <div class="tab" onclick="showTab('metrics')">Schema Metrics</div>
                <div class="tab" onclick="showTab('raw')">Raw Response</div>
            </div>
            
            <div id="extracted" class="tab-content active">
                <h3>Extracted Data <span id="overallConfidence"></span></h3>
                <pre id="extractedData"></pre>
            </div>
            
            <div id="confidence" class="tab-content">
                <h3>Confidence Analysis</h3>
                <div id="confidenceDetails"></div>
                <div id="reviewCandidates"></div>
            </div>
            
            <div id="metrics" class="tab-content">
                <h3>Schema Analysis</h3>
                <div id="schemaMetrics" class="metrics"></div>
            </div>
            
            <div id="raw" class="tab-content">
                <h3>Raw API Response</h3>
                <pre id="rawResponse"></pre>
            </div>
        </div>
        
        <div id="loading" class="loading" style="display: none;">
            Processing your request... This may take a moment for complex schemas.
        </div>
        
        <div id="error" class="error" style="display: none;"></div>
    </div>

    <script>
        // File upload handlers
        document.getElementById('schemaFile').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('schema').value = e.target.result;
                };
                reader.readAsText(file);
            }
        });

        document.getElementById('textFile').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('text').value = e.target.result;
                };
                reader.readAsText(file);
            }
        });

        // Form submission
        document.getElementById('extractionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const schema = document.getElementById('schema').value;
            const text = document.getElementById('text').value;
            
            if (!schema || !text) {
                showError('Please provide both schema and text.');
                return;
            }
            
            // Validate JSON schema
            try {
                JSON.parse(schema);
            } catch (e) {
                showError('Invalid JSON schema: ' + e.message);
                return;
            }
            
            showLoading(true);
            hideError();
            hideResults();
            
            try {
                const response = await fetch('/extract', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        schema: JSON.parse(schema),
                        text: text
                    })
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    displayResults(result);
                } else {
                    showError(result.error || 'An error occurred during extraction.');
                }
            } catch (error) {
                showError('Network error: ' + error.message);
            } finally {
                showLoading(false);
            }
        });

        // Clear button
        document.getElementById('clearBtn').addEventListener('click', function() {
            document.getElementById('schema').value = '';
            document.getElementById('text').value = '';
            document.getElementById('schemaFile').value = '';
            document.getElementById('textFile').value = '';
            hideResults();
            hideError();
        });

        function showLoading(show) {
            document.getElementById('loading').style.display = show ? 'block' : 'none';
            document.getElementById('extractBtn').disabled = show;
        }

        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
        }

        function hideError() {
            document.getElementById('error').style.display = 'none';
        }

        function hideResults() {
            document.getElementById('results').style.display = 'none';
        }

        function displayResults(result) {
            // Show results container
            document.getElementById('results').style.display = 'block';
            
            // Display extracted data
            document.getElementById('extractedData').textContent = JSON.stringify(result.data, null, 2);
            
            // Display overall confidence
            const confidence = result.confidence.overall;
            const confidenceClass = confidence >= 0.8 ? 'confidence-high' : 
                                  confidence >= 0.6 ? 'confidence-medium' : 'confidence-low';
            document.getElementById('overallConfidence').innerHTML = 
                `<span class="confidence-score ${confidenceClass}">${(confidence * 100).toFixed(1)}%</span>`;
            
            // Display confidence details
            displayConfidenceDetails(result.confidence);
            
            // Display schema metrics
            displaySchemaMetrics(result.schema_analysis);
            
            // Display raw response
            document.getElementById('rawResponse').textContent = JSON.stringify(result, null, 2);
        }

        function displayConfidenceDetails(confidence) {
            let html = '<h4>Field Confidence Scores:</h4>';
            
            for (const [field, score] of Object.entries(confidence.fields)) {
                const confidenceClass = score >= 0.8 ? 'confidence-high' : 
                                      score >= 0.6 ? 'confidence-medium' : 'confidence-low';
                html += `<div><strong>${field}:</strong> <span class="confidence-score ${confidenceClass}">${(score * 100).toFixed(1)}%</span></div>`;
            }
            
            if (confidence.completion) {
                html += '<h4>Completion Metrics:</h4>';
                html += `<div>Fields Completed: ${confidence.completion.completed_fields}/${confidence.completion.total_fields} (${(confidence.completion.completion_rate * 100).toFixed(1)}%)</div>`;
                html += `<div>Required Fields: ${confidence.completion.completed_required}/${confidence.completion.required_fields} (${(confidence.completion.required_completion_rate * 100).toFixed(1)}%)</div>`;
            }
            
            document.getElementById('confidenceDetails').innerHTML = html;
            
            // Display review candidates
            if (confidence.review_candidates && confidence.review_candidates.length > 0) {
                let reviewHtml = '<div class="review-candidates"><h4>⚠️ Fields Needing Review:</h4>';
                confidence.review_candidates.forEach(candidate => {
                    reviewHtml += `<div><strong>${candidate.field}</strong> (${(candidate.confidence * 100).toFixed(1)}%) - ${candidate.reason}</div>`;
                });
                reviewHtml += '</div>';
                document.getElementById('reviewCandidates').innerHTML = reviewHtml;
            } else {
                document.getElementById('reviewCandidates').innerHTML = '<div style="color: green;">✅ All fields meet confidence threshold</div>';
            }
        }

        function displaySchemaMetrics(analysis) {
            if (!analysis) return;
            
            const metrics = analysis.metrics;
            let html = '';
            
            html += `<div class="metric-card"><div class="metric-value">${metrics.max_depth}</div><div class="metric-label">Max Depth</div></div>`;
            html += `<div class="metric-card"><div class="metric-value">${metrics.total_fields}</div><div class="metric-label">Total Fields</div></div>`;
            html += `<div class="metric-card"><div class="metric-value">${metrics.object_count}</div><div class="metric-label">Objects</div></div>`;
            html += `<div class="metric-card"><div class="metric-value">${metrics.enum_complexity}</div><div class="metric-label">Enum Values</div></div>`;
            html += `<div class="metric-card"><div class="metric-value">${analysis.complexity_score}</div><div class="metric-label">Complexity Score</div></div>`;
            html += `<div class="metric-card"><div class="metric-value">${analysis.strategy}</div><div class="metric-label">Strategy Used</div></div>`;
            
            document.getElementById('schemaMetrics').innerHTML = html;
        }

        function showTab(tabName) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the main web interface."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/extract', methods=['POST'])
def extract_endpoint():
    """Main extraction endpoint."""
    try:
        data = request.get_json()
        
        if not data or 'schema' not in data or 'text' not in data:
            return jsonify({'error': 'Missing required fields: schema and text'}), 400
        
        schema = data['schema']
        text = data['text']
        
        # Validate schema
        if not isinstance(schema, dict):
            return jsonify({'error': 'Schema must be a valid JSON object'}), 400
        
        # Analyze schema complexity
        schema_analysis = schema_analyzer.analyze_complexity(schema)
        
        # Process document if needed
        doc_info = document_processor.process_document(text, schema)
        
        if doc_info['needs_chunking']:
            # Handle large documents
            chunk_results = []
            for chunk in doc_info['chunks']:
                chunk_result = extraction_engine.extract(chunk['text'], schema)
                chunk_results.append(chunk_result)
            
            # Merge results from all chunks
            final_result = document_processor.merge_extractions(chunk_results)
        else:
            # Single extraction
            final_result = extraction_engine.extract(text, schema)
        
        # Enhanced confidence scoring
        enhanced_confidence = confidence_scorer.score_extraction(
            final_result['data'], schema, text
        )
        
        # Combine all results
        response = {
            'data': final_result['data'],
            'confidence': enhanced_confidence,
            'strategy': final_result['strategy'],
            'token_usage': final_result.get('token_usage', 0),
            'schema_analysis': schema_analysis,
            'document_info': doc_info,
            'processing_time': None  # Could add timing if needed
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-schema', methods=['POST'])
def analyze_schema_endpoint():
    """Analyze schema complexity without extraction."""
    try:
        data = request.get_json()
        
        if not data or 'schema' not in data:
            return jsonify({'error': 'Missing required field: schema'}), 400
        
        schema = data['schema']
        
        if not isinstance(schema, dict):
            return jsonify({'error': 'Schema must be a valid JSON object'}), 400
        
        analysis = schema_analyzer.analyze_complexity(schema)
        
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'components': {
            'extraction_engine': 'ready',
            'document_processor': 'ready',
            'confidence_scorer': 'ready',
            'schema_analyzer': 'ready'
        }
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Check for required environment variables
    if not os.getenv('OPENAI_API_KEY'):
        print("Warning: OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in the .env file.")
    
    # Run the application
    debug_mode = os.getenv('FLASK_ENV') == 'development'
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)
