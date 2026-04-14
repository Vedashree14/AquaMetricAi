"""
Flask web application for AquaMetric AI.
"""

import os
import logging
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# Silence noisy ChromaDB telemetry error (posthog API mismatch in this version)
logging.getLogger('chromadb.telemetry.product.posthog').setLevel(logging.CRITICAL)
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from backend.pdf_extractor import PDFExtractor
from backend.table_extractor import TableExtractor
from backend.data_processor import DataProcessor
from backend.rag_pipeline import RAGPipeline
from backend.agent import WaterSustainabilityAgent

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
ALLOWED_EXTENSIONS = {'pdf'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

pdf_extractor = PDFExtractor()
table_extractor = TableExtractor()
data_processor = DataProcessor()
rag_pipeline = RAGPipeline(vectorstore_dir="./vectorstore")

agent = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_agent():
    global agent
    if agent is None:
        try:
            agent = WaterSustainabilityAgent()
            logger.info("Agent initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing agent: {str(e)}")
            raise
    return agent


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Only PDF files are allowed'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"File saved: {filepath}")

        try:
            # Extract text
            logger.info("Extracting PDF text...")
            pdf_text = pdf_extractor.extract_text(filepath)
            if not pdf_text:
                return jsonify({'error': 'Failed to extract text from PDF'}), 400

            # Extract tables
            logger.info("Extracting tables from PDF...")
            tables = table_extractor.extract_tables(filepath)
            tables_context = ""
            if tables:
                for i, table in enumerate(tables):
                    tables_context += f"\n\nTable {i + 1}:\n{table.to_string()}"

            full_text = pdf_text + tables_context

            # Build vector store
            logger.info("Processing document into vector store...")
            rag_pipeline.process_pdf_to_vectorstore(full_text, chunk_size=1500, chunk_overlap=300)

            # Targeted RAG queries — cast a wide net for water data tables
            logger.info("Searching for relevant context...")
            search_queries = [
                # Gross withdrawal — the primary target
                "total water withdrawal megalitres ML annual gross",
                "water withdrawal by source surface groundwater third-party",
                # WUE
                "water usage effectiveness WUE liters per kilowatt hour L/kWh",
                # Consumption / net (secondary — LLM will deprioritise these)
                "water consumption net water returned replenishment",
                # Goals and context
                "water positive water scarcity stress restoration 2030",
                # Data center specific
                "data center cooling water usage operations facilities",
                # Tables with numbers
                "billion liters megalitres water data environmental metrics table",
            ]

            seen = set()
            all_docs = []
            for query in search_queries:
                docs = rag_pipeline.search(query, k=6)
                if docs:
                    for d in docs:
                        if d not in seen:
                            seen.add(d)
                            all_docs.append(d)

            context = "\n\n---\n\n".join(all_docs) if all_docs else ""
            logger.info(f"RAG retrieved {len(all_docs)} unique chunks")
            logger.info(f"RAG context preview:\n{context[:800]}")

            water_scarcity_context = request.form.get('waterScarcityContext', '')

            logger.info("Running sustainability analysis...")
            agent_instance = get_agent()
            analysis_result = agent_instance.analyze_sustainability_report(
                full_text,
                water_scarcity_context=water_scarcity_context or context
            )

            if not analysis_result:
                logger.warning("Full analysis failed, attempting metric extraction...")
                analysis_result = agent_instance.extract_metrics_only(full_text)

                if analysis_result:
                    water_usage = analysis_result.get('water_usage', 'Unknown')
                    wue = analysis_result.get('WUE', 'Unknown')
                    region = analysis_result.get('region', 'Unknown')

                    recommendations = agent_instance.generate_recommendations(water_usage, wue, region)
                    analysis_result['recommendations'] = recommendations or []
                    analysis_result['risk_level'] = data_processor.estimate_risk_level(
                        float(water_usage) if isinstance(water_usage, str) and water_usage.replace('.', '', 1).isdigit() else 0,
                        region
                    )
                else:
                    return jsonify({'error': 'Failed to analyze PDF'}), 400

            logger.info("Analysis completed successfully")
            return jsonify({'success': True, 'data': analysis_result})

        finally:
            try:
                os.remove(filepath)
            except Exception as e:
                logger.warning(f"Could not delete uploaded file: {str(e)}")

    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File is too large. Maximum size is 50MB.'}), 413


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    logger.info("Starting AquaMetric AI Flask application...")
    logger.info("API Key configured: " + ("Yes" if os.getenv("OPENAI_API_KEY") else "No"))
    app.run(
        host='127.0.0.1',
        port=5001,
        debug=os.getenv('FLASK_ENV') == 'development'
    )
