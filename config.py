"""
Configuration and initialization module for AquaMetric AI.

This module loads environment variables and provides configuration
for the entire application.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration."""

    # Flask
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    ENV = os.getenv('FLASK_ENV', 'production')

    # OpenAI
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    
    # Groq/Grok
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')
    
    # LLM Provider: 'openai' or 'groq' (groq if no openai key)
    @property
    def LLM_PROVIDER(self):
        return 'groq' if self.GROQ_API_KEY and not self.OPENAI_API_KEY else 'openai'
    
    @property
    def LLM_MODEL(self):
        return self.GROQ_MODEL if self.LLM_PROVIDER == 'groq' else self.OPENAI_MODEL

    # Application
    MAX_PDF_SIZE = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = os.getenv('UPLOAD_DIR', './uploads')
    VECTORSTORE_DIR = os.getenv('VECTORSTORE_DIR', './vectorstore')
    DATA_DIR = os.getenv('DATA_DIR', './data')

    # RAG Pipeline
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 200))

    # Validation
    ALLOWED_EXTENSIONS = {'pdf'}

    def validate(self):
        """Validate configuration."""
        if self.LLM_PROVIDER == 'openai' and not self.OPENAI_API_KEY:
            raise ValueError('OPENAI_API_KEY required for OpenAI')
        if self.LLM_PROVIDER == 'groq' and not self.GROQ_API_KEY:
            raise ValueError('GROQ_API_KEY required for Groq')
        
        # Create necessary directories
        os.makedirs(self.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(self.VECTORSTORE_DIR, exist_ok=True)
        os.makedirs(self.DATA_DIR, exist_ok=True)


# Create config instance
config = Config()
