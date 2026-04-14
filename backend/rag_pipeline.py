"""
RAG (Retrieval-Augmented Generation) pipeline for sustainability reports.

This module splits documents into chunks, generates embeddings, and stores them in ChromaDB.
"""

import os
import logging
from typing import Optional, List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import config
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG pipeline for document retrieval and context generation."""

    def __init__(self, vectorstore_dir: str = "./vectorstore"):
        """
        Initialize RAG pipeline.

        Args:
            vectorstore_dir: Directory to store vector database
        """
        self.logger = logger
        self.vectorstore_dir = vectorstore_dir
        self.vectorstore = None
        self.embeddings = None

        # Create vectorstore directory if it doesn't exist
        os.makedirs(vectorstore_dir, exist_ok=True)

    def initialize_embeddings(self) -> bool:
        """
        Initialize embeddings based on config (OpenAI or free HuggingFace).
        """
        try:
            cfg = config.config
            if cfg.OPENAI_API_KEY:
                self.embeddings = OpenAIEmbeddings(api_key=cfg.OPENAI_API_KEY)
                self.logger.info("OpenAI embeddings initialized")
            else:
                self.embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
                self.logger.info("Free HuggingFace embeddings initialized (all-MiniLM-L6-v2)")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing embeddings: {str(e)}")
            return False

    def split_text_into_chunks(
        self,
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[str]:
        """
        Split text into chunks for embedding.

        Args:
            text: Text to split
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""],
            )
            chunks = splitter.split_text(text)
            self.logger.info(f"Split text into {len(chunks)} chunks")
            return chunks

        except Exception as e:
            self.logger.error(f"Error splitting text: {str(e)}")
            return []

    def create_vectorstore(self, documents: List[str], collection_name: str = "sustainability") -> bool:
        """
        Create vector store from documents, replacing any existing collection.
        """
        try:
            if not self.embeddings:
                self.initialize_embeddings()

            # Delete existing collection to prevent stale data from prior uploads
            try:
                import chromadb
                client = chromadb.PersistentClient(path=self.vectorstore_dir)
                client.delete_collection(collection_name)
                self.logger.info(f"Deleted existing collection: {collection_name}")
            except Exception:
                pass  # Collection may not exist yet

            self.vectorstore = Chroma.from_texts(
                texts=documents,
                embedding=self.embeddings,
                collection_name=collection_name,
                persist_directory=self.vectorstore_dir,
            )
            self.logger.info(f"Vector store created with {len(documents)} documents")
            return True

        except Exception as e:
            self.logger.error(f"Error creating vector store: {str(e)}")
            return False

    def load_vectorstore(self, collection_name: str = "sustainability") -> bool:
        """
        Load existing vector store.

        Args:
            collection_name: Name of the collection

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.embeddings:
                self.initialize_embeddings()

            self.vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.vectorstore_dir,
            )
            self.logger.info("Vector store loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error loading vector store: {str(e)}")
            return None

    def search(self, query: str, k: int = 5) -> Optional[List[str]]:
        """
        Search vector store for relevant documents.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of relevant documents or None
        """
        try:
            if not self.vectorstore:
                self.logger.error("Vector store not initialized")
                return None

            results = self.vectorstore.similarity_search(query, k=k)
            documents = [result.page_content for result in results]

            self.logger.info(f"Found {len(documents)} relevant documents")
            return documents

        except Exception as e:
            self.logger.error(f"Error searching vector store: {str(e)}")
            return None

    def process_pdf_to_vectorstore(
        self,
        text: str,
        collection_name: str = "sustainability",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> bool:
        """
        Process PDF text and store in vector database.

        Args:
            text: Extracted text from PDF
            collection_name: Name of the collection
            chunk_size: Size of chunks
            chunk_overlap: Overlap between chunks

        Returns:
            True if successful, False otherwise
        """
        try:
            # Split text into chunks
            chunks = self.split_text_into_chunks(text, chunk_size, chunk_overlap)

            if not chunks:
                self.logger.error("No chunks created from text")
                return False

            # Create vector store
            return self.create_vectorstore(chunks, collection_name)

        except Exception as e:
            self.logger.error(f"Error processing PDF: {str(e)}")
            return False
