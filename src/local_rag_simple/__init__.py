"""
local-rag-simple: A minimal Retrieval-Augmented Generation (RAG) pipeline.

This package provides a simple RAG pipeline for querying PDF documents
using embeddings and local LLMs without external API calls.
"""

__version__ = "0.1.0"
__author__ = "Teja Nallagatla"
__email__ = "tejaswini.15n@gmail.com"
__license__ = "MIT"

# Import RAGPipeline so users can do:
# from local_rag_simple import RAGPipeline
from .main import RAGPipeline

# Define public API
__all__ = [
    "RAGPipeline",
    "__version__",
    "__author__",
]