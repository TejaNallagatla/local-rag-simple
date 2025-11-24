"""
COMPONENT 2: SEMANTIC LAYER
Requirement: Transform data and queries into semantic representations
Tool: sentence-transformers (all-MiniLM-L6-v2 model)
"""

from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticLayer:
    """
    Converts text to dense vector embeddings.
    Uses transformer model to capture semantic meaning.
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print("COMPONENT 2: Semantic Layer initializing...")
        print(f"   Loading model: {model_name}")
        print("   (First run downloads 22MB - takes 30 seconds)")
        
        # Load pre-trained model
        self.model = SentenceTransformer(model_name)
        self.dimension = 384  # MiniLM output dimension
        
        print(f"✅ Model loaded! Embeddings dimension: {self.dimension}")
    
    def encode_documents(self, documents):
        """
        Convert PDF chunks to embeddings.
        
        This transforms text into numerical vectors where:
        - Similar meanings = similar vectors
        - Each chunk becomes a 384-d vector
        
        Args:
            documents: List of dicts with 'chunk' or 'content' field
        
        Returns:
            numpy array of shape (num_chunks, 384)
        """
        print(f"\n🔄 Encoding {len(documents)} document chunks to embeddings...")
        
        # Extract text content
        texts = [doc.get('content', doc.get('chunk', '')) for doc in documents]

        
        # Convert to embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True  # Normalized for cosine similarity
        )
        
        print(f"✅ Embeddings created: {embeddings.shape}")
        return embeddings
    
    def encode_query(self, query):
        """
        Convert query to embedding.
        Uses same model so query and documents are in same semantic space.
        
        Args:
            query: Question string
        
        Returns:
            numpy array of shape (1, 384)
        """
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )
        return embedding