"""
COMPONENT 3: RETRIEVAL SYSTEM
Requirement: Identify and retrieve most relevant documents
Tool: FAISS (Facebook AI Similarity Search)
"""

import faiss
import numpy as np

class RetrievalSystem:
    """
    Fast similarity search using FAISS index.
    Retrieves top-k most relevant documents for a query.
    """
    
    def __init__(self, dimension=384):
        self.dimension = dimension
        self.index = None
        self.documents = []
        print(f"🔍 COMPONENT 3: Retrieval System initialized (dim={dimension})")
    
    def build_index(self, embeddings, documents):
        """
        Build FAISS index for fast similarity search.
        
        FAISS = Facebook AI Similarity Search
        - Industry standard for vector search
        - Fast: can search millions of vectors in milliseconds
        
        Args:
            embeddings: numpy array (num_docs, 384)
            documents: List of document dictionaries
        """
        print(f"\n🔨 Building FAISS index...")
        print(f"   Documents: {len(documents)}")
        print(f"   Embedding shape: {embeddings.shape}")
        
        self.documents = documents
        
        # Create flat L2 index (exact search)
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Add embeddings to index
        embeddings_array = np.array(embeddings).astype('float32')
        self.index.add(embeddings_array)
        
        print(f"✅ Index built with {self.index.ntotal} vectors")
    
    def search(self, query_embedding, top_k=3):
        """
        Search for most similar documents.
        
        Args:
            query_embedding: Query vector (1, 384)
            top_k: Number of results to return
        
        Returns:
            List of dicts with document, similarity score, rank
        """
        print(f"\n🔎 Searching for top {top_k} relevant documents...")
        
        # Ensure correct format
        query_array = np.array(query_embedding).astype('float32')
        if query_array.ndim == 1:
            query_array = query_array.reshape(1, -1)
        
        # Search index
        distances, indices = self.index.search(query_array, top_k)
        
        # Format results
        results = []
        for rank, (idx, distance) in enumerate(zip(indices[0], distances[0]), 1):
            similarity = 1 / (1 + float(distance))
            
            results.append({
                'rank': rank,
                'document': self.documents[idx],
                'similarity': similarity,
                'distance': float(distance)
            })
        
        print(f"✅ Retrieved {len(results)} documents")
        return results