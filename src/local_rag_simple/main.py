"""
Main RAG Pipeline Module

This module orchestrates the complete Retrieval-Augmented Generation (RAG) pipeline.
It combines all 5 components to answer user questions based on a PDF knowledge base.

Components:
    1. Knowledge Base (PDF loading & chunking)
    2. Semantic Layer (embeddings)
    3. Retrieval System (FAISS indexing & search)
    4. Augmentation (context enrichment)
    5. Generation (LLM answer synthesis)
"""

import warnings
from typing import List, Dict, Optional

# Suppress transformers warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

from .knowledge_base import KnowledgeBase
from .semantic_layer import SemanticLayer
from .retrieval_system import RetrievalSystem
from .augmentation import Augmentation
from .generation import Generation


class RAGPipeline:
    """
    Complete Retrieval-Augmented Generation Pipeline.
    
    Orchestrates all 5 components to answer questions using a PDF knowledge base.
    
    Example:
        >>> pipeline = RAGPipeline(pdf_path="data/document.pdf")
        >>> answer = pipeline.ask_question("What is the main topic?")
        >>> print(answer)
    """
    
    def __init__(
        self,
        pdf_path: str,
        chunk_size: int = 200,
        chunk_overlap: int = 50,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama3.2:3b",
        use_llm: bool = True,
    ):
        """
        Initialize the RAG Pipeline.
        
        Args:
            pdf_path: Path to the PDF file
            chunk_size: Size of text chunks (in characters)
            chunk_overlap: Overlap between chunks (in characters)
            embedding_model: HuggingFace model for embeddings
            llm_model: LLM model to use for generation
            use_llm: Whether to use LLM for generation
        """
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.use_llm = use_llm
        
        # Will be initialized in setup()
        self.kb = None
        self.semantic = None
        self.retrieval = None
        self.augmentor = None
        self.generator = None
        self.document_embeddings = None
        
    def setup(self) -> None:
        """
        Initialize all pipeline components.
        
        This must be called before asking questions.
        """
        print(f"🚀 Initializing RAG Pipeline...")
        print(f"   PDF: {self.pdf_path}")
        print(f"   Chunk size: {self.chunk_size}")
        print(f"   Chunk overlap: {self.chunk_overlap}")
        print()
        
        # COMPONENT 1: Knowledge Base
        print("📚 [Component 1] Initializing Knowledge Base...")
        self.kb = KnowledgeBase(
            pdf_path=self.pdf_path,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Load PDF and create chunks
        documents = self.kb.load_pdf_data()
        chunks = self.kb.create_chunks()
        
        # Get statistics
        stats = self.kb.get_stats()
        print(f"   ✅ Loaded {stats['total_pages']} pages")
        print(f"   ✅ Created {stats['total_chunks']} chunks")
        print()
        
        # COMPONENT 2: Semantic Layer
        print("🧠 [Component 2] Initializing Semantic Layer...")
        self.semantic = SemanticLayer()
        self.document_embeddings = self.semantic.encode_documents(self.kb.chunks)
        print(f"   ✅ Embeddings created: {self.document_embeddings.shape}")
        print()
        
        # COMPONENT 3: Retrieval System
        print("🔍 [Component 3] Initializing Retrieval System...")
        self.retrieval = RetrievalSystem(dimension=self.document_embeddings.shape[1])
        self.retrieval.build_index(self.document_embeddings, self.kb.chunks)
        print(f"   ✅ Index built with {len(self.kb.chunks)} vectors")
        print()
        
        # COMPONENT 4: Augmentation
        print("⚡ [Component 4] Initializing Augmentation...")
        self.augmentor = Augmentation()
        print(f"   ✅ Augmentor ready")
        print()
        
        # COMPONENT 5: Generation
        print("🤖 [Component 5] Initializing Generation...")
        self.generator = Generation(model_name=self.llm_model, use_llm=self.use_llm)
        print(f"   ✅ Generator ready ({self.llm_model})")
        print()
        
        print("="*70)
        print("✅ RAG Pipeline Ready!")
        print("="*70)
        print()
    
    def ask_question(self, question: str, top_k: int = 3) -> str:
        """
        Ask a question and get an answer from the RAG pipeline.
        
        Args:
            question: The question to ask
            top_k: Number of top documents to retrieve
            
        Returns:
            The generated answer
        """
        if self.kb is None:
            raise RuntimeError("Pipeline not initialized! Call setup() first.")
        
        print("="*70)
        print(f"❓ QUESTION: {question}")
        print("="*70)
        print()
        
        # STEP 1: Encode query
        print("[Step 1] Encoding question...")
        query_embedding = self.semantic.encode_query(question)
        
        # STEP 2: Retrieve relevant chunks
        print("[Step 2] Searching knowledge base...")
        results = self.retrieval.search(query_embedding, top_k=top_k)
        
        # Display retrieved documents
        print(f"\n📚 Retrieved Top {top_k} Chunks:")
        for r in results:
            doc_page = r['document']['page']
            doc_text = r['document']['chunk']
            similarity_pct = r['similarity'] * 100
            print(f"  • Page {doc_page} (similarity: {similarity_pct:.1f}%)")
            print(f"    Text snippet: {doc_text[:200]}...\n")
        
        # STEP 3: Augment with context
        print("[Step 3] Creating enriched context...")
        enriched = self.augmentor.create_context(question, results)
        
        # STEP 4: Generate answer
        print("[Step 4] Generating answer...")
        answer = self.generator.generate(enriched, results)
        
        # Display results
        print("\n" + "="*70)
        print("💬 ANSWER:")
        print("="*70)
        print(answer)
        print("="*70)
        print()
        
        return answer
    
    def interactive_mode(self) -> None:
        """
        Enter interactive mode for continuous questioning.
        
        Users can ask multiple questions until they type 'quit'.
        """
        print("\n🎯 Entering Interactive Mode")
        print("Type 'quit' or 'exit' to stop\n")
        
        while True:
            user_question = input("Your question (or 'quit' to stop): ").strip()
            
            if user_question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using RAG Pipeline!")
                break
            
            if user_question:
                self.ask_question(user_question)
            else:
                print("Please enter a question!\n")


def main():
    """
    Main entry point for the RAG pipeline.
    
    Run this to start the RAG system with an interactive interface.
    """
    import sys
    import os
    
    # Determine PDF path
    # First check if path is provided as argument
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Default to data/hpe-pcai.pdf
        pdf_path = "data/hpe-pcai.pdf"
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file not found: {pdf_path}")
        print(f"\nUsage: python -m local_rag_simple.main [path_to_pdf]")
        print(f"Example: python -m local_rag_simple.main data/hpe-pcai.pdf")
        sys.exit(1)
    
    # Create and initialize pipeline
    pipeline = RAGPipeline(
        pdf_path=pdf_path,
        chunk_size=200,
        chunk_overlap=50,
        llm_model="llama3.2:3b",
        use_llm=True,
    )
    
    # Setup all components
    pipeline.setup()
    
    # Enter interactive mode
    pipeline.interactive_mode()


if __name__ == "__main__":
    main()