from pathlib import Path
from PyPDF2 import PdfReader
#import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

class KnowledgeBase:
    """
    Manages PDF documents and creates sentence-preserving chunks for embeddings.
    """

    def __init__(self, pdf_path: str = "data/hpe-pcai.pdf", chunk_size: int = 200, chunk_overlap: int = 50):
        """
        pdf_path: Path to PDF file
        chunk_size: Approximate number of words per chunk
        chunk_overlap: Number of overlapping words between consecutive chunks
        """
        self.pdf_path = Path(pdf_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = []
        self.chunks = []
        print("📚 Knowledge Base initialized")

    def load_pdf_data(self):
        """Load text content from the PDF"""
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"❌ PDF file not found: {self.pdf_path}")

        print(f"\n📥 Loading PDF: {self.pdf_path}")
        reader = PdfReader(str(self.pdf_path))
        self.documents = []

        for i, page in enumerate(reader.pages):
            content = page.extract_text() or ""
            if content.strip():
                self.documents.append({"page": i + 1, "content": content.strip()})

        print(f"✅ Loaded {len(self.documents)} pages from PDF")
        return self.documents

    def create_chunks(self):
        """Convert all pages into sentence-preserving chunks"""
        if not self.documents:
            raise ValueError("No documents loaded. Call load_pdf_data() first.")
        
        print("\n Creating chunks with LangChain RecursiveCharacterTextSplitter...")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )
        #Init storage for self.chunks
        self.chunks = []
        # loop through each pdf page
        for d in self.documents:
            split = splitter.split_text(d["content"])
            for chunk in split:
                self.chunks.append({
                    "page": d["page"],
                    "chunk": chunk
                })

        print(f"✅ Created {len(self.chunks)} chunks")
        return self.chunks

    def get_stats(self):
        return {
            "total_pages": len(self.documents),
            "total_chunks": len(self.chunks),
            "source": str(self.pdf_path)
        }
