from pathlib import Path
from PyPDF2 import PdfReader
import re

def sent_tokenize(text):
    """
    Simple regex-based sentence tokenizer.
    Splits text at '.', '!', '?' followed by a space or end of line.
    """
    # Remove multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    # Split sentences using regex
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Remove empty strings
    return [s for s in sentences if s]

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

    def chunk_text(self, text: str):
        """
        Split text into sentence-preserving chunks with overlap
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sent in sentences:
            words = sent.split()
            sent_length = len(words)

            # If adding this sentence exceeds chunk size, finalize current chunk
            if current_length + sent_length > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Keep overlap words
                overlap_words = current_chunk[-self.chunk_overlap:] if self.chunk_overlap < len(current_chunk) else current_chunk
                current_chunk = overlap_words.copy()
                current_length = len(current_chunk)

            # Add current sentence
            current_chunk.extend(words)
            current_length += sent_length

        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def create_chunks(self):
        """Convert all pages into sentence-preserving chunks"""
        if not self.documents:
            raise ValueError("No documents loaded. Call load_pdf_data() first.")

        self.chunks = []
        for doc in self.documents:
            page_chunks = self.chunk_text(doc['content'])
            for c in page_chunks:
                self.chunks.append({"page": doc['page'], "chunk": c})

        print(f"✅ Created {len(self.chunks)} text chunks")
        return self.chunks

    def get_stats(self):
        return {
            "total_pages": len(self.documents),
            "total_chunks": len(self.chunks),
            "source": str(self.pdf_path)
        }
