from typing import List
from pypdf import PdfReader
import docx

class DocumentLoader:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def _split_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks
    
    def load_pdf(self, file_path: str) -> List[str]:
        reader = PdfReader(file_path)
        text = "".join([page.extract_text() for page in reader.pages])
        return self._split_text(text)
    
    def load_docx(self, file_path: str) -> List[str]:
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return self._split_text(text)
    
    def load_txt(self, file_path: str) -> List[str]:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return self._split_text(text)
