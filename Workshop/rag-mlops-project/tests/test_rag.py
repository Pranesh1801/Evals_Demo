import pytest
from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingModel

def test_document_loader():
    loader = DocumentLoader(chunk_size=100, chunk_overlap=20)
    text = "This is a test. " * 50
    chunks = loader.splitter.split_text(text)
    assert len(chunks) > 0
    assert all(len(chunk) <= 120 for chunk in chunks)

def test_embedding_model():
    model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
    texts = ["Hello world", "Test document"]
    embeddings = model.embed_documents(texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384

def test_query_embedding():
    model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
    embedding = model.embed_query("Test query")
    assert len(embedding) == 384
