import chromadb
from typing import List, Dict
from src.embeddings import EmbeddingModel
import mlflow

class VectorStore:
    def __init__(self, persist_dir: str, embedding_model: EmbeddingModel):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_model = embedding_model
        self.collection = self.client.get_or_create_collection("documents")
        mlflow.log_param("vector_db", "chromadb")
    
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None):
        embeddings = self.embedding_model.embed_documents(texts)
        ids = [f"doc_{i}" for i in range(len(texts))]
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas or [{}] * len(texts),
            ids=ids
        )
        mlflow.log_metric("documents_indexed", len(texts))
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        query_embedding = self.embedding_model.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return [
            {"text": doc, "metadata": meta}
            for doc, meta in zip(results['documents'][0], results['metadatas'][0])
        ]
