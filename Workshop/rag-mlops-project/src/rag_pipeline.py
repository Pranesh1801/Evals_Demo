from src.vector_store import VectorStore
from src.llm import LLMModel
from src.embeddings import EmbeddingModel
from src.document_loader import DocumentLoader
from src.config import Config
import mlflow
import time

class RAGPipeline:
    def __init__(self):
        self.config = Config()
        self.embedding_model = EmbeddingModel(self.config.EMBEDDING_MODEL)
        self.vector_store = VectorStore(self.config.CHROMA_PERSIST_DIR, self.embedding_model)
        self.llm = LLMModel(self.config.OPENAI_API_KEY, self.config.MODEL_NAME)
        self.document_loader = DocumentLoader(self.config.CHUNK_SIZE, self.config.CHUNK_OVERLAP)
    
    def ingest_documents(self, file_paths: list):
        with mlflow.start_run(run_name="document_ingestion"):
            all_chunks = []
            for path in file_paths:
                if path.endswith('.pdf'):
                    chunks = self.document_loader.load_pdf(path)
                elif path.endswith('.docx'):
                    chunks = self.document_loader.load_docx(path)
                else:
                    chunks = self.document_loader.load_txt(path)
                all_chunks.extend(chunks)
            
            self.vector_store.add_documents(all_chunks)
            mlflow.log_metric("total_chunks", len(all_chunks))
    
    def query(self, question: str) -> dict:
        with mlflow.start_run(run_name="rag_query"):
            start_time = time.time()
            
            # Retrieval
            context = self.vector_store.search(question, self.config.TOP_K)
            mlflow.log_metric("retrieval_time", time.time() - start_time)
            
            # Generation
            gen_start = time.time()
            answer = self.llm.generate_response(question, context)
            mlflow.log_metric("generation_time", time.time() - gen_start)
            mlflow.log_metric("total_latency", time.time() - start_time)
            
            return {
                "question": question,
                "answer": answer,
                "sources": context
            }
