from openai import OpenAI
from typing import List, Dict
import mlflow

class LLMModel:
    def __init__(self, api_key: str, model_name: str):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        mlflow.log_param("llm_model", model_name)
    
    def generate_response(self, query: str, context: List[Dict]) -> str:
        context_text = "\n\n".join([doc["text"] for doc in context])
        
        prompt = f"""Answer the question based on the context below.
        
Context:
{context_text}

Question: {query}

Answer:"""
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        answer = response.choices[0].message.content
        mlflow.log_metric("tokens_used", response.usage.total_tokens)
        return answer
