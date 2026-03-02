from src.rag_pipeline import RAGPipeline
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.live import Live
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

console = Console()

def visualize_rag(query: str, top_k: int = 3):
    pipeline = RAGPipeline()
    
    console.print(Panel.fit(
        f"[bold cyan]Query:[/bold cyan] {query}",
        title="RAG Pipeline Visualizer",
        border_style="cyan"
    ))
    
    # Step 1: Embedding
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[yellow]Embedding query...", total=None)
        start = time.time()
        query_emb = pipeline.embedding_model.embed_query(query)
        emb_time = time.time() - start
        progress.update(task, completed=True)
    
    console.print(f"✅ Query embedded: {len(query_emb)}D vector in {emb_time:.3f}s\n")
    
    # Step 2: Search
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[yellow]Searching vector database...", total=None)
        start = time.time()
        results = pipeline.vector_store.search(query, top_k)
        search_time = time.time() - start
        progress.update(task, completed=True)
    
    console.print(f"✅ Retrieved {len(results)} documents in {search_time:.3f}s\n")
    
    # Similarity scores
    doc_texts = [r['text'] for r in results]
    doc_embs = pipeline.embedding_model.embed_documents(doc_texts)
    all_embs = np.array([query_emb] + doc_embs)
    sim_matrix = cosine_similarity(all_embs)
    
    table = Table(title="Similarity Scores", show_header=True, header_style="bold magenta")
    table.add_column("Doc", style="cyan", width=8)
    table.add_column("Similarity", justify="right", style="green")
    table.add_column("Preview", style="white", width=60)
    
    for i, (doc, score) in enumerate(zip(results, sim_matrix[0, 1:]), 1):
        preview = doc['text'][:60] + "..." if len(doc['text']) > 60 else doc['text']
        table.add_row(f"Doc {i}", f"{score:.4f}", preview)
    
    console.print(table)
    console.print()
    
    # Step 3: Generation
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[yellow]Generating response...", total=None)
        start = time.time()
        answer = pipeline.llm.generate_response(query, results)
        gen_time = time.time() - start
        progress.update(task, completed=True)
    
    console.print(f"✅ Response generated in {gen_time:.2f}s\n")
    
    # Final answer
    console.print(Panel(
        f"[bold green]{answer}[/bold green]",
        title="Generated Answer",
        border_style="green"
    ))
    
    # Metrics
    console.print("\n[bold]Performance Metrics:[/bold]")
    metrics_table = Table(show_header=False, box=None)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="yellow")
    
    metrics_table.add_row("Embedding Time", f"{emb_time*1000:.1f}ms")
    metrics_table.add_row("Search Time", f"{search_time*1000:.1f}ms")
    metrics_table.add_row("Generation Time", f"{gen_time:.2f}s")
    metrics_table.add_row("Total Latency", f"{emb_time+search_time+gen_time:.2f}s")
    metrics_table.add_row("Avg Similarity", f"{np.mean(sim_matrix[0, 1:]):.4f}")
    
    console.print(metrics_table)

if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "What is RAG?"
    visualize_rag(query)
