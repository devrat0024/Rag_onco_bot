import os, json, faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# CONFIG
# =========================================================
JSON_FOLDER = "jsons_4"
OUTPUT_DIR = "embeddings_biomed"
MODEL_NAME = "pritamdeka/S-BioBert-snli-multinli-stsb"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Choose your LLM backend
LLM_BACKEND = "groq"  # Options: "groq", "anthropic", "openai", "local"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Set your Groq API key
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Medical term expansions
MEDICAL_EXPANSIONS = {
    'cancer': ['carcinoma', 'tumor', 'tumour', 'neoplasm', 'malignancy'],
    'treatment': ['therapy', 'intervention', 'management', 'care'],
    'diagnosis': ['detection', 'screening', 'identification', 'examination'],
    'symptom': ['sign', 'manifestation', 'indicator'],
}

# =========================================================
# STEP 1: LOAD TEXTS FROM ALL JSON FILES
# =========================================================
def load_json_folder(json_folder: str) -> Tuple[List[str], List[Dict]]:
    """Load all JSON files and extract text chunks with metadata."""
    texts, meta = [], []
    files = [f for f in os.listdir(json_folder) if f.endswith(".json")]
    print(f"📂 Found {len(files)} JSON files in '{json_folder}'")

    for file in tqdm(files, desc="Loading JSON files"):
        path = os.path.join(json_folder, file)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️ Skipping {file} due to read error: {e}")
            continue

        if isinstance(data, dict) and "pages" in data:
            file_name = data.get("file_name", file)
            for page in data["pages"]:
                page_num = page.get("page_number", None)
                paragraphs = page.get("paragraphs", [])
                for pid, para in enumerate(paragraphs):
                    text = para.get("paragraph_text", "").strip()
                    if not text or len(text) < 20:
                        continue
                    texts.append(text)
                    meta.append({
                        "source": file_name,
                        "page_number": page_num,
                        "para_id": pid,
                        "preview": text[:100]
                    })
        else:
            print(f"⚠️ Unrecognized JSON structure in {file}")

    print(f"✅ Loaded {len(texts)} total text segments from {len(files)} files.")
    return texts, meta

# =========================================================
# STEP 2: BUILD INDEX
# =========================================================
def build_index(chunks: List[str]) -> Tuple[faiss.Index, np.ndarray]:
    """Build FAISS index from text chunks."""
    print(f"🔧 Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"✅ FAISS index built with {embeddings.shape[0]} vectors (dim={dim}).")
    return index, embeddings

def save_outputs(index: faiss.Index, texts: List[str], metadata: List[Dict], embeddings: np.ndarray):
    """Save FAISS index, texts, metadata, and embeddings to disk."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(OUTPUT_DIR, "pdf_faiss.index"))
    np.save(os.path.join(OUTPUT_DIR, "pdf_texts.npy"), np.array(texts, dtype=object))
    np.save(os.path.join(OUTPUT_DIR, "pdf_embeddings.npy"), embeddings)
    with open(os.path.join(OUTPUT_DIR, "pdf_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"💾 Saved FAISS index and embeddings to {OUTPUT_DIR}")

# =========================================================
# RETRIEVAL METHODS
# =========================================================
def retrieve_context(query: str, method: str = "hybrid", top_k: int = 5, alpha: float = 0.6) -> Tuple[List[str], List[Dict]]:
    """
    Retrieve relevant context chunks for a query.
    
    Args:
        query: Search query
        method: 'semantic', 'hybrid', or 'reranked'
        top_k: Number of chunks to retrieve
        alpha: Weight for hybrid search (semantic vs BM25)
    
    Returns:
        List of text chunks and their metadata
    """
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(os.path.join(OUTPUT_DIR, "pdf_faiss.index"))
    texts = np.load(os.path.join(OUTPUT_DIR, "pdf_texts.npy"), allow_pickle=True)
    
    with open(os.path.join(OUTPUT_DIR, "pdf_metadata.json"), "r") as f:
        metadata = json.load(f)
    
    if method == "semantic":
        # Pure semantic search
        q_emb = model.encode([query], convert_to_numpy=True)
        D, I = index.search(q_emb, top_k)
        indices = I[0]
        
    elif method == "hybrid":
        # Hybrid: BM25 + Semantic
        q_emb = model.encode([query], convert_to_numpy=True)
        D_sem, I_sem = index.search(q_emb, len(texts))
        
        tokenized_corpus = [text.lower().split() for text in texts]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(query.lower().split())
        
        # Normalize and combine
        sem_scores = 1 / (1 + D_sem[0])
        sem_scores = (sem_scores - sem_scores.min()) / (sem_scores.max() - sem_scores.min() + 1e-10)
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-10)
        
        combined_scores = alpha * sem_scores + (1 - alpha) * bm25_scores
        indices = np.argsort(combined_scores)[::-1][:top_k]
        
    elif method == "reranked":
        # Two-stage with reranking
        reranker = CrossEncoder(RERANKER_MODEL)
        q_emb = model.encode([query], convert_to_numpy=True)
        D, I = index.search(q_emb, top_k * 3)
        
        pairs = [[query, texts[idx]] for idx in I[0]]
        rerank_scores = reranker.predict(pairs)
        top_indices = np.argsort(rerank_scores)[::-1][:top_k]
        indices = [I[0][i] for i in top_indices]
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Return texts and metadata
    retrieved_texts = [texts[idx] for idx in indices]
    retrieved_meta = [metadata[idx] for idx in indices]
    
    return retrieved_texts, retrieved_meta

# =========================================================
# LLM ANSWER GENERATION
# =========================================================
def generate_answer_groq(query: str, context_chunks: List[str], metadata: List[Dict]) -> str:
    """Generate answer using Groq API (ultra-fast inference)."""
    from groq import Groq
    
    client = Groq(api_key=GROQ_API_KEY)
    
    # Build context with sources
    context_text = ""
    for i, (chunk, meta) in enumerate(zip(context_chunks, metadata), 1):
        source = meta.get('source', 'Unknown')
        page = meta.get('page_number', 'N/A')
        context_text += f"[Source {i}: {source}, Page {page}]\n{chunk}\n\n"
    
    prompt = f"""You are a medical AI assistant. Answer the following question based ONLY on the provided context from medical documents.

Context:
{context_text}

Question: {query}

Instructions:
- Provide a clear, accurate answer based on the context
- Cite sources using [Source X] notation
- If the context doesn't contain enough information, say so
- Use medical terminology appropriately
- Be concise but comprehensive

Answer:"""

    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",  # Fast and accurate model
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024,
        top_p=1,
        stream=False
    )
    
    return completion.choices[0].message.content

def generate_answer_anthropic(query: str, context_chunks: List[str], metadata: List[Dict]) -> str:
    """Generate answer using Anthropic Claude API."""
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    # Build context with sources
    context_text = ""
    for i, (chunk, meta) in enumerate(zip(context_chunks, metadata), 1):
        source = meta.get('source', 'Unknown')
        page = meta.get('page_number', 'N/A')
        context_text += f"[Source {i}: {source}, Page {page}]\n{chunk}\n\n"
    
    prompt = f"""You are a medical AI assistant. Answer the following question based ONLY on the provided context from medical documents.

Context:
{context_text}

Question: {query}

Instructions:
- Provide a clear, accurate answer based on the context
- Cite sources using [Source X] notation
- If the context doesn't contain enough information, say so
- Use medical terminology appropriately
- Be concise but comprehensive

Answer:"""

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return message.content[0].text

def generate_answer_openai(query: str, context_chunks: List[str], metadata: List[Dict]) -> str:
    """Generate answer using OpenAI API."""
    import openai
    openai.api_key = OPENAI_API_KEY
    
    context_text = ""
    for i, (chunk, meta) in enumerate(zip(context_chunks, metadata), 1):
        source = meta.get('source', 'Unknown')
        page = meta.get('page_number', 'N/A')
        context_text += f"[Source {i}: {source}, Page {page}]\n{chunk}\n\n"
    
    prompt = f"""You are a medical AI assistant. Answer the following question based ONLY on the provided context from medical documents.

Context:
{context_text}

Question: {query}

Instructions:
- Provide a clear, accurate answer based on the context
- Cite sources using [Source X] notation
- If the context doesn't contain enough information, say so
- Use medical terminology appropriately
- Be concise but comprehensive

Answer:"""

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.3
    )
    
    return response.choices[0].message.content

def generate_answer_local(query: str, context_chunks: List[str], metadata: List[Dict]) -> str:
    """Generate answer using a local model (e.g., via transformers)."""
    from transformers import pipeline
    
    # Load a local model (example: FLAN-T5)
    generator = pipeline("text2text-generation", model="google/flan-t5-large")
    
    context_text = ""
    for i, (chunk, meta) in enumerate(zip(context_chunks, metadata), 1):
        context_text += f"Source {i}: {chunk}\n\n"
    
    prompt = f"""Answer this medical question using the provided context.

Context: {context_text[:2000]}  # Truncate for local model

Question: {query}

Answer:"""

    result = generator(prompt, max_length=512, do_sample=False)
    return result[0]['generated_text']

# =========================================================
# MAIN RAG PIPELINE
# =========================================================
def rag_query(query: str, 
              retrieval_method: str = "hybrid", 
              top_k: int = 5, 
              llm_backend: str = LLM_BACKEND,
              verbose: bool = True) -> Dict:
    """
    Complete RAG pipeline: retrieve context and generate answer.
    
    Args:
        query: User question
        retrieval_method: 'semantic', 'hybrid', or 'reranked'
        top_k: Number of context chunks to retrieve
        llm_backend: 'anthropic', 'openai', or 'local'
        verbose: Print retrieval details
    
    Returns:
        Dictionary with answer, sources, and context
    """
    print(f"\n{'='*60}")
    print(f"🔍 Query: {query}")
    print(f"{'='*60}\n")
    
    # Step 1: Retrieve relevant context
    print(f"📚 Retrieving context using {retrieval_method} method...")
    context_chunks, metadata = retrieve_context(query, method=retrieval_method, top_k=top_k)
    
    if verbose:
        print(f"\n✅ Retrieved {len(context_chunks)} relevant chunks:\n")
        for i, (chunk, meta) in enumerate(zip(context_chunks, metadata), 1):
            print(f"[{i}] {meta['source']} (Page {meta.get('page_number', 'N/A')})")
            print(f"    {chunk[:150]}...\n")
    
    # Step 2: Generate answer using LLM
    print(f"🤖 Generating answer using {llm_backend}...\n")
    
    try:
        if llm_backend == "groq":
            answer = generate_answer_groq(query, context_chunks, metadata)
        elif llm_backend == "anthropic":
            answer = generate_answer_anthropic(query, context_chunks, metadata)
        elif llm_backend == "openai":
            answer = generate_answer_openai(query, context_chunks, metadata)
        elif llm_backend == "local":
            answer = generate_answer_local(query, context_chunks, metadata)
        else:
            raise ValueError(f"Unknown LLM backend: {llm_backend}")
    except Exception as e:
        print(f"❌ Error generating answer: {e}")
        answer = "Error: Could not generate answer. Please check your API keys or model configuration."
    
    # Display answer
    print(f"{'─'*60}")
    print(f"💡 ANSWER:\n")
    print(answer)
    print(f"\n{'─'*60}")
    
    # Display sources
    print(f"\n📖 SOURCES:")
    for i, meta in enumerate(metadata, 1):
        print(f"  [{i}] {meta['source']} - Page {meta.get('page_number', 'N/A')}")
    print(f"{'='*60}\n")
    
    return {
        "query": query,
        "answer": answer,
        "context": context_chunks,
        "metadata": metadata,
        "method": retrieval_method
    }

# =========================================================
# INTERACTIVE MODE
# =========================================================
def interactive_rag():
    """Interactive Q&A session."""
    print("\n" + "="*60)
    print("🏥 MEDICAL RAG SYSTEM - Interactive Mode")
    print("="*60)
    print("\nCommands:")
    print("  - Type your question to get an answer")
    print("  - Type 'method:<name>' to change retrieval (semantic/hybrid/reranked)")
    print("  - Type 'exit' or 'quit' to stop")
    print("\n" + "="*60 + "\n")
    
    current_method = "hybrid"
    
    while True:
        query = input("🔍 Your question: ").strip()
        
        if not query:
            continue
            
        if query.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Goodbye!")
            break
            
        if query.lower().startswith('method:'):
            method = query.split(':')[1].strip()
            if method in ['semantic', 'hybrid', 'reranked']:
                current_method = method
                print(f"✅ Switched to {method} retrieval\n")
            else:
                print(f"❌ Invalid method. Use: semantic, hybrid, or reranked\n")
            continue
        
        # Process query
        rag_query(query, retrieval_method=current_method, top_k=5)

# =========================================================
# BATCH PROCESSING
# =========================================================
def batch_rag(queries: List[str], output_file: str = "rag_results.json"):
    """Process multiple queries and save results."""
    results = []
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Processing Query {i}/{len(queries)}")
        print(f"{'='*60}")
        
        result = rag_query(query, verbose=False)
        results.append(result)
    
    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\n💾 Saved {len(results)} results to {output_file}")
    return results

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    # Build index (set to True for first run)
    BUILD_INDEX = False
    
    if BUILD_INDEX:
        texts, meta = load_json_folder(JSON_FOLDER)
        if not texts:
            raise RuntimeError("No text found in any JSON file.")
        
        index, embeddings = build_index(texts)
        save_outputs(index, texts, meta, embeddings)
        print("\n✅ Index built successfully!\n")
    
    # Example usage
    print("\n" + "="*60)
    print("🏥 MEDICAL RAG SYSTEM")
    print("="*60)
    
    # Single query example
    rag_query(
        query="How is breast cancer detected and what are the screening methods?",
        retrieval_method="hybrid",
        top_k=5
    )
    
    # Multiple queries example
    # queries = [
    #     "What are the symptoms of lung cancer?",
    #     "How is oral cancer treated?",
    #     "What are the risk factors for breast cancer?"
    # ]
    # batch_rag(queries)
    
    # Interactive mode (uncomment to use)
    # interactive_rag()