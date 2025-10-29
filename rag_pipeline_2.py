import os, json, faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import nltk
from typing import List, Dict, Tuple, Optional

# =========================================================
# CONFIG
# =========================================================
JSON_FOLDER = "jsons_4"
OUTPUT_DIR = "embeddings_biomed"
MODEL_NAME = "pritamdeka/S-BioBert-snli-multinli-stsb"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Medical term expansions for query enhancement
MEDICAL_EXPANSIONS = {
    'cancer': ['carcinoma', 'tumor', 'tumour', 'neoplasm', 'malignancy'],
    'treatment': ['therapy', 'intervention', 'management', 'care'],
    'diagnosis': ['detection', 'screening', 'identification', 'examination'],
    'symptom': ['sign', 'manifestation', 'indicator'],
    'disease': ['condition', 'disorder', 'illness', 'pathology'],
    'patient': ['individual', 'case', 'subject'],
    'procedure': ['operation', 'surgery', 'intervention'],
    'medication': ['drug', 'medicine', 'pharmaceutical', 'therapeutic']
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

        # Handle nested page/paragraph structure
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
# STEP 2: EMBEDDINGS + FAISS
# =========================================================
def build_index(chunks: List[str]) -> Tuple[faiss.Index, np.ndarray, SentenceTransformer]:
    """Build FAISS index from text chunks."""
    print(f"🔧 Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"✅ FAISS index built with {embeddings.shape[0]} vectors (dim={dim}).")
    return index, embeddings, model

# =========================================================
# STEP 3: SAVE OUTPUTS
# =========================================================
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
# UTILITY: QUERY EXPANSION
# =========================================================
def expand_query(query: str, expansions: Dict[str, List[str]] = MEDICAL_EXPANSIONS) -> List[str]:
    """Expand query with medical synonyms."""
    expanded = [query]
    query_lower = query.lower()
    
    for term, synonyms in expansions.items():
        if term in query_lower:
            for syn in synonyms[:2]:  # Limit to 2 synonyms per term
                expanded_query = query_lower.replace(term, syn)
                if expanded_query != query_lower:
                    expanded.append(expanded_query)
    
    return list(set(expanded))  # Remove duplicates

# =========================================================
# SEARCH METHOD 1: BASIC SEMANTIC SEARCH
# =========================================================
def semantic_search(query: str, top_k: int = 3, filter_source: Optional[str] = None) -> List[Tuple[int, float]]:
    """Basic semantic search using FAISS."""
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(os.path.join(OUTPUT_DIR, "pdf_faiss.index"))
    texts = np.load(os.path.join(OUTPUT_DIR, "pdf_texts.npy"), allow_pickle=True)
    
    with open(os.path.join(OUTPUT_DIR, "pdf_metadata.json"), "r") as f:
        metadata = json.load(f)
    
    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k * 3 if filter_source else top_k)
    
    results = []
    print(f"\n🔎 Semantic Search Results for: '{query}'\n")
    
    for idx, score in zip(I[0], D[0]):
        meta = metadata[idx]
        
        # Apply filters
        if filter_source and meta['source'] != filter_source:
            continue
            
        results.append((idx, score))
        
        if len(results) >= top_k:
            break
    
    for idx, score in results:
        meta = metadata[idx]
        print(f"→ [Score: {score:.4f}] Source: {meta['source']}, Page: {meta.get('page_number', 'N/A')}")
        print(f"   {texts[idx][:200]}...\n")
    
    return results

# =========================================================
# SEARCH METHOD 2: HYBRID SEARCH (BM25 + SEMANTIC)
# =========================================================
def hybrid_search(query: str, top_k: int = 3, alpha: float = 0.5) -> List[Tuple[int, float]]:
    """
    Hybrid search combining BM25 and semantic search.
    
    Args:
        query: Search query
        top_k: Number of results
        alpha: Weight for semantic search (1-alpha for BM25). Range: 0-1
    """
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(os.path.join(OUTPUT_DIR, "pdf_faiss.index"))
    texts = np.load(os.path.join(OUTPUT_DIR, "pdf_texts.npy"), allow_pickle=True)
    
    with open(os.path.join(OUTPUT_DIR, "pdf_metadata.json"), "r") as f:
        metadata = json.load(f)
    
    # Semantic search
    q_emb = model.encode([query], convert_to_numpy=True)
    D_sem, I_sem = index.search(q_emb, len(texts))
    
    # BM25 search
    tokenized_corpus = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.lower().split())
    
    # Normalize scores to [0, 1]
    sem_scores = 1 / (1 + D_sem[0])  # Convert L2 distance to similarity
    sem_scores = (sem_scores - sem_scores.min()) / (sem_scores.max() - sem_scores.min() + 1e-10)
    bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-10)
    
    # Combine scores
    combined_scores = alpha * sem_scores + (1 - alpha) * bm25_scores
    top_indices = np.argsort(combined_scores)[::-1][:top_k]
    
    print(f"\n🔎 Hybrid Search Results for: '{query}' (α={alpha})\n")
    
    results = []
    for idx in top_indices:
        meta = metadata[idx]
        combined_score = combined_scores[idx]
        results.append((idx, combined_score))
        
        print(f"→ [Combined: {combined_score:.4f}, Semantic: {sem_scores[idx]:.4f}, BM25: {bm25_scores[idx]:.4f}]")
        print(f"   Source: {meta['source']}, Page: {meta.get('page_number', 'N/A')}")
        print(f"   {texts[idx][:200]}...\n")
    
    return results

# =========================================================
# SEARCH METHOD 3: RERANKED SEARCH
# =========================================================
def reranked_search(query: str, top_k: int = 3, initial_k: int = 20) -> List[Tuple[int, float]]:
    """
    Two-stage search with cross-encoder reranking.
    
    Args:
        query: Search query
        top_k: Final number of results
        initial_k: Number of candidates for reranking
    """
    print(f"🔧 Loading models...")
    model = SentenceTransformer(MODEL_NAME)
    reranker = CrossEncoder(RERANKER_MODEL)
    
    index = faiss.read_index(os.path.join(OUTPUT_DIR, "pdf_faiss.index"))
    texts = np.load(os.path.join(OUTPUT_DIR, "pdf_texts.npy"), allow_pickle=True)
    
    with open(os.path.join(OUTPUT_DIR, "pdf_metadata.json"), "r") as f:
        metadata = json.load(f)
    
    # Stage 1: Initial retrieval
    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, initial_k)
    
    # Stage 2: Rerank with cross-encoder
    pairs = [[query, texts[idx]] for idx in I[0]]
    rerank_scores = reranker.predict(pairs)
    
    # Get top_k after reranking
    top_indices = np.argsort(rerank_scores)[::-1][:top_k]
    
    print(f"\n🔎 Reranked Search Results for: '{query}'\n")
    
    results = []
    for i in top_indices:
        idx = I[0][i]
        meta = metadata[idx]
        rerank_score = rerank_scores[i]
        initial_score = D[0][i]
        results.append((idx, rerank_score))
        
        print(f"→ [Rerank: {rerank_score:.4f}, Initial: {initial_score:.4f}]")
        print(f"   Source: {meta['source']}, Page: {meta.get('page_number', 'N/A')}")
        print(f"   {texts[idx][:200]}...\n")
    
    return results

# =========================================================
# SEARCH METHOD 4: QUERY-EXPANDED SEARCH
# =========================================================
def expanded_search(query: str, top_k: int = 3, use_rerank: bool = True) -> List[Tuple[int, float]]:
    """
    Search with automatic query expansion using medical synonyms.
    
    Args:
        query: Original search query
        top_k: Number of results
        use_rerank: Whether to use reranking
    """
    expanded_queries = expand_query(query)
    print(f"\n🔍 Query Expansion:")
    print(f"   Original: {query}")
    print(f"   Expanded: {expanded_queries[1:]}\n")
    
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(os.path.join(OUTPUT_DIR, "pdf_faiss.index"))
    texts = np.load(os.path.join(OUTPUT_DIR, "pdf_texts.npy"), allow_pickle=True)
    
    with open(os.path.join(OUTPUT_DIR, "pdf_metadata.json"), "r") as f:
        metadata = json.load(f)
    
    # Search with all expanded queries
    all_indices = set()
    all_scores = {}
    
    for exp_query in expanded_queries:
        q_emb = model.encode([exp_query], convert_to_numpy=True)
        D, I = index.search(q_emb, top_k * 2)
        
        for idx, score in zip(I[0], D[0]):
            all_indices.add(idx)
            if idx not in all_scores or score < all_scores[idx]:
                all_scores[idx] = score
    
    # Sort by best score
    sorted_results = sorted(all_scores.items(), key=lambda x: x[1])[:top_k * 2]
    
    # Optional reranking
    if use_rerank and len(sorted_results) > top_k:
        reranker = CrossEncoder(RERANKER_MODEL)
        pairs = [[query, texts[idx]] for idx, _ in sorted_results]
        rerank_scores = reranker.predict(pairs)
        
        # Combine indices with rerank scores
        reranked = [(sorted_results[i][0], rerank_scores[i]) for i in range(len(sorted_results))]
        reranked.sort(key=lambda x: x[1], reverse=True)
        final_results = reranked[:top_k]
    else:
        final_results = sorted_results[:top_k]
    
    print(f"🔎 Expanded Search Results for: '{query}'\n")
    
    for idx, score in final_results:
        meta = metadata[idx]
        print(f"→ [Score: {score:.4f}]")
        print(f"   Source: {meta['source']}, Page: {meta.get('page_number', 'N/A')}")
        print(f"   {texts[idx][:200]}...\n")
    
    return final_results

# =========================================================
# BATCH EVALUATION
# =========================================================
def batch_search(queries: List[str], method: str = "semantic", top_k: int = 3):
    """
    Run multiple queries and compare results.
    
    Args:
        queries: List of search queries
        method: Search method ('semantic', 'hybrid', 'reranked', 'expanded')
        top_k: Number of results per query
    """
    print(f"\n{'='*60}")
    print(f"BATCH SEARCH - Method: {method.upper()}")
    print(f"{'='*60}\n")
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i}/{len(queries)} ---")
        
        if method == "semantic":
            semantic_search(query, top_k)
        elif method == "hybrid":
            hybrid_search(query, top_k)
        elif method == "reranked":
            reranked_search(query, top_k)
        elif method == "expanded":
            expanded_search(query, top_k)
        else:
            print(f"Unknown method: {method}")
        
        print(f"{'─'*60}")
# =========================================================
# STEP 5: ANSWER GENERATOR (LOCAL HUGGING FACE MODEL)
# =========================================================
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

def generate_answer_from_chunks(query: str, results: List[Tuple[int, float]], top_k: int = 3):
    """
    Generate an answer using retrieved context.
    Works fully offline if the model is already cached.
    """
    from transformers import pipeline

    # Load retrieved texts
    texts = np.load(os.path.join(OUTPUT_DIR, "pdf_texts.npy"), allow_pickle=True)
    with open(os.path.join(OUTPUT_DIR, "pdf_metadata.json"), "r") as f:
        metadata = json.load(f)

    # Collect context from top_k results
    context = ""
    for idx, score in results[:top_k]:
        meta = metadata[idx]
        context += f"Source: {meta['source']} (Page {meta.get('page_number', 'N/A')})\n{texts[idx]}\n\n"

    # Select a local summarization / instruction model
    MODEL_ID = "google/flan-t5-large"  # Replace with flan-t5-xl if you have GPU
    generator = pipeline("text2text-generation", model=MODEL_ID, device_map="auto")

    prompt = f"""
You are a biomedical expert. Using the provided context, answer the user's question accurately and concisely.

Question: {query}

Context:
{context[:3000]}

Answer clearly and medically precisely:
"""

    print("🧠 Generating answer...")
    response = generator(prompt, max_new_tokens=256, do_sample=False)[0]["generated_text"]
    print("\n🩺 Final Answer:")
    print(response)
    return response

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    # Build index (run once)
    BUILD_INDEX = False  # Set to True for first run
    
    if BUILD_INDEX:
        texts, meta = load_json_folder(JSON_FOLDER)
        if not texts:
            raise RuntimeError("No text found in any JSON file. Check your folder or JSON format.")
        
        index, embeddings, model = build_index(texts)
        save_outputs(index, texts, meta, embeddings)
    
    # Example searches
    print("\n" + "="*60)
    print("ENHANCED RAG PIPELINE - SEARCH EXAMPLES")
    print("="*60)
    
    # Test queries
    test_queries = [
        "how to find breast cancer",
        "lung cancer treatment mechanism",
        "what is oral cancer"
    ]
    
    # Method 1: Basic Semantic Search
    print("\n\n### METHOD 1: SEMANTIC SEARCH ###")
    semantic_search(test_queries[0], top_k=3)
    
    # Method 2: Hybrid Search
    print("\n\n### METHOD 2: HYBRID SEARCH ###")
    hybrid_search(test_queries[1], top_k=3, alpha=0.6)
    
    # Method 3: Reranked Search
    print("\n\n### METHOD 3: RERANKED SEARCH ###")
    reranked_search(test_queries[2], top_k=3, initial_k=15)
    
    # Method 4: Expanded Search
    print("\n\n### METHOD 4: EXPANDED SEARCH ###")
    expanded_search("cancer treatment", top_k=3, use_rerank=True)
    
    # Batch evaluation (uncomment to run)
    # batch_search(test_queries, method="hybrid", top_k=3)