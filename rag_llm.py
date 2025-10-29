#!/usr/bin/env python3
# =========================================================
# 🧬 ENHANCED BIOMEDICAL RAG PIPELINE
# Author: Dev + GPT-5
# =========================================================
import os, json, faiss, torch, nltk, numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# =========================================================
# CONFIGURATION
# =========================================================
JSON_FOLDER = "jsons_4"
OUTPUT_DIR = "embeddings_biomed"

# Embedding and Reranker Models
MODEL_NAME = "pritamdeka/S-BioBert-snli-multinli-stsb"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GENERATOR_MODEL = "google/flan-t5-large"  # or "microsoft/BioGPT-Large"

# =========================================================
# MEDICAL TERM EXPANSION
# =========================================================
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
# STEP 1: LOAD JSON FILES
# =========================================================
def load_json_folder(json_folder: str) -> Tuple[List[str], List[Dict]]:
    """Load text and metadata from JSON files."""
    texts, meta = [], []
    if not os.path.exists(json_folder):
        raise FileNotFoundError(f"❌ Folder '{json_folder}' not found.")

    files = [f for f in os.listdir(json_folder) if f.endswith(".json")]
    print(f"📂 Found {len(files)} JSON files in '{json_folder}'")

    for file in tqdm(files, desc="Loading JSON files"):
        path = os.path.join(json_folder, file)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️ Skipping {file}: {e}")
            continue

        if isinstance(data, dict) and "pages" in data:
            file_name = data.get("file_name", file)
            for page in data["pages"]:
                page_num = page.get("page_number", None)
                for pid, para in enumerate(page.get("paragraphs", [])):
                    text = para.get("paragraph_text", "").strip()
                    if text and len(text) > 20:
                        texts.append(text)
                        meta.append({
                            "source": file_name,
                            "page_number": page_num,
                            "para_id": pid,
                            "preview": text[:150]
                        })
    # Deduplicate
    uniq = list(dict.fromkeys(texts))
    print(f"✅ Loaded {len(uniq)} unique text segments from {len(files)} files.")
    return uniq, meta[:len(uniq)]

# =========================================================
# STEP 2: BUILD FAISS INDEX
# =========================================================
def build_index(chunks: List[str]):
    """Build and save FAISS index."""
    model = SentenceTransformer(MODEL_NAME, device='cuda' if torch.cuda.is_available() else 'cpu')
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(OUTPUT_DIR, "pdf_faiss.index"))
    np.save(os.path.join(OUTPUT_DIR, "pdf_embeddings.npy"), embeddings)
    np.save(os.path.join(OUTPUT_DIR, "pdf_texts.npy"), np.array(chunks, dtype=object))
    with open(os.path.join(OUTPUT_DIR, "pdf_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    print(f"💾 Saved FAISS index with {len(chunks)} vectors (dim={dim}).")

# =========================================================
# STEP 3: QUERY EXPANSION
# =========================================================
def expand_query(query: str, expansions=MEDICAL_EXPANSIONS):
    expanded = [query]
    query_lower = query.lower()
    for term, synonyms in expansions.items():
        if term in query_lower:
            for syn in synonyms[:2]:
                expanded.append(query_lower.replace(term, syn))
    return list(set(expanded))

# =========================================================
# STEP 4: SEMANTIC SEARCH
# =========================================================
def semantic_search(query: str, top_k: int = 3):
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(os.path.join(OUTPUT_DIR, "pdf_faiss.index"))
    texts = np.load(os.path.join(OUTPUT_DIR, "pdf_texts.npy"), allow_pickle=True)
    metadata = json.load(open(os.path.join(OUTPUT_DIR, "pdf_metadata.json")))

    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k)
    results = [(idx, D[0][i]) for i, idx in enumerate(I[0])]

    print(f"\n🔎 Results for: '{query}'\n")
    for idx, score in results:
        meta = metadata[idx]
        print(f"→ [Score: {score:.4f}] {meta['source']} - Pg {meta['page_number']}")
        print(f"   {texts[idx][:200]}...\n")
    return results

# =========================================================
# STEP 5: HYBRID SEARCH (BM25 + SEMANTIC)
# =========================================================
def hybrid_search(query: str, alpha=0.5, top_k=5):
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(os.path.join(OUTPUT_DIR, "pdf_faiss.index"))
    texts = np.load(os.path.join(OUTPUT_DIR, "pdf_texts.npy"), allow_pickle=True)
    metadata = json.load(open(os.path.join(OUTPUT_DIR, "pdf_metadata.json")))

    tokenized_corpus = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.lower().split())

    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, len(texts))
    sem_scores = 1 / (1 + D[0])
    sem_scores = (sem_scores - sem_scores.min()) / (sem_scores.max() - sem_scores.min() + 1e-10)
    bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-10)
    combined = alpha * sem_scores + (1 - alpha) * bm25_scores

    top_idx = np.argsort(combined)[::-1][:top_k]
    print(f"\n🔍 Hybrid Search: '{query}' (α={alpha})\n")
    for idx in top_idx:
        meta = metadata[idx]
        print(f"→ [{combined[idx]:.4f}] {meta['source']} Pg:{meta['page_number']}")
        print(f"   {texts[idx][:200]}...\n")
    return [(idx, combined[idx]) for idx in top_idx]

# =========================================================
# STEP 6: RERANKED SEARCH
# =========================================================
def reranked_search(query: str, top_k: int = 3, initial_k: int = 20):
    model = SentenceTransformer(MODEL_NAME)
    reranker = CrossEncoder(RERANKER_MODEL, device='cuda' if torch.cuda.is_available() else 'cpu')

    index = faiss.read_index(os.path.join(OUTPUT_DIR, "pdf_faiss.index"))
    texts = np.load(os.path.join(OUTPUT_DIR, "pdf_texts.npy"), allow_pickle=True)
    metadata = json.load(open(os.path.join(OUTPUT_DIR, "pdf_metadata.json")))

    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, initial_k)

    pairs = [[query, texts[idx]] for idx in I[0]]
    scores = reranker.predict(pairs, show_progress_bar=True)

    top_idx = np.argsort(scores)[::-1][:top_k]
    print(f"\n🔎 Reranked Search Results for: '{query}'\n")
    for i in top_idx:
        idx = I[0][i]
        meta = metadata[idx]
        print(f"→ [Rerank: {scores[i]:.4f}] {meta['source']} Pg:{meta['page_number']}")
        print(f"   {texts[idx][:200]}...\n")
    return [(I[0][i], scores[i]) for i in top_idx]

# =========================================================
# STEP 7: ANSWER GENERATION
# =========================================================
def generate_answer(query: str, results: List[Tuple[int, float]], top_k=3):
    texts = np.load(os.path.join(OUTPUT_DIR, "pdf_texts.npy"), allow_pickle=True)
    metadata = json.load(open(os.path.join(OUTPUT_DIR, "pdf_metadata.json")))

    context = ""
    for idx, _ in results[:top_k]:
        meta = metadata[idx]
        context += f"Source: {meta['source']} (Pg {meta['page_number']})\n{texts[idx]}\n\n"

    generator = pipeline("text2text-generation", model=GENERATOR_MODEL, device_map="auto")
    prompt = f"""
You are a biomedical expert. Answer precisely using context.

Question: {query}
Context: {context[:3000]}
"""
    print("🧠 Generating answer...")
    response = generator(prompt, max_new_tokens=256, do_sample=False)[0]['generated_text']
    print("\n🩺 Final Answer:\n", response)
    return response

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    BUILD_INDEX = False  # ⚠️ Change to True to rebuild index
    if BUILD_INDEX:
        texts, meta = load_json_folder(JSON_FOLDER)
        if not texts:
            raise RuntimeError("No text extracted. Check JSON structure.")
        build_index(texts)

    # Example runs
    q = "How is breast cancer detected?"
    results = hybrid_search(q, alpha=0.6, top_k=3)
    generate_answer(q, results, top_k=2)
