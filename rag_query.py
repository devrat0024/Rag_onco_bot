#!/usr/bin/env python3
# =========================================================
# 🧠 Cancer Counselling RAG Query (Groq + Local, Concise & Empathetic)
# Author: Devvrat Shukla + GPT-5
# =========================================================

import os, json, faiss, numpy as np, torch
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

# =========================================================
# ⚙️ Environment Setup
# =========================================================
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    print("🔑 GROQ_API_KEY loaded successfully.")
else:
    print("⚠️ GROQ_API_KEY not found. Running in local-only mode.")

# =========================================================
# 📦 Configuration
# =========================================================
OUTPUT_DIR = "embeddings_biomed"
EMBED_MODEL = "pritamdeka/S-BioBert-snli-multinli-stsb"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GENERATOR_MODEL = "google/flan-t5-large"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🧠 Using device: {device}")

# =========================================================
# 📂 Load FAISS Data
# =========================================================
def _load_data():
    index_path = os.path.join(OUTPUT_DIR, "pdf_faiss.index")
    text_path = os.path.join(OUTPUT_DIR, "pdf_texts.npy")
    meta_path = os.path.join(OUTPUT_DIR, "pdf_metadata.json")

    if not all(os.path.exists(p) for p in [index_path, text_path, meta_path]):
        raise FileNotFoundError(
            "❌ Missing FAISS or embedding data. Run 'rag_llm.py' with BUILD_INDEX=True first."
        )

    index = faiss.read_index(index_path)
    texts = np.load(text_path, allow_pickle=True)
    metadata = json.load(open(meta_path, encoding="utf-8"))
    return index, texts, metadata

# =========================================================
# 🔍 Retrieve Relevant Context
# =========================================================
def retrieve_context(query: str, method: str = "hybrid", top_k: int = 5):
    """Retrieve top-k relevant segments using FAISS + BM25 + reranker."""
    index, texts, metadata = _load_data()

    embedder = SentenceTransformer(EMBED_MODEL, device=device)
    q_emb = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, len(texts))

    sem_scores = 1 / (1 + D[0])
    sem_scores = (sem_scores - sem_scores.min()) / (sem_scores.max() - sem_scores.min() + 1e-10)

    tokenized_corpus = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-10)

    if method == "semantic":
        combined = sem_scores
    elif method == "reranked":
        reranker = CrossEncoder(RERANK_MODEL, device=device)
        pairs = [[query, texts[idx]] for idx in I[0][:50]]
        scores = reranker.predict(pairs, show_progress_bar=False)
        top_idx = np.argsort(scores)[::-1][:top_k]
        chosen_idx = [I[0][i] for i in top_idx]
        return [texts[i] for i in chosen_idx], [metadata[i] for i in chosen_idx]
    else:
        alpha = 0.6
        combined = alpha * sem_scores + (1 - alpha) * bm25_scores

    top_idx = np.argsort(combined)[::-1][:top_k]
    chosen_texts = [texts[i] for i in top_idx]
    chosen_meta = [metadata[i] for i in top_idx]
    return chosen_texts, chosen_meta

# =========================================================
# 🤖 Generate Response (Groq API + Local Fallback)
# =========================================================
def generate_answer(query: str, context: str, style: str = "Simple & Short"):
    """
    Generate a concise, empathetic response using Groq API if available,
    else fall back to a local Flan-T5 model.
    """
    style_instruction = (
        "Keep your answer short (4–6 sentences), simple, and kind. "
        "Avoid medical jargon. Use everyday words and finish with a reassuring tone."
        if style == "Simple & Short"
        else "Provide a detailed but clear explanation, suitable for a general audience."
    )

    # --- Try Groq API first ---
    if groq_api_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_api_key)
            print("🌐 Using Groq API for generation...")

            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a compassionate AI counsellor specialized in oncology. "
                            "You explain medical topics clearly, kindly, and accurately."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"""
Question: {query}

Context (medical info):
{context[:3500]}

Instruction:
{style_instruction}
"""
                    },
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                max_tokens=350,
            )

            answer = chat_completion.choices[0].message.content.strip()
            print("✅ Groq API generated a response.")

            # ✂️ Trim overly long answers (max 8 sentences)
            sentences = answer.split(". ")
            if len(sentences) > 8:
                answer = ". ".join(sentences[:8]) + "."

            return answer

        except Exception as e:
            print("⚠️ Groq API call failed:", e)
            print("💻 Falling back to local model...")

    # --- Local Model Fallback ---
    try:
        generator = pipeline("text2text-generation", model=GENERATOR_MODEL, device_map="auto")
        prompt = f"""
You are a biomedical counsellor. {style_instruction}

Question: {query}

Context:
{context[:3500]}

Answer:
"""
        result = generator(prompt, max_new_tokens=256, do_sample=False)[0]["generated_text"].strip()

        # ✂️ Truncate long local responses too
        sentences = result.split(". ")
        if len(sentences) > 8:
            result = ". ".join(sentences[:8]) + "."

        return result

    except Exception as e:
        print("❌ Local model generation failed:", e)
        return "⚠️ Could not generate answer. Please check your API keys or model configuration."

# =========================================================
# 🩺 Full RAG Query
# =========================================================
def rag_query(query: str, retrieval_method: str = "hybrid", top_k: int = 5, verbose: bool = True, style: str = "Simple & Short"):
    """Full RAG pipeline: retrieve → generate (Groq + Local fallback)."""
    context_chunks, metadata = retrieve_context(query, method=retrieval_method, top_k=top_k)

    if not context_chunks:
        return {"answer": "⚠️ No relevant context found.", "metadata": []}

    combined_context = "\n\n".join(context_chunks)
    answer = generate_answer(query, combined_context, style)
    return {"answer": answer, "metadata": metadata}

# =========================================================
# 🧪 Test Run
# =========================================================
if __name__ == "__main__":
    q = "What is cancer and how does it affect the body?"
    result = rag_query(q, retrieval_method="hybrid", top_k=3, style="Simple & Short")
    print("\n🩺 Final Answer:\n", result["answer"])
