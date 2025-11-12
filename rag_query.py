#!/usr/bin/env python3
# =========================================================
# 🧠 Dr. Nova – RAG Query (Short Conversational Replies)
# Author: Devvrat Shukla + GPT-5
# =========================================================

import os, json, faiss, numpy as np, torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

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
GENERATOR_MODEL = "google/flan-t5-large"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🧠 Using device: {device}")

# =========================================================
# 📂 Load FAISS + Data (FIXED PATHS)
# =========================================================
def _load_data():
    index_path = os.path.join(OUTPUT_DIR, "faiss_index.index")  # Fixed path
    text_path = os.path.join(OUTPUT_DIR, "texts.npy")  # Fixed path
    meta_path = os.path.join(OUTPUT_DIR, "metadata.json")  # Fixed path

    if not all(os.path.exists(p) for p in [index_path, text_path, meta_path]):
        raise FileNotFoundError("❌ Missing FAISS or embedding data. Run 'rag_pipe_embedding.py' first.")

    index = faiss.read_index(index_path)
    texts = np.load(text_path, allow_pickle=True)
    metadata = json.load(open(meta_path, encoding="utf-8"))
    return index, texts, metadata

# =========================================================
# 🔍 Retrieve Relevant Context
# =========================================================
def retrieve_context(query: str, top_k: int = 3):
    index, texts, metadata = _load_data()
    embedder = SentenceTransformer(EMBED_MODEL, device=device)
    q_emb = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, len(texts))

    sem_scores = 1 / (1 + D[0])
    sem_scores = (sem_scores - sem_scores.min()) / (sem_scores.max() - sem_scores.min() + 1e-10)

    bm25 = BM25Okapi([t.lower().split() for t in texts])
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-10)

    combined = 0.6 * sem_scores + 0.4 * bm25_scores
    top_idx = np.argsort(combined)[::-1][:top_k]

    chosen_texts = [texts[i] for i in top_idx]
    chosen_meta = [metadata[i] for i in top_idx]
    return chosen_texts, chosen_meta

# =========================================================
# 🛠️ Text Truncation Helper
# =========================================================
def truncate_for_model(text, max_tokens=400):
    """Truncate text to prevent token overflow"""
    words = text.split()
    if len(words) > max_tokens:
        return " ".join(words[:max_tokens]) + "..."
    return text

# =========================================================
# 🤖 Generate Short Conversational Answer
# =========================================================
def generate_answer(query: str, context: str, chat_context: str = ""):
    style_instruction = (
        "Reply like a friendly doctor. Keep it very short (1–2 sentences). "
        "Be warm, natural, and empathetic — not robotic or overly formal."
    )

    # Truncate inputs to prevent token overflow
    truncated_context = truncate_for_model(context, 300)
    truncated_chat = truncate_for_model(chat_context, 200)

    prompt = f"""
Chat history:
{truncated_chat}

User question: {query}

Relevant info:
{truncated_context}

Instruction: {style_instruction}
"""

    # Try Groq API first with better error handling
    if groq_api_key:
        try:
            # Use requests-based fallback to avoid httpx issues
            import requests
            headers = {
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "messages": [
                    {"role": "system", "content": "You are a kind, concise medical counselor."},
                    {"role": "user", "content": prompt}
                ],
                "model": "llama-3.3-70b-versatile",
                "temperature": 0.7,
                "max_tokens": 150
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            answer = result['choices'][0]['message']['content'].strip()
            print("✅ Groq API generated a response.")
            return answer
        except Exception as e:
            print(f"⚠️ Groq API failed: {e}")
            print("💻 Using local model fallback...")

    # Local model fallback with better token handling
    try:
        # Initialize tokenizer and generator separately
        tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
        generator = pipeline(
            "text2text-generation", 
            model=GENERATOR_MODEL, 
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Tokenize and truncate if necessary
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        result = generator(
            prompt,
            max_new_tokens=100,
            do_sample=False,
            truncation=True
        )[0]["generated_text"].strip()
        
        # Keep it short — 1–2 sentences
        sentences = result.split(". ")
        final_answer = ". ".join(sentences[:2]) + ("." if not result.endswith(".") else "")
        return final_answer
        
    except Exception as e:
        print(f"❌ Local model generation failed: {e}")
        return "I apologize, but I'm having trouble generating a response right now. Please try again or rephrase your question."

# =========================================================
# 🩺 Full RAG Query
# =========================================================
def rag_query(query: str, top_k: int = 3, chat_context: str = ""):
    """Retrieve → Generate → Return short conversational answer."""
    try:
        chunks, meta = retrieve_context(query, top_k)
        if not chunks:
            return {
                "answer": "I don't have specific information about that in my knowledge base. Please consult with a healthcare professional for personalized medical advice.",
                "metadata": [],
                "safety_warning": False
            }
        answer = generate_answer(query, "\n".join(chunks), chat_context)
        return {
            "answer": answer, 
            "metadata": meta,
            "safety_warning": False
        }
    except Exception as e:
        print(f"❌ RAG query error: {e}")
        return {
            "answer": "I'm experiencing technical difficulties. Please try again in a moment.",
            "metadata": [],
            "safety_warning": True
        }

# =========================================================
# 🧪 Test Run
# =========================================================
if __name__ == "__main__":
    q = "What is oral cancer?"
    result = rag_query(q)
    print("\n🩺 Final Answer:\n", result["answer"])