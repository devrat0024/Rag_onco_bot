#!/usr/bin/env python3
# =========================================================
# 🔬 Biomedical RAG Embedding Pipeline
# Handles both structured PDFs and Q&A JSONs
# =========================================================

import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# =========================================================
# CONFIGURATION
# =========================================================
JSON_FOLDER = "jsons_4"          # Folder containing all JSONs
OUTPUT_DIR = "embeddings_biomed" # Output directory
MODEL_NAME = "pritamdeka/S-BioBert-snli-multinli-stsb"

# =========================================================
# STEP 1: LOAD TEXT FROM JSON FILES
# =========================================================
def load_json_folder(json_folder):
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

        # === CASE 1: Structured PDF-paragraph JSON ===
        if isinstance(data, dict) and "pages" in data:
            file_name = data.get("file_name", file)
            for page in data["pages"]:
                page_num = page.get("page_number", None)
                for pid, para in enumerate(page.get("paragraphs", [])):
                    text = para.get("paragraph_text", "").strip()
                    if len(text) < 20:
                        continue
                    texts.append(text)
                    meta.append({
                        "source": file_name,
                        "page_number": page_num,
                        "para_id": pid,
                        "type": "structured_doc",
                        "preview": text[:120]
                    })
            continue

        # === CASE 2: Q&A Medical or Gender-based JSON ===
        elif isinstance(data, dict):
            qna_found = False
            for key, value in data.items():
                if isinstance(value, list) and all(isinstance(v, dict) for v in value):
                    qna_found = True
                    for entry in value:
                        q = entry.get("question", "").strip()
                        answers = entry.get("answers", [])
                        if not q or not isinstance(answers, list):
                            continue
                        for ans in answers:
                            ans = ans.strip()
                            if len(ans) < 10:
                                continue
                            combined_text = f"Question: {q}\nAnswer: {ans}"
                            texts.append(combined_text)
                            meta.append({
                                "source": file,
                                "question": q[:120],
                                "type": "qna_json",
                                "preview": ans[:120]
                            })
                    break

            # Store metadata section if available
            if "metadata" in data and isinstance(data["metadata"], dict):
                meta.append({
                    "source": file,
                    "type": "metadata",
                    "preview": json.dumps(data["metadata"])[:150]
                })

            if not qna_found:
                print(f"⚠️ Unrecognized structure in {file}")
            continue

        else:
            print(f"⚠️ Unrecognized JSON structure in {file}")

    print(f"✅ Loaded {len(texts)} text chunks from {len(files)} files.")
    return texts, meta

# =========================================================
# STEP 2: BUILD FAISS INDEX
# =========================================================
def build_index(chunks):
    print(f"🔍 Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"✅ FAISS index built with {embeddings.shape[0]} vectors.")
    return index, embeddings

# =========================================================
# STEP 3: SAVE OUTPUTS
# =========================================================
def save_outputs(index, texts, metadata, embeddings):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(OUTPUT_DIR, "faiss_index.index"))
    np.save(os.path.join(OUTPUT_DIR, "texts.npy"), np.array(texts, dtype=object))
    np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), embeddings)
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"💾 Saved FAISS index and embeddings to '{OUTPUT_DIR}'")

# =========================================================
# STEP 4: SEARCH (for quick testing)
# =========================================================
def search(query, top_k=3):
    print(f"\n🔎 Searching for: '{query}'\n")
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(os.path.join(OUTPUT_DIR, "faiss_index.index"))
    texts = np.load(os.path.join(OUTPUT_DIR, "texts.npy"), allow_pickle=True)

    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k)

    for idx, score in zip(I[0], D[0]):
        print(f"→ [{score:.4f}] {texts[idx][:250]}...\n")

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    texts, meta = load_json_folder(JSON_FOLDER)

    if not texts:
        raise RuntimeError("❌ No text found in any JSON file. Check your folder or JSON structure.")

    index, embeddings = build_index(texts)
    save_outputs(index, texts, meta, embeddings)

    # Optional: quick test
    search("what are the risk factors for lung cancer in women", top_k=3)
