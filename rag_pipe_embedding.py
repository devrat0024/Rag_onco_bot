import os, json, faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# =========================================================
# CONFIG
# =========================================================
JSON_FOLDER = "jsons_4"      # folder containing your structured JSONs
OUTPUT_DIR = "embeddings_biomed"
MODEL_NAME = "pritamdeka/S-BioBert-snli-multinli-stsb"

# =========================================================
# STEP 1: LOAD TEXTS FROM ALL JSON FILES
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

        # === Handle nested page/paragraph structure ===
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
    faiss.write_index(index, os.path.join(OUTPUT_DIR, "pdf_faiss.index"))
    np.save(os.path.join(OUTPUT_DIR, "pdf_texts.npy"), np.array(texts, dtype=object))
    np.save(os.path.join(OUTPUT_DIR, "pdf_embeddings.npy"), embeddings)
    with open(os.path.join(OUTPUT_DIR, "pdf_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"💾 Saved FAISS index and embeddings to {OUTPUT_DIR}")

# =========================================================
# STEP 4: RETRIEVAL (for RAG)
# =========================================================
def search(query, top_k=3):
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(os.path.join(OUTPUT_DIR, "pdf_faiss.index"))
    texts = np.load(os.path.join(OUTPUT_DIR, "pdf_texts.npy"), allow_pickle=True)
    
    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k)
    
    print(f"\n🔎 Query Results for: '{query}'\n")
    for idx, score in zip(I[0], D[0]):
        print(f"→ [{score:.4f}] {texts[idx][:250]}...\n")

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    texts, meta = load_json_folder(JSON_FOLDER)
    if not texts:
        raise RuntimeError("No text found in any JSON file. Check your folder or JSON format.")
    
    index, embeddings = build_index(texts)
    save_outputs(index, texts, meta, embeddings)
    search("how to find breast cancer ", top_k=3)



