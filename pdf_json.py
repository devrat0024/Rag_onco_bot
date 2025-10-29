import os, re, json
import numpy as np
import fitz                   # PyMuPDF
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# =========================================================
# CONFIG
# =========================================================
PDF_FOLDER   = "./pdfs"
OUTPUT_DIR   = "embeddings_biomed"
MODEL_NAME   = "pritamdeka/S-BioBert-snli-multinli-stsb"  # biomedical sentence model
CHUNK_SIZE   = 800
CHUNK_OVERLAP = 100

# =========================================================
# STEP 1: CLEAN PARAGRAPH EXTRACTION
# =========================================================
def extract_clean_paragraphs(pdf_path):
    """Return clean information-rich paragraphs from a scientific PDF."""
    doc = fitz.open(pdf_path)
    file_name = os.path.basename(pdf_path)
    abs_path  = os.path.abspath(pdf_path)
    paragraphs = []

    for page in doc:
        text = page.get_text("text")
        # split by blank lines (paragraph boundaries)
        raw_paras = re.split(r'\n{2,}|\r{2,}', text)

        for p in raw_paras:
            t = re.sub(r'\s+', ' ', p).strip()
            if not t:
                continue
            # filters for obvious non-content
            if len(t.split()) < 40:                 # too short
                continue
            if sum(c.isalpha() for c in t)/len(t) < 0.7:   # too many symbols
                continue
            if re.match(r'^\d+[\.\)]', t):          # numbered lists
                continue
            if re.search(r'J\s?\w+\s?\d{4}', t):    # journal citations
                continue
            if any(k in t.lower() for k in [
                "references", "acknowledg", "correspondence", "author",
                "affiliation", "email", "figure", "table", "supplementary",
                "weakness", "strength"
            ]):
                continue
            if "©" in t or "doi:" in t:
                continue

            paragraphs.append(t)
    doc.close()

    return {
        "file_name": file_name,
        "pdf_path": abs_path,
        "paragraphs": paragraphs
    }

# =========================================================
# STEP 2: AGGREGATE ALL PDFs
# =========================================================
def collect_all_paragraphs(pdf_folder):
    all_paras, meta = [], []
    pdfs = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    for pdf in tqdm(pdfs, desc="Extracting paragraphs"):
        path = os.path.join(pdf_folder, pdf)
        data = extract_clean_paragraphs(path)
        for i, p in enumerate(data["paragraphs"]):
            all_paras.append(p)
            meta.append({
                "file_name": data["file_name"],
                "pdf_path": data["pdf_path"],
                "para_id": i
            })
    print(f"✅ Extracted {len(all_paras)} clean paragraphs total.")
    return all_paras, meta

# =========================================================
# STEP 3: CHUNKING
# =========================================================
def chunk_paragraphs(paragraphs, metadata, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, meta_out = [], []
    chunk_id = 0
    for text, meta in zip(paragraphs, metadata):
        words = text.split()
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            m = meta.copy()
            m["chunk_id"] = chunk_id
            m["chunk_preview"] = chunk[:100]
            meta_out.append(m)
            chunk_id += 1
            if end == len(words):
                break
            start = end - overlap
    print(f"✅ Created {len(chunks)} chunks for embedding.")
    return chunks, meta_out

# =========================================================
# STEP 4: EMBEDDINGS + FAISS
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
# STEP 5: SAVE OUTPUTS
# =========================================================
def save_outputs(index, chunks, metadata, embeddings):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(OUTPUT_DIR, "pdf_faiss.index"))
    np.save(os.path.join(OUTPUT_DIR, "pdf_texts.npy"), np.array(chunks, dtype=object))
    np.save(os.path.join(OUTPUT_DIR, "pdf_embeddings.npy"), embeddings)
    with open(os.path.join(OUTPUT_DIR, "pdf_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"💾 Saved FAISS index and embeddings to {OUTPUT_DIR}")

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    paras, meta = collect_all_paragraphs(PDF_FOLDER)
    chunks, chunk_meta = chunk_paragraphs(paras, meta)
    index, embeddings = build_index(chunks)
    save_outputs(index, chunks, chunk_meta, embeddings)
