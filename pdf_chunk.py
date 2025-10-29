# =====================================
# 📄 Step 3. Build the Vector Index
# =====================================
import json, os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def load_json_texts(folder_path):
    texts = []
    for f in os.listdir(folder_path):
        if f.endswith(".json"):
            with open(os.path.join(folder_path, f), 'r', encoding='utf-8') as jf:
                data = json.load(jf)
                # The JSON files contain a list of strings, not a dictionary.
                if isinstance(data, list):
                    texts.extend(data)
    return texts

folder = "data_line_level"  # folder with your extracted JSON files
texts = load_json_texts(folder)
print(f"Loaded {len(texts)} sentences.")

# Convert to embeddings
embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "pdf_knowledge.index")
np.save("pdf_texts.npy", np.array(texts))
print("✅ Vector index built and saved.")

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load the open-source LLM
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load the index + data
index = faiss.read_index("pdf_knowledge.index")
texts = np.load("pdf_texts.npy", allow_pickle=True)

def retrieve_context(query, top_k=5):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    results = [texts[i] for i in indices[0]]
    return "\n".join(results)

def ask_chatbot(query):
    context = retrieve_context(query)
    prompt = f"Use the following context to answer:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    response = generator(prompt, max_new_tokens=250, temperature=0.3, do_sample=False)
    return response[0]['generated_text'].split("Answer:")[-1].strip()