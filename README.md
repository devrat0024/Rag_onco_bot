# 🩺 Dr. Nova – Multilingual Medical Oncology Chatbot

**Dr. Nova** is an advanced, Retrieval-Augmented Generation (RAG) based medical chatbot specializing in oncology. Designed for both accessibility and accuracy, it processes complex medical documents (PDFs and Q&A JSONs) to provide precise, bite-sized answers. The application natively supports **multilingual interactions**, automatically translating non-English queries to English for internal processing, and translating the generated responses back to the user's preferred language.

---

## ✨ Key Features

1. **Retrieval-Augmented Generation (RAG) Engine**
   - Combines the power of **FAISS** (dense vector search) and **BM25Okapi** (sparse keyword search) to retrieve the most relevant medical context from embedded documents.
   - Utilizes `pritamdeka/S-BioBert-snli-multinli-stsb` as the embedding model, specifically tuned for biomedical and clinical text.

2. **Multilingual Capabilities**
   - Supports 13+ languages including Hindi, Bengali, Tamil, Telugu, Urdu, and more.
   - User inputs are automatically translated to English before information retrieval.
   - The final medical advice generated is translated back to the user's selected language using a specialized local translator (`translator_local.py`) with a custom medical glossary.

3. **Hybrid Answer Generation**
   - **Primary**: Connects to the **Groq API** (`llama-3.3-70b-versatile`) for ultra-fast, high-quality, and empathetic conversational replies.
   - **Local Fallback**: If the API is unavailable, the system safely falls back to a locally hosted `google/flan-t5-large` model to ensure zero downtime.

4. **Data Pipeline & PDF Extraction**
   - Extracts structured textual paragraphs from medical PDFs.
   - Processes Q&A JSON datasets.
   - Truncates and safely manages context size to prevent token overflow.

5. **Conversational Memory & State**
   - Flask-based web server (`app.py`) maintains a rolling recent chat history to understand context across follow-up questions.
   - Automatically saves and deduplicates chat logs locally into `chats.json`.

---

## 🏗️ Architecture & Modules

- `app.py` / `server.py`: The Flask web servers handling routing, multilingual translations, and user-facing endpoints.
- `rag_pipe_embedding.py`: The data-ingestion pipeline. Reads JSONs/PDFs, chunks the data, and builds the FAISS index and embeddings.
- `rag_query.py`: The inference engine. Handles context retrieval using BM25 and FAISS, then generates an answer using Groq or a local LLM.
- `pdf_extractor.py` / `pdf_json.py`: PDF processing scripts to cleanly convert documents into chunkable JSONs.
- `translator.py` / `translator_local.py`: Specialized translation modules handling the multilingual aspects with a built-in medical glossary (e.g., `tb` -> `tuberculosis`).

---

## 🚀 Setup & Installation

### 1. Install Dependencies
Ensure you have Python 3.8+ installed. Install the required libraries using:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```
*(If the key is omitted, the application will automatically run in local-fallback mode).*

### 3. Build the Vector Database
Before querying, you must generate the embeddings from your unstructured documents. Place your JSON files in `jsons_4` and run:
```bash
python rag_pipe_embedding.py
```
This will create an `embeddings_biomed/` directory containing the `faiss_index.index` and metadata.

### 4. Run the Web Application
Start the Flask server:
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000` to start chatting with Dr. Nova!

---

## 🛡️ Disclaimer
*Dr. Nova is designed as an informational tool and should not be used as a replacement for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for any health concerns.*
