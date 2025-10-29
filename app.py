# =========================================================
# 🧠 Medical RAG Streamlit App (Improved Response Formatting)
# =========================================================
import streamlit as st
import traceback
from rag import retrieve_context, rag_query, OUTPUT_DIR

# =========================================================
# 🔧 Streamlit Page Setup
# =========================================================
st.set_page_config(page_title="🏥 Medical RAG Assistant", layout="wide")
st.title("🏥 **Biomedical RAG Assistant**")
st.markdown("Ask precise questions based on your extracted medical PDFs.")

# =========================================================
# 🧠 Sidebar Controls
# =========================================================
st.sidebar.header("🔧 Search Configuration")

search_mode = st.sidebar.radio(
    "Search Mode:",
    ["Semantic Search", "Hybrid Search", "Reranked Search"]
)

top_k = st.sidebar.slider("Top-K Results:", 1, 10, 5)
st.sidebar.markdown("---")
st.sidebar.caption("⚙️ Using FAISS + SentenceTransformers backend")

# =========================================================
# 💬 Query Input
# =========================================================
query = st.text_input(
    "Enter your medical question:",
    placeholder="e.g., What are the early symptoms of lung cancer?"
)

# =========================================================
# 🚀 Main Execution
# =========================================================
if st.button("🔍 Search & Generate Answer"):
    if not query.strip():
        st.warning("Please enter a valid medical question.")
        st.stop()

    try:
        # Map UI selection to backend retrieval method
        method = (
            "semantic"
            if search_mode == "Semantic Search"
            else "reranked"
            if search_mode == "Reranked Search"
            else "hybrid"
        )

        # -----------------------------------------------------
        # STEP 1: Retrieve Context
        # -----------------------------------------------------
        with st.spinner("📚 Retrieving relevant context..."):
            context_chunks, metadata = retrieve_context(query, method=method, top_k=top_k)

        if not context_chunks:
            st.error("⚠️ No relevant context found. Try rebuilding your FAISS index.")
            st.stop()

        # Display retrieved context nicely
        st.subheader("📄 Retrieved Medical Context")
        for i, (chunk, meta) in enumerate(zip(context_chunks, metadata), start=1):
            with st.expander(f"🧩 Context {i}: {meta['source']} (Page {meta.get('page_number', 'N/A')})"):
                st.write(chunk[:1000] + ("..." if len(chunk) > 1000 else ""))

        # -----------------------------------------------------
        # STEP 2: Generate Answer
        # -----------------------------------------------------
        with st.spinner("🧠 Generating biomedical answer..."):
            result = rag_query(query, retrieval_method=method, top_k=top_k, verbose=False)

        # Validate structure
        if not isinstance(result, dict) or "answer" not in result:
            raise ValueError("Unexpected response format from RAG backend.")

        # -----------------------------------------------------
        # STEP 3: Display Answer
        # -----------------------------------------------------
        st.subheader("🩺 Final Biomedical Answer")
        answer = result["answer"].strip() or "⚠️ No valid answer generated."
        st.success(answer)

        # -----------------------------------------------------
        # STEP 4: Display Sources
        # -----------------------------------------------------
        st.markdown("### 🔗 **Sources Used**")
        for i, meta in enumerate(metadata, 1):
            st.markdown(f"**[{i}]** {meta['source']} — Page {meta.get('page_number', 'N/A')}")

    except Exception as e:
        st.error("❌ Error while generating answer. Check backend logs.")
        with st.expander("Show error details"):
            st.code(traceback.format_exc())

# =========================================================
# 🧾 Footer
# =========================================================
st.markdown("---")
st.caption("Built with 🧬 Biomedical RAG | FAISS + Transformers + Streamlit")
