#!/usr/bin/env python3
# =========================================================
# 🌐 Dr. Nova – Flask Backend (Improved Medical Chatbot)
# =========================================================

from flask import Flask, render_template, request, jsonify
from rag_query import rag_query
import re

app = Flask(__name__)
chat_history = []  # Temporary in-memory conversation history

# =========================================================
# 🏠 Home Route
# =========================================================
@app.route("/")
def home():
    return render_template("index.html")

# =========================================================
# 💬 Chat Route
# =========================================================
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("message", "").strip()

        if not user_input:
            return jsonify({"response": "Please type a medical question."})

        # Add user message to memory
        chat_history.append({"role": "user", "content": user_input})

        # Handle greetings or small talk directly (no RAG)
        if re.match(r'^(hi|hello|hey|thanks|thank you|how are you)\b', user_input.lower()):
            response = (
                "Hi there! 👋 I’m Dr. Nova — your medical assistant. "
                "How can I help you with a cancer-related question today?"
            )
            chat_history.append({"role": "assistant", "content": response})
            return jsonify({
                "response": response,
                "safety_warning": False,
                "sources_count": 0
            })

        # Build context from recent messages
        recent_context = "\n".join(
            f"{m['role']}: {m['content']}" for m in chat_history[-4:]
        )

        # Generate medical RAG response
        result = rag_query(user_input, top_k=4, chat_context=recent_context)
        response = result.get(
            "answer",
            "I couldn't find relevant medical information. Please consult your healthcare provider."
        )

        # Add bot reply to memory
        chat_history.append({"role": "assistant", "content": response})

        # Keep chat history short (last 10 messages)
        if len(chat_history) > 10:
            chat_history.pop(0)

        return jsonify({
            "response": response,
            "safety_warning": result.get("safety_warning", False),
            "sources_count": len(result.get("metadata", []))
        })

    except Exception as e:
        print("❌ Chat route error:", e)
        return jsonify({
            "response": (
                "I'm experiencing technical difficulties. Please try again later "
                "or consult your doctor for urgent medical questions."
            ),
            "safety_warning": True,
            "sources_count": 0
        })

# =========================================================
# 🧹 Clear Chat History
# =========================================================
@app.route("/clear", methods=["POST"])
def clear_chat():
    global chat_history
    chat_history = []
    return jsonify({"status": "Chat history cleared"})

# =========================================================
# ℹ️ Chat Info
# =========================================================
@app.route("/info")
def chat_info():
    return jsonify({
        "total_messages": len(chat_history),
        "recent_context": chat_history[-3:] if chat_history else []
    })

# =========================================================
# 🩺 Health Check
# =========================================================
@app.route("/health")
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "Dr. Nova Medical Chatbot",
        "chat_history_length": len(chat_history)
    })

# =========================================================
# 🚀 Run Server
# =========================================================
if __name__ == "__main__":
    print("🩺 Dr. Nova Medical Chatbot Starting...")
    print("🌐 Server running on http://127.0.0.1:5000")
    print("💬 Medical RAG system ready for queries")
    app.run(debug=True, host="127.0.0.1", port=5000)
