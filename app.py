#!/usr/bin/env python3
# =========================================================
# 🌐 Flask Backend for Dr. Nova – Medical Chatbot
# Author: Devvrat Shukla + GPT-5
# =========================================================

from flask import Flask, render_template, request, jsonify
from rag_query import rag_query

app = Flask(__name__)

# In-memory chat history
chat_history = []


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("message", "").strip()

        if not user_input:
            return jsonify({"response": "Please type something so I can help you."})

        chat_history.append({"role": "user", "content": user_input})

        # Generate answer via RAG pipeline
        result = rag_query(user_input, retrieval_method="hybrid", top_k=3)
        response = result.get("answer", "⚠️ Sorry, I couldn’t find a reliable answer right now.")

        chat_history.append({"role": "bot", "content": response})

        return jsonify({"response": response})

    except Exception as e:
        print("❌ Chat route error:", e)
        return jsonify({"response": "⚠️ Internal server error. Please try again."})


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
