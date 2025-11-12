#!/usr/bin/env python3
# =========================================================
# 🌐 Dr. Nova – Multilingual Flask Medical Chatbot (Local-only)
# Updated version: frontend auto-translates user input to English
# =========================================================
from flask import Flask, render_template, request, jsonify
from rag_query import rag_query
from translator_local import LocalTranslator
import os, re, time, json

app = Flask(__name__)
chat_history = []

# Default language and glossary setup
DEFAULT_TARGET_LANG = os.getenv("DEFAULT_TARGET_LANG", "EN")
DEFAULT_GLOSSARY = {
    "tb": "tuberculosis",
    "5 fu": "5-fluorouracil",
    "bp": "blood pressure",
    "chemo": "chemotherapy",
    "rt": "radiotherapy",
}

translator = LocalTranslator(glossary=DEFAULT_GLOSSARY, round_trip_check=False)

UI_LANG_CODES = {
    "English": "EN", "Hindi": "HI", "Bengali": "BN", "Gujarati": "GU",
    "Kannada": "KN", "Malayalam": "ML", "Marathi": "MR", "Punjabi": "PA",
    "Tamil": "TA", "Telugu": "TE", "Urdu": "UR", "Odia": "OR", "Assamese": "AS",
}

def to_lang_code(label_or_code: str) -> str:
    if not label_or_code:
        return DEFAULT_TARGET_LANG
    label_or_code = label_or_code.strip()
    return UI_LANG_CODES.get(label_or_code, label_or_code.upper())

@app.route("/")
def home():
    try:
        return render_template("index.html")
    except Exception:
        return "Dr. Nova Multilingual Chatbot (Local-only) is running."

@app.route("/chat", methods=["POST"])
def chat():
    global chat_history
    try:
        data = request.get_json(force=True, silent=True) or {}
        user_input = (data.get("message") or "").strip()
        user_lang_raw = data.get("language", "English")
        user_lang = to_lang_code(user_lang_raw)

        if not user_input:
            return jsonify({"response": "Please type a medical question."})

        chat_history.append({"role": "user", "content": user_input, "lang": "EN"})  # now always stored in English

        # 1️⃣ User input is already in English from frontend
        text_en = user_input

        # 2️⃣ Handle small talk
        if re.match(r'^(hi|hello|hey|thanks|thank you|how are you)\b', text_en.lower()):
            english_reply = "Hi! I'm Dr. Nova. How can I help you with a cancer-related question today?"
            result = {"safety_warning": False, "metadata": []}
        else:
            recent_context = "\n".join(f"{m['role']}: {m['content']}" for m in chat_history[-4:])
            result = rag_query(text_en, top_k=4, chat_context=recent_context)
            english_reply = result.get("answer", "I couldn't find enough info. Please consult your clinician.")

        # 3️⃣ Translate reply back to user’s selected language
        try:
            tr_out = translator.translate(english_reply, target_lang=user_lang)
            final_reply = tr_out.translated_text
        except Exception:
            final_reply = english_reply

        chat_history.append({"role": "assistant", "content": final_reply, "lang": user_lang})

        # 🧠 Save only the latest exchange
        SAVE_PATH = "chats.json"
        if len(chat_history) >= 2:
            new_exchange = chat_history[-2:]
        else:
            new_exchange = chat_history

        save_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "conversation": new_exchange
        }

        try:
            if os.path.exists(SAVE_PATH):
                with open(SAVE_PATH, "r", encoding="utf-8") as f:
                    all_chats = json.load(f)
            else:
                all_chats = []

            if not all_chats or all_chats[-1]["conversation"] != save_data["conversation"]:
                all_chats.append(save_data)

            with open(SAVE_PATH, "w", encoding="utf-8") as f:
                json.dump(all_chats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("⚠️ Failed to save chat:", e)

        if len(chat_history) > 12:
            chat_history = chat_history[-8:]

        return jsonify({
            "response": final_reply,
            "safety_warning": bool(result.get("safety_warning", False)),
            "sources_count": len(result.get("metadata", []))
        })

    except Exception as e:
        print("❌ Chat error:", e)
        return jsonify({
            "response": "An internal error occurred. Please try again.",
            "safety_warning": True,
            "sources_count": 0
        }), 500

@app.route("/clear", methods=["POST"])
def clear_chat():
    global chat_history
    chat_history = []
    return jsonify({"status": "Chat history cleared"})

@app.route("/health")
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "Dr. Nova Multilingual Chatbot (Local-only)",
        "chat_history_length": len(chat_history)
    })

if __name__ == "__main__":
    print("🩺 Dr. Nova Medical Chatbot (Local-only) Starting...")
    print("🌐 http://127.0.0.1:5000")
    print("💾 Chats are saved in chats.json after every reply (deduplicated).")
    app.run(debug=True, host="127.0.0.1", port=5000)







