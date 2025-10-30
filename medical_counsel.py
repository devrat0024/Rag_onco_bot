#!/usr/bin/env python3
# =========================================================
# 🩺 Dr. Nova – Cancer Counselling Brain
# Conversational flow engine for Lung, Breast, and Oral cancer
# Author: Devvrat Shukla + GPT-5
# =========================================================

import json
from rag_query import rag_query

# =========================================================
# 💾 Conversation State Management
# =========================================================
def load_state():
    try:
        with open("conversation_state.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"last_topic": None, "symptoms": [], "cancer_type": None}

def save_state(state):
    with open("conversation_state.json", "w") as f:
        json.dump(state, f, indent=2)

# =========================================================
# 🧠 Keyword Definitions for Cancer Type Detection
# =========================================================
CANCER_KEYWORDS = {
    "breast": ["breast", "lump", "nipple", "pain", "swelling", "discharge"],
    "lung": ["cough", "breathing", "chest", "wheezing", "mucus", "blood"],
    "oral": ["mouth", "ulcer", "sore", "tongue", "gum", "jaw", "bleeding"],
}

# =========================================================
# 💬 Main Conversation Logic
# =========================================================
def dr_nova_response(user_input: str):
    """Main conversational flow for Dr. Nova."""
    user_input = user_input.strip().lower()
    state = load_state()

    # -----------------------------------------------------
    # Detect cancer type (only once if not already set)
    # -----------------------------------------------------
    if not state.get("cancer_type"):
        for cancer, keywords in CANCER_KEYWORDS.items():
            if any(word in user_input for word in keywords):
                state["cancer_type"] = cancer
                break

    if not state["cancer_type"]:
        reply = (
            "Hi! I'm Dr. Nova, your medical assistant. Can you tell me what symptoms you're noticing? "
            "(like cough, breast lump, or mouth sore)"
        )
        save_state(state)
        return reply

    cancer = state["cancer_type"]

    # =====================================================
    # 🩷 BREAST CANCER COUNSELLING FLOW
    # =====================================================
    if cancer == "breast":
        if "lump" in user_input:
            reply = "I see. Is the lump painful or painless? Has it changed in size recently?"
        elif "pain" in user_input:
            reply = "Pain can have many causes, not always cancer. Have you noticed any nipple discharge or skin dimpling?"
        elif "nipple" in user_input or "discharge" in user_input:
            reply = "Thank you for sharing. It’s a good idea to have an ultrasound or mammogram soon. Would you like me to explain what that involves?"
        else:
            result = rag_query(user_input)
            reply = result["answer"]

    # =====================================================
    # 🫁 LUNG CANCER COUNSELLING FLOW
    # =====================================================
    elif cancer == "lung":
        if "cough" in user_input:
            reply = "How long have you had the cough? Is it dry or producing mucus?"
        elif "mucus" in user_input or "blood" in user_input:
            reply = "Thanks for letting me know. Have you also experienced chest pain or shortness of breath?"
        elif "breath" in user_input or "chest" in user_input:
            reply = "That’s important to mention. It might be good to get a chest X-ray or CT scan. Would you like me to tell you what tests are usually done?"
        else:
            result = rag_query(user_input)
            reply = result["answer"]

    # =====================================================
    # 👄 ORAL CANCER COUNSELLING FLOW
    # =====================================================
    elif cancer == "oral":
        if "sore" in user_input or "ulcer" in user_input:
            reply = "Okay, how long has the sore been there? Does it hurt when you eat or talk?"
        elif "bleeding" in user_input or "gum" in user_input:
            reply = "Bleeding in the mouth can have many causes. Have you noticed any white or red patches or difficulty swallowing?"
        elif "patch" in user_input or "tongue" in user_input:
            reply = "Thank you for sharing. Persistent sores or patches lasting more than two weeks should be checked by a dentist or oral surgeon."
        else:
            result = rag_query(user_input)
            reply = result["answer"]

    # Save conversation context
    save_state(state)
    return reply
