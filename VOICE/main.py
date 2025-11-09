import os
import google.generativeai as gen
from RealtimeSTT.RealtimeSTT import AudioToTextRecorder
from Modules.Vitals.vitals import get_vitals,get_hr,read_temp_c_float

# ========== GEMINI SETUP ==========
gen.configure(api_key="AIzaSyCuy5O29H5aYZ5r6wuyX86X-gNfZg_O69k")

intent_model = gen.GenerativeModel("gemini-2.5-flash")
chat_model   = gen.GenerativeModel("gemini-2.5-flash")

# persistent chat session for general conversation
chat_session = chat_model.start_chat(history=[
    {"role": "user", "parts": ["You are MedBot, a friendly robot assistant. Be concise."]}
])

# ========== YOUR FUNCTIONS ==========
def check_patient_vitals():
    print("[ROBOT] Checking patient vitals (HR, BP, SpO2, RR)...")

def bring_water():
    print("[ROBOT] Bringing water to the patient...")

def call_nurse():
    print("[ROBOT] Calling the nurse to this room...")

# ========== YOUR INDEX DICTIONARY ==========
indexes = {
    "check_patient_vitals": {
        "fn": get_vitals,
        "description": "Check the patient's vitals (heart rate, blood pressure, oxygen saturation, respiratory rate).",
    },
    "check_heartrate":{
        "fn":get_hr,
        "description": "Check the patient`s HEARTRATE AND SPO2"
    },
    "check_temperature":{
        "fn":read_temp_c_float,
        "description": "Check the patient`s Temprature"
    },

    "bring_water": {
        "fn": bring_water,
        "description": "Bring water to the patient.",
    },
    "call_nurse": {
        "fn": call_nurse,
        "description": "Call or alert the nurse / medical staff.",
    },
}

ACTION_LIST = "\n".join(
    f"- {key}: {value['description']}"
    for key, value in indexes.items()
)

# ========== INTENT ROUTER (LLM RETURNS A KEY OR NONE) ==========
def route_to_index_key(user_command: str) -> str | None:
    """
    Ask Gemini which index key (if any) matches this command.
    Returns the key string or None.
    """
    prompt = f"""
You are an intent router for a medical robot assistant.

You are given a user's instruction.
Choose EXACTLY ONE index key from the list below,
or return NONE if no index is appropriate.

Available index keys:
{ACTION_LIST}

Rules:
- Respond with ONLY the key (e.g., check_patient_vitals), with no extra words.
- If the user is greeting, chatting, asking general questions, or the command
  doesn't correspond clearly to any index key, respond with: NONE
- Do NOT invent new keys.

User instruction:
\"\"\"{user_command}\"\"\"
""".strip()

    resp = intent_model.generate_content(prompt)
    key = resp.text.strip()

    if key in indexes:
        return key
    if key.upper() == "NONE":
        return None
    return None  # safety fallback

# ========== GENERAL CHAT (WHEN NO ACTION KEY) ==========
def chat_reply(user_text: str):
    """
    Normal ChatGPT-style answer, streamed.
    """
    print("MedBot: ", end="", flush=True)
    stream = chat_session.send_message(user_text, stream=True)
    for chunk in stream:
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print()

# ========== PROCESS TEXT FROM STT (NO WAKE WORD) ==========
def process_text(text: str):
    """
    Called continuously by AudioToTextRecorder.
    For every non-empty text:
      1. Ask Gemini which index key (if any) matches.
      2. If a key is returned → call indexes[key]['fn']().
      3. If NONE → general chat response.
    """
    if not text or not text.strip():
        return

    print(f"\n[USER]: {text}")

    key = route_to_index_key(text)
    if key is not None:
        print(f"[SYSTEM] Intent matched index key: {key}")
        indexes[key]["fn"]()
    else:
        print("[SYSTEM] No matching index key; responding as chat.")
        chat_reply(text)

# ========== MAIN LOOP ==========
if __name__ == "__main__":
    print("Wait until it says 'speak now'")
    recorder = AudioToTextRecorder()

    while True:
        recorder.text(process_text)