import os
import json
import time
import tempfile
from datetime import datetime
import streamlit as st
import pandas as pd

# local whisper and audio helpers
try:
    import whisper
except Exception:
    whisper = None
from pydub import AudioSegment

DB_FILE = "call_history.json"

def load_db():
    if not os.path.exists(DB_FILE):
        return []
    with open(DB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_db(data):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def convert_to_wav(infile_path, out_path):
    """Convert many formats to wav using pydub (ffmpeg required)."""
    audio = AudioSegment.from_file(infile_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(out_path, format="wav")
    return out_path

# load whisper model lazily (so app starts faster)
_whisper_model = None
def get_whisper_model(name="tiny"):
    global _whisper_model
    if _whisper_model is None:
        if whisper is None:
            raise RuntimeError("whisper package not installed. pip install openai-whisper")
        _whisper_model = whisper.load_model(name)
    return _whisper_model

def transcribe_local(file_path):
    """Transcribe using local whisper model"""
    try:
        model = get_whisper_model("tiny")   # tiny is fast; change to 'base' for better accuracy
        result = model.transcribe(file_path, language=None)  # auto detect language
        return result.get("text", "")
    except Exception as e:
        return f"Local transcription failed: {e}"

def extract_intent_simple(text):
    """Very small local heuristic to mark intent/urgency/summary when offline"""
    txt = (text or "").lower()
    if any(w in txt for w in ["appointment", "book", "schedule", "reschedule"]):
        intent = "appointment"
    elif any(w in txt for w in ["price", "cost", "enquiry", "question", "how much", "where"]):
        intent = "enquiry"
    elif any(w in txt for w in ["not working", "complain", "issue", "problem", "broken"]):
        intent = "complaint"
    elif any(w in txt for w in ["love", "like", "great", "thank you", "thanks"]):
        intent = "feedback"
    else:
        intent = "other"

    if any(w in txt for w in ["urgent", "asap", "immediately", "right now"]):
        urgency = "high"
    elif any(w in txt for w in ["soon", "within", "tomorrow", "today"]):
        urgency = "medium"
    else:
        urgency = "low"

    summary = (text.strip().replace("\n", " ")[:250]) if text else ""
    recommended_action = "follow_up"
    return {"intent": intent, "urgency": urgency, "summary": summary, "recommended_action": recommended_action}

# Streamlit UI
st.set_page_config(page_title="Reception Voice Agent (local)", layout="wide")
st.title("Reception Voice Agent — Local Whisper (no OpenAI required)")

uploaded_file = st.file_uploader("Upload audio (wav/mp3/m4a/ogg/mp4)", type=["wav", "mp3", "m4a", "ogg", "mp4"])
caller_name = st.text_input("Caller Name (optional)")

if st.button("Process Audio") and uploaded_file:
    # save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tf:
        tf.write(uploaded_file.read())
        temp_in = tf.name

    st.info("Converting to WAV (if needed)...")
    wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    try:
        convert_to_wav(temp_in, wav_path)
    except Exception as e:
        st.error(f"Conversion failed: {e}")
        wav_path = temp_in  # try raw file

    st.info("Transcribing locally with Whisper...")
    transcript = transcribe_local(wav_path)
    st.subheader("Transcript")
    st.write(transcript)

    st.info("Extracting metadata...")
    meta = extract_intent_simple(transcript)
    st.subheader("Metadata")
    st.json(meta)

    record = {
        "caller_name": caller_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "transcript": transcript,
        "intent": meta["intent"],
        "urgency": meta["urgency"],
        "summary": meta["summary"],
        "recommended_action": meta["recommended_action"]
    }

    data = load_db()
    data.append(record)
    save_db(data)
    st.success("Saved to local JSON database!")

# show history
st.header("Call History")
history = load_db()
if history:
    df = pd.DataFrame(history)
    st.dataframe(df)
else:
    st.info("No call records yet.")
