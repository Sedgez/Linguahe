import joblib
import torch
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer
from faster_whisper import WhisperModel

# -----------------------------
# DEVICE INITIALIZATION
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

print(f"\n=== SYSTEM DEVICE: {device.upper()} ({compute_type}) ===")


# -----------------------------
# ACCENT ANN MODEL
# -----------------------------
print("--- Loading Accent ANN Models ---")
try:
    accent_model = load_model("accent_ai/models/accent_ann.keras")
    accent_scaler = joblib.load("accent_ai/models/scaler.pkl")
    accent_encoder = joblib.load("accent_ai/models/label_encoder.pkl")
    print("Accent ANN Loaded Successfully")
except Exception as e:
    print(f"Warning: Failed to load Accent ANN models: {e}")
    accent_model, accent_scaler, accent_encoder = None, None, None


# -----------------------------
# EMOTION ANN MODEL
# -----------------------------
print("--- Loading Emotion ANN Models ---")
try:
    emotion_model = load_model("emotion_ai/models/emotion_ann.keras")
    emotion_scaler = joblib.load("emotion_ai/models/scaler.pkl")
    emotion_encoder = joblib.load("emotion_ai/models/label_encoder.pkl")
    print("Emotion ANN Loaded Successfully")
except Exception as e:
    print(f"Warning: Failed to load Emotion ANN models: {e}")
    emotion_model, emotion_scaler, emotion_encoder = None, None, None


# -----------------------------
# FASTER-WHISPER MODEL
# -----------------------------
print("--- Initializing Faster-Whisper Model ---")
whisper_model = None
try:
    whisper_model = WhisperModel("turbo", device=device, compute_type=compute_type)
    print("Loaded Faster-Whisper Turbo Model")
except Exception as e:
    print(f"Turbo unavailable -> fallback to medium: {e}")
    try:
        whisper_model = WhisperModel("medium", device=device, compute_type=compute_type)
        print("Loaded Faster-Whisper Medium Model")
    except Exception as fallback_e:
        print(f"Warning: Failed to load Faster-Whisper fallback: {fallback_e}")


# -----------------------------
# SBERT MODEL
# -----------------------------
print("--- Loading Multilingual SBERT Model ---")
try:
    sbert_model = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2", 
        device=device
    )
    print("SBERT Model Loaded Successfully")
except Exception as e:
    print(f"Warning: Failed to load SBERT model: {e}")
    sbert_model = None