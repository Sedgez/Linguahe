import joblib
import torch
import whisper

from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer


# -----------------------------
# DEVICE INITIALIZATION
# -----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n=== SYSTEM DEVICE: {device.upper()} ===")


# -----------------------------
# ACCENT ANN MODEL
# -----------------------------

print("--- Loading Accent ANN Models ---")

try:
    accent_model = load_model(
        "accent_ai/models/accent_ann.keras"
    )

    accent_scaler = joblib.load(
        "accent_ai/models/scaler.pkl"
    )

    accent_encoder = joblib.load(
        "accent_ai/models/label_encoder.pkl"
    )

    print("Accent ANN Loaded Successfully")

except Exception as e:

    print(f"Warning: Failed to load Accent ANN models: {e}")

    accent_model = None
    accent_scaler = None
    accent_encoder = None


# -----------------------------
# EMOTION ANN MODEL
# -----------------------------

print("--- Loading Emotion ANN Models ---")

try:
    emotion_model = load_model(
        "emotion_ai/models/emotion_ann.keras"
    )

    emotion_scaler = joblib.load(
        "emotion_ai/models/scaler.pkl"
    )

    emotion_encoder = joblib.load(
        "emotion_ai/models/label_encoder.pkl"
    )

    print("Emotion ANN Loaded Successfully")

except Exception as e:

    print(f"Warning: Failed to load Emotion ANN models: {e}")

    emotion_model = None
    emotion_scaler = None
    emotion_encoder = None


# -----------------------------
# WHISPER MODEL
# -----------------------------

print("--- Initializing Whisper Model ---")

try:
    whisper_model = whisper.load_model(
        "turbo",
        device=device
    )

    print("Loaded Whisper Turbo Model")

except Exception as e:

    print(f"Turbo unavailable -> fallback to medium: {e}")

    whisper_model = whisper.load_model(
        "medium",
        device=device
    )


# -----------------------------
# SBERT MODEL
# -----------------------------

print("--- Loading Multilingual SBERT Model ---")

sbert_model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2",
    device=device
)

print("SBERT Model Loaded Successfully")