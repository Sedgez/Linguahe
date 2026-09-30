from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


TEST_DATASET = Path("test_dataset")
OUTPUT_CSV = Path("test_results.csv")
VALID_EMOTIONS = {"angry", "happy", "normal", "sad"}


def extract_features(audio_path):
    signal, sample_rate = librosa.load(audio_path, sr=22050)

    mfcc = np.mean(
        librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13).T,
        axis=0,
    )

    return [
        *mfcc,
        np.mean(librosa.feature.zero_crossing_rate(signal)),
        np.mean(librosa.feature.rms(y=signal)),
        np.mean(librosa.feature.spectral_centroid(y=signal, sr=sample_rate)),
        np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate)),
        np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sample_rate)),
    ]


if not TEST_DATASET.exists():
    raise FileNotFoundError(f"Test dataset folder not found: {TEST_DATASET}")

model = load_model("models/emotion_ann.keras")
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/label_encoder.pkl")

rows = []
feature_rows = []

for emotion_folder in sorted(TEST_DATASET.iterdir()):
    if not emotion_folder.is_dir():
        continue

    true_label = emotion_folder.name.capitalize()
    if emotion_folder.name.lower() not in VALID_EMOTIONS:
        print(f"Skipping unknown emotion folder: {emotion_folder}")
        continue

    for audio_path in sorted(emotion_folder.iterdir()):
        if not audio_path.is_file() or audio_path.suffix.lower() != ".wav":
            continue

        rows.append({
            "category": "Emotion",
            "filename": audio_path.name,
            "true_label": true_label,
        })
        feature_rows.append(extract_features(audio_path))

if not rows:
    raise ValueError("No WAV files found in test_dataset.")

probabilities = model(scaler.transform(feature_rows), training=False).numpy()

for row, prediction in zip(rows, probabilities):
    predicted_index = int(np.argmax(prediction))
    row["predicted_label"] = encoder.inverse_transform([predicted_index])[0]
    row["confidence_pct"] = round(float(prediction[predicted_index]) * 100, 2)

results = pd.DataFrame(rows)
results.to_csv(OUTPUT_CSV, index=False)

print(f"Created {OUTPUT_CSV} with {len(results)} test predictions.")
print(results.groupby("true_label").size())
