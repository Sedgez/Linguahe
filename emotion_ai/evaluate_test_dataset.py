from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from audio_features import extract_emotion_features_from_file


TEST_DATASET = Path("test_dataset")
OUTPUT_CSV = Path("test_results.csv")
if not TEST_DATASET.exists():
    raise FileNotFoundError(f"Test dataset folder not found: {TEST_DATASET}")

model = load_model("models/emotion_ann.keras")
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/label_encoder.pkl")
known_emotions = {str(label).strip().lower(): str(label) for label in encoder.classes_}
known_emotions["normal"] = known_emotions.get("neutral", "Neutral")

rows = []
feature_rows = []

for emotion_folder in sorted(TEST_DATASET.iterdir()):
    if not emotion_folder.is_dir():
        continue

    true_label = known_emotions.get(emotion_folder.name.lower())
    if true_label is None:
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
        feature_rows.append(extract_emotion_features_from_file(audio_path))

if not rows:
    raise ValueError("No WAV files found in test_dataset.")

probabilities = model(scaler.transform(feature_rows), training=False).numpy()

for row, prediction in zip(rows, probabilities):
    predicted_index = int(np.argmax(prediction))
    row["predicted_label"] = encoder.inverse_transform([predicted_index])[0]
    row["confidence_pct"] = round(float(prediction[predicted_index]) * 100, 2)

feature_names = [
    *(f"mfcc_{index}" for index in range(1, 14)),
    "zcr",
    "rms",
    "centroid",
    "bandwidth",
    "rolloff",
]
for row, values in zip(rows, feature_rows):
    row.update(dict(zip(feature_names, map(float, values))))

results = pd.DataFrame(rows)
results.to_csv(OUTPUT_CSV, index=False)

print(f"Created {OUTPUT_CSV} with {len(results)} test predictions.")
print(results.groupby("true_label").size())
