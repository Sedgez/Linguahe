from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from audio_features import extract_emotion_features_from_file


ROOT = Path(__file__).resolve().parent
TEST_DATASET = ROOT / "test_dataset"
OUTPUT_CSV = ROOT / "test_results.csv"
MATLAB_OUTPUT_CSV = ROOT / "matlab" / "emotion_test_results.csv"
MANIFEST = ROOT / "matlab_test_manifest.csv"
if not TEST_DATASET.exists():
    raise FileNotFoundError(f"Test dataset folder not found: {TEST_DATASET}")

model = load_model(ROOT / "models" / "emotion_ann_v2.keras")
scaler = joblib.load(ROOT / "models" / "scaler_v2.pkl")
encoder = joblib.load(ROOT / "models" / "label_encoder_v2.pkl")
known_emotions = {str(label).strip().lower(): str(label) for label in encoder.classes_}
known_emotions["normal"] = known_emotions.get("neutral", "Neutral")

manifest = pd.read_csv(MANIFEST)
rows = []
feature_rows = []

for record in manifest.itertuples(index=False):
    emotion_folder = TEST_DATASET / str(record.emotion).strip().lower()
    requested_name = str(record.filename)
    audio_path = emotion_folder / requested_name
    if not audio_path.is_file():
        heldout_path = emotion_folder / f"holdout_{requested_name}"
        if heldout_path.is_file():
            audio_path = heldout_path
        else:
            matches = [
                path for path in emotion_folder.iterdir()
                if path.is_file() and path.name.lower() == requested_name.lower()
            ]
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected exactly one manifest audio file for {requested_name} "
                    f"in {emotion_folder}; found {len(matches)}."
                )
            audio_path = matches[0]

    true_label = known_emotions.get(str(record.emotion).strip().lower())
    if true_label is None:
        raise ValueError(f"Unknown emotion in manifest: {record.emotion}")

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
MATLAB_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(MATLAB_OUTPUT_CSV, index=False)

print(f"Created {OUTPUT_CSV} with {len(results)} test predictions.")
print(f"Created {MATLAB_OUTPUT_CSV} with {len(results)} test predictions.")
print(f"Single-label accuracy: {(results['predicted_label'] == results['true_label']).mean():.2%}")
print(results.groupby("true_label").size())
