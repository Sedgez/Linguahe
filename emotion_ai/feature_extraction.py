import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from audio_features import extract_emotion_features_from_file

# labels.csv excludes the clips reserved in matlab_test_manifest.csv.
labels = pd.read_csv("labels.csv")

output = []

print("Extracting audio features...")

for _, row in tqdm(labels.iterrows(), total=len(labels)):

    emotion = row["emotion"].lower()
    filename = row["filename"]
    speaker = row["speaker_id"]
    path = Path("dataset") / emotion / filename

    if not path.exists():

        print("Missing:", path)

        continue

    extracted = extract_emotion_features_from_file(path)

    features = {

        "filename": filename,

        "speaker_id": speaker,
        "emotion": row["emotion"]

    }

    for i in range(13):
        features[f"mfcc_{i+1}"] = extracted[i]

    features["zcr"] = extracted[13]
    features["rms"] = extracted[14]
    features["centroid"] = extracted[15]
    features["bandwidth"] = extracted[16]
    features["rolloff"] = extracted[17]

    output.append(features)

os.makedirs(
    "features",
    exist_ok=True
)

pd.DataFrame(output).to_csv(
    "features/features.csv",
    index=False
)

print("Feature extraction completed!")
