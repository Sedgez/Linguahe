import os
import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm

labels = pd.read_csv("labels.csv")

output = []

print("Extracting audio features...")

for _, row in tqdm(labels.iterrows(), total=len(labels)):

    emotion = row["emotion"].lower()

    filename = row["filename"]

    speaker = row["speaker_id"]

    path = os.path.join(
        "dataset",
        emotion,
        filename
    )

    if not os.path.exists(path):

        print("Missing:", path)

        continue

    y, sr = librosa.load(path, sr=22050)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    mfcc = np.mean(mfcc.T, axis=0)

    zcr = np.mean(
        librosa.feature.zero_crossing_rate(y)
    )

    rms = np.mean(
        librosa.feature.rms(y=y)
    )

    centroid = np.mean(
        librosa.feature.spectral_centroid(
            y=y,
            sr=sr
        )
    )

    bandwidth = np.mean(
        librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr
        )
    )

    rolloff = np.mean(
        librosa.feature.spectral_rolloff(
            y=y,
            sr=sr
        )
    )

    features = {

        "filename": filename,

        "speaker_id": speaker,

        "emotion": row["emotion"]

    }

    for i in range(13):

        features[f"mfcc_{i+1}"] = mfcc[i]

    features["zcr"] = zcr
    features["rms"] = rms
    features["centroid"] = centroid
    features["bandwidth"] = bandwidth
    features["rolloff"] = rolloff

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