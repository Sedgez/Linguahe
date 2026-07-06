import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

labels = pd.read_csv("labels.csv")

all_features = []

print("Extracting features...\n")

for _, row in tqdm(labels.iterrows(), total=len(labels)):

    accent = row["accent"].lower()

    filename = row["filename"]

    audio_path = os.path.join(
        "dataset",
        accent,
        filename
    )

    if not os.path.exists(audio_path):
        print("Missing:", audio_path)
        continue

    try:

        signal, sr = librosa.load(
            audio_path,
            sr=22050
        )

        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=13
        )

        mfcc = np.mean(
            mfcc.T,
            axis=0
        )

        zcr = np.mean(
            librosa.feature.zero_crossing_rate(signal)
        )

        rms = np.mean(
            librosa.feature.rms(y=signal)
        )

        centroid = np.mean(
            librosa.feature.spectral_centroid(
                y=signal,
                sr=sr
            )
        )

        bandwidth = np.mean(
            librosa.feature.spectral_bandwidth(
                y=signal,
                sr=sr
            )
        )

        rolloff = np.mean(
            librosa.feature.spectral_rolloff(
                y=signal,
                sr=sr
            )
        )

        row_data = {
            "filename": filename,
            "speaker_id": row["speaker_id"],
            "accent": row["accent"]
        }

        for i in range(13):
            row_data[f"mfcc_{i+1}"] = mfcc[i]

        row_data["zcr"] = zcr
        row_data["rms"] = rms
        row_data["centroid"] = centroid
        row_data["bandwidth"] = bandwidth
        row_data["rolloff"] = rolloff

        all_features.append(row_data)

    except Exception as e:

        print(e)

df = pd.DataFrame(all_features)

os.makedirs(
    "features",
    exist_ok=True
)

df.to_csv(
    "features/features.csv",
    index=False
)

print("\nFinished!")
print(df.head())