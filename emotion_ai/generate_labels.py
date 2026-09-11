import os
import csv

DATASET_FOLDER = "dataset"

EMOTIONS = [
    "happy",
    "sad",
    "angry",
    "normal"
]

with open("labels.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "filename",
        "emotion",
        "speaker_id"
    ])

    for emotion in EMOTIONS:

        folder = os.path.join(DATASET_FOLDER, emotion)

        if not os.path.exists(folder):
            continue

        for audio in sorted(os.listdir(folder)):

            if audio.lower().endswith(".wav"):

                writer.writerow([
                    audio,
                    emotion.capitalize(),
                    os.path.splitext(audio)[0]
                ])

print("labels.csv created successfully!")