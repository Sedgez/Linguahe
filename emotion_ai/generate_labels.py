import csv
from pathlib import Path

DATASET_FOLDER = Path("dataset")
VALID_EMOTIONS = {"happy", "sad", "angry", "normal"}

with open("labels.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "emotion", "speaker_id"])

    for emotion_folder in sorted(DATASET_FOLDER.iterdir()):
        if not emotion_folder.is_dir():
            continue

        emotion = emotion_folder.name.lower()
        if emotion not in VALID_EMOTIONS:
            print(f"Skipping unknown emotion folder: {emotion_folder}")
            continue

        for audio_path in sorted(emotion_folder.iterdir()):
            if audio_path.is_file() and audio_path.suffix.lower() == ".wav":
                writer.writerow([
                    audio_path.name,
                    emotion.capitalize(),
                    audio_path.stem,
                ])

print("labels.csv created successfully!")
