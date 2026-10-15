
import csv
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
import io

# Existing downloaded FLEURS training file
PARQUET_FILE = Path(
    r"C:\Users\sedge\.cache\huggingface\hub\datasets--google--fleurs"
    r"\snapshots\70bb2e84b976b7e960aa89f1c648e09c59f894dd"
    r"\parquet-data\fil_ph\train-00000-of-00001.parquet"
)

# Output folders
OUTPUT_DIR = Path(r"C:\Thesis\Linguahe\dataset")
AUDIO_DIR = OUTPUT_DIR / "audio"
CSV_FILE = OUTPUT_DIR / "fleurs_training_labels.csv"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)

print("Reading the local FLEURS training file...")
parquet = pq.ParquetFile(PARQUET_FILE)

with CSV_FILE.open("w", newline="", encoding="utf-8-sig") as csv_file:
    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "id",
            "audio_file",
            "transcription",
            "raw_transcription",
            "gender",
            "diction_label",
            "intonation_label",
            "emotion_label",
        ],
    )
    writer.writeheader()

    exported = 0

    for batch in parquet.iter_batches(batch_size=16):
        for row in batch.to_pylist():
            audio_data = row["audio"]["bytes"]
            original_path = row["audio"]["path"]

            if not audio_data:
                print(f"Skipping ID {row['id']}: audio data is empty")
                continue

            # Use the recording ID to create a consistent filename.
            audio_file = f"fleurs_{row['id']}.wav"
            output_audio = AUDIO_DIR / audio_file

            if not output_audio.exists():
                with sf.SoundFile(io.BytesIO(audio_data)) as source:
                    samples = source.read(dtype="float32")
                    sample_rate = source.samplerate

                sf.write(output_audio, samples, sample_rate)

            writer.writerow({
                "id": row["id"],
                "audio_file": audio_file,
                "transcription": row["transcription"],
                "raw_transcription": row["raw_transcription"],
                "gender": row["gender"],
                "diction_label": "",
                "intonation_label": "",
                "emotion_label": "",
            })

            exported += 1

        print(f"Processed {exported} recordings...")

print("\nExport finished!")
print(f"Recordings exported: {exported}")
print(f"Audio folder: {AUDIO_DIR}")
print(f"Label CSV: {CSV_FILE}")
