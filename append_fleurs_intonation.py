
from pathlib import Path
import shutil

import librosa
import pandas as pd

# Import Linguahe's existing acoustic feature extractor
from app import analyze_audio_features_in_memory


# =========================================================
# FILE PATHS — UPDATE THESE IF YOUR FOLDERS ARE DIFFERENT
# =========================================================
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"

INTONATION_CSV = PROJECT_DIR / "intonation_dataset.csv"
FLEURS_CSV = DATASET_DIR / "fleurs_training_labels.csv"
AUDIO_DIR = DATASET_DIR / "audio"

OUTPUT_CSV = DATASET_DIR / "intonation_dataset_with_fleurs_features.csv"

FEATURE_COLUMNS = [
    "normalized_final_pitch_change",
    "normalized_final_pitch_slope",
    "normalized_final_pitch_range",
]

DATASET_COLUMNS = [
    "audio_id",
    "audio_file",
    "transcript",
    "normalized_final_pitch_change",
    "normalized_final_pitch_slope",
    "normalized_final_pitch_range",
    "suggested_label",
    "intonation_label",
]


def main():
    # Check that all required files exist.
    for path in [INTONATION_CSV, FLEURS_CSV, AUDIO_DIR]:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    # Load both CSV files.
    existing = pd.read_csv(INTONATION_CSV)
    fleurs = pd.read_csv(FLEURS_CSV)

    required_existing = set(DATASET_COLUMNS)
    missing_columns = required_existing - set(existing.columns)

    if missing_columns:
        raise ValueError(
            f"Existing dataset is missing columns: {sorted(missing_columns)}"
        )

    if not {"audio_file", "transcription"}.issubset(fleurs.columns):
        raise ValueError(
            "FLEURS CSV must contain 'audio_file' and 'transcription'."
        )

    # Back up the original dataset once.
    backup = INTONATION_CSV.with_name(
        INTONATION_CSV.stem + "_backup.csv"
    )
    if not backup.exists():
        shutil.copy2(INTONATION_CSV, backup)

    # Prevent duplicate audio filenames.
    existing_files = set(
        existing["audio_file"].dropna().astype(str).str.lower()
    )

    new_rows = []
    missing_audio = []
    failed_audio = []

    print(f"Existing recordings: {len(existing)}")
    print(f"FLEURS metadata records: {len(fleurs)}")

    for row_number, row in fleurs.iterrows():
        filename = str(row["audio_file"]).strip()

        if not filename or filename.lower() == "nan":
            failed_audio.append(("<missing filename>", "No filename"))
            continue

        if filename.lower() in existing_files:
            continue

        audio_path = AUDIO_DIR / filename

        if not audio_path.is_file():
            missing_audio.append(filename)
            continue

        try:
            # Load audio the same way Linguahe expects it.
            audio, sample_rate = librosa.load(
                str(audio_path),
                sr=16000,
                mono=True,
            )

            # Use the existing Linguahe acoustic feature extractor.
            result = analyze_audio_features_in_memory(
                audio,
                sample_rate,
                transcript=str(row["transcription"]),
            )

            feature_values = {
                column: float(result[column])
                for column in FEATURE_COLUMNS
            }

            fleurs_id = row["id"] if "id" in fleurs.columns else row_number

            new_rows.append({
                "audio_id": f"FLEURS_{fleurs_id}",
                "audio_file": filename,
                "transcript": str(row["transcription"]),
                **feature_values,
                "suggested_label": "",
                "intonation_label": "",
            })

            existing_files.add(filename.lower())

            if len(new_rows) % 50 == 0:
                print(f"Processed {len(new_rows)} new recordings...")

        except Exception as error:
            failed_audio.append((filename, str(error)))

    # Create rows using the existing dataset's column structure.
    additions = pd.DataFrame(new_rows, columns=DATASET_COLUMNS)

    # Preserve the original rows and append the successfully processed rows.
    combined = pd.concat(
        [existing[DATASET_COLUMNS], additions],
        ignore_index=True,
    )

    # Save to a new CSV; never overwrite the original.
    combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n========== COMPLETE ==========")
    print(f"Original rows preserved: {len(existing)}")
    print(f"New rows added: {len(additions)}")
    print(f"Total rows: {len(combined)}")
    print(f"Original backup: {backup}")
    print(f"Combined dataset: {OUTPUT_CSV}")
    print(f"Missing audio files: {len(missing_audio)}")
    print(f"Failed extractions: {len(failed_audio)}")

    if missing_audio:
        pd.Series(missing_audio, name="audio_file").to_csv(
            DATASET_DIR / "fleurs_missing_audio.csv",
            index=False,
        )

    if failed_audio:
        pd.DataFrame(
            failed_audio,
            columns=["audio_file", "error"],
        ).to_csv(
            DATASET_DIR / "fleurs_failed_extractions.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print("\nReview the output before using it for model training.")
    print("New FLEURS intonation labels remain blank.")


if __name__ == "__main__":
    main()
