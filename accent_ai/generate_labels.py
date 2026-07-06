import os
import csv

DATASET_PATH = "dataset"
OUTPUT_CSV = "labels.csv"

ACCENTS = [
    "batangas",
    "cavite",
    "laguna",
    "quezon",
    "rizal"
]

with open(OUTPUT_CSV, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "filename",
        "accent",
        "speaker_id"
    ])

    for accent in ACCENTS:

        folder = os.path.join(DATASET_PATH, accent)

        if not os.path.exists(folder):
            print(f"Folder not found: {folder}")
            continue

        files = sorted(os.listdir(folder))

        count = 1

        for filename in files:

            if filename.lower().endswith(".wav"):

                speaker_id = f"{accent[0].upper()}{count:03d}"

                writer.writerow([
                    filename,
                    accent.capitalize(),
                    speaker_id
                ])

                count += 1

print("labels.csv generated successfully!")