
import csv
import os
import time
import requests

API = "https://datasets-server.huggingface.co/rows"
DATASET = "sapinsapin/filipinospeechcorpus"
OUTPUT = "sapinsapin_candidates.csv"
PROGRESS = "sapinsapin_scan_progress.txt"

BATCH_SIZE = 100
SCAN_LIMIT = 10000

FIELDS = [
    "sentence", "duration", "num_words", "speech_type",
    "speaker_id", "gender", "age_group", "source_file", "audio_url"
]

session = requests.Session()

if os.path.exists(OUTPUT):
    with open(OUTPUT, newline="", encoding="utf-8-sig") as f:
        candidates = list(csv.DictReader(f))
else:
    candidates = []

if os.path.exists(PROGRESS):
    with open(PROGRESS, encoding="utf-8") as f:
        start_offset = int(f.read().strip())
else:
    start_offset = 0

# Avoid duplicating existing candidates if restarting a scan.
seen = {row.get("source_file", "") for row in candidates}

for offset in range(start_offset, SCAN_LIMIT, BATCH_SIZE):
    params = {
        "dataset": DATASET,
        "config": "default",
        "split": "train",
        "offset": offset,
        "length": BATCH_SIZE,
    }

    rows = None

    for attempt in range(1, 6):
        try:
            response = session.get(API, params=params, timeout=45)
            response.raise_for_status()
            rows = response.json()["rows"]
            break
        except Exception as e:
            wait = attempt * 3
            print(
                f"Offset {offset}: attempt {attempt}/5 failed: {e}",
                flush=True,
            )
            if attempt < 5:
                time.sleep(wait)

    if rows is None:
        print(f"Could not retrieve offset {offset}. Progress saved.")
        break

    for item in rows:
        row = item["row"]

        try:
            duration = float(row["duration"])
            num_words = int(row["num_words"])
        except (TypeError, ValueError, KeyError):
            continue

        source_file = row.get("source_file", "")
        if (
            row.get("speech_type") in ("read", "spontaneous")
            and 1.5 <= duration <= 30
            and num_words >= 3
            and source_file not in seen
        ):
            audio = row.get("audio")
            audio_url = ""

            if isinstance(audio, list) and audio:
                audio_url = audio[0].get("src", "")
            elif isinstance(audio, dict):
                audio_url = audio.get("src", "")

            candidates.append({
                "sentence": row.get("sentence", ""),
                "duration": duration,
                "num_words": num_words,
                "speech_type": row.get("speech_type", ""),
                "speaker_id": row.get("speaker_id", ""),
                "gender": row.get("gender", ""),
                "age_group": row.get("age_group", ""),
                "source_file": source_file,
                "audio_url": audio_url,
            })
            seen.add(source_file)

    with open(OUTPUT, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(candidates)

    # Save the next offset only after this batch was processed and saved.
    with open(PROGRESS, "w", encoding="utf-8") as f:
        f.write(str(offset + len(rows)))

    print(
        f"Scanned through row {offset + len(rows)} | "
        f"candidates saved: {len(candidates)}",
        flush=True,
    )

    time.sleep(0.5)

print(f"\nFinished or paused. Candidates saved: {len(candidates)}")
print(f"CSV: {OUTPUT}")
print(f"Resume position: {PROGRESS}")
