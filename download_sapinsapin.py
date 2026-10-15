
import csv
import io
import os
import time
import requests

# Dataset Viewer settings
DATASET = "sapinsapin/filipinospeechcorpus"
API_URL = "https://datasets-server.huggingface.co/rows"
SPLIT = "train"

# Scan up to 10,000 rows per run.
# The script saves progress so it can resume later.
SCAN_LIMIT = 100_000
BATCH_SIZE = 100

AUDIO_DIR = "sapinsapin_audio"
METADATA_FILE = "sapinsapin_downloaded.csv"
PROGRESS_FILE = "sapinsapin_download_progress.txt"

os.makedirs(AUDIO_DIR, exist_ok=True)
session = requests.Session()


def get_with_retry(url, params=None, attempts=6):
    """Retry temporary network errors and rate limits."""
    for attempt in range(attempts):
        try:
            response = session.get(
                url, params=params, timeout=90
            )

            if response.status_code == 429 or response.status_code in (
                500, 502, 503, 504
            ):
                retry_after = response.headers.get("Retry-After")
                try:
                    wait = float(retry_after)
                except (TypeError, ValueError):
                    wait = min(10 * (attempt + 1), 60)

                print(
                    f"Temporary HTTP {response.status_code}. "
                    f"Waiting {wait:.0f}s..."
                )
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response

        except requests.RequestException as exc:
            if attempt == attempts - 1:
                raise
            wait = min(5 * (attempt + 1), 30)
            print(f"Request failed: {exc}")
            print(f"Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError("Request retries exhausted")


def get_rows(offset, length):
    params = {
        "dataset": DATASET,
        "config": "default",
        "split": SPLIT,
        "offset": offset,
        "length": length,
    }
    return get_with_retry(API_URL, params=params).json()


def get_audio_url(row):
    audio = row.get("audio")

    if isinstance(audio, list) and audio:
        audio = audio[0]

    if isinstance(audio, dict):
        return audio.get("src", "")

    return ""


def download_audio(row_index, row):
    """Fetch a fresh URL and save a verified WAV."""
    path = os.path.join(
        AUDIO_DIR, f"row_{row_index:06d}.wav"
    )

    # A completed file doesn't need to be downloaded again.
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path

    for attempt in range(3):
        url = get_audio_url(row)

        if not url:
            raise RuntimeError("No audio URL in dataset row")

        response = session.get(url, timeout=90)

        # Refresh the temporary URL if the cached asset rejects it.
        if response.status_code == 403:
            print("Refreshing expired/rejected audio URL...")
            row = get_rows(row_index, 1)["rows"][0]["row"]
            time.sleep(2)
            continue

        if response.status_code == 429:
            wait = min(15 * (attempt + 1), 60)
            print(f"Audio rate-limited; waiting {wait}s...")
            time.sleep(wait)
            row = get_rows(row_index, 1)["rows"][0]["row"]
            continue

        response.raise_for_status()
        content = response.content

        # Reject error pages or non-WAV responses.
        if len(content) < 1000 or content[:4] != b"RIFF":
            raise RuntimeError("Downloaded data is not a valid WAV")

        if content[8:12] != b"WAVE":
            raise RuntimeError("Invalid WAV header")

        temporary_path = path + ".part"
        with open(temporary_path, "wb") as audio_file:
            audio_file.write(content)

        os.replace(temporary_path, path)
        return path

    raise RuntimeError("Could not retrieve audio after URL refreshes")


# Load the set of completed dataset row indices.
completed = set()
if os.path.exists(METADATA_FILE):
    with open(
        METADATA_FILE, newline="", encoding="utf-8-sig"
    ) as file:
        for item in csv.DictReader(file):
            completed.add(int(item["row_idx"]))

# Resume from the last saved scanning position.
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, "r", encoding="utf-8") as file:
        start_offset = int(file.read().strip() or "0")
else:
    start_offset = 0

end_offset = min(start_offset + SCAN_LIMIT, 274730)

fields = [
    "row_idx", "audio_file", "sentence", "duration",
    "num_words", "speech_type", "speaker_id",
    "gender", "age_group", "source_file"
]

write_header = not os.path.exists(METADATA_FILE)

with open(
    METADATA_FILE, "a", newline="", encoding="utf-8"
) as metadata_file:

    writer = csv.DictWriter(metadata_file, fieldnames=fields)

    if write_header:
        writer.writeheader()

    offset = start_offset

    while offset < end_offset:
        length = min(BATCH_SIZE, end_offset - offset)

        try:
            batch = get_rows(offset, length)
        except Exception as exc:
            print(f"Could not retrieve rows at {offset}: {exc}")
            print("Progress is saved. Run this script again later.")
            break

        rows = batch.get("rows", [])
        if not rows:
            print("No more rows returned; stopping.")
            break

        for item in rows:
            row_index = item["row_idx"]
            row = item["row"]

            if row_index in completed:
                continue

            try:
                speech_type = str(
                    row.get("speech_type", "")
                ).lower()
                duration = float(row.get("duration") or 0)
                num_words = int(row.get("num_words") or 0)

                # Keep only longer read or spontaneous speech.
                if speech_type not in ("read", "spontaneous"):
                    continue
                if not (3.0 <= duration <= 30.0):
                    continue
                if num_words < 3:
                    continue

                audio_path = download_audio(row_index, row)

                writer.writerow({
                    "row_idx": row_index,
                    "audio_file": audio_path,
                    "sentence": row.get("sentence", ""),
                    "duration": duration,
                    "num_words": num_words,
                    "speech_type": speech_type,
                    "speaker_id": row.get("speaker_id", ""),
                    "gender": row.get("gender", ""),
                    "age_group": row.get("age_group", ""),
                    "source_file": row.get("source_file", ""),
                })
                metadata_file.flush()
                completed.add(row_index)

                print(
                    f"Saved row {row_index}: "
                    f"{duration:.2f}s, {speech_type} "
                    f"({len(completed)} total saved)"
                )

                # Avoid hammering the audio endpoint.
                time.sleep(1)

            except Exception as exc:
                print(f"Row {row_index} failed: {exc}")

        # Save the next batch position after processing this batch.
        offset += len(rows)
        with open(PROGRESS_FILE, "w", encoding="utf-8") as file:
            file.write(str(offset))

        print(f"Scanned through row {offset} / 274730")
        time.sleep(2)

print("\nRun finished.")
print("Audio folder:", AUDIO_DIR)
print("Metadata:", METADATA_FILE)
print("Next scan position:", PROGRESS_FILE)
print("Completed recordings:", len(completed))
