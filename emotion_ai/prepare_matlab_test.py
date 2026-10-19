"""Create a seven-class held-out audio set while keeping the original test clips."""

import hashlib
import random
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
TEST_DATASET = ROOT / "test_dataset"
FULL_LABELS = ROOT / "labels_all.csv"
TRAIN_LABELS = ROOT / "labels.csv"
MANIFEST = ROOT / "matlab_test_manifest.csv"
SEED = 20261019
HOLDOUT_FRACTION = 0.20


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_emotion(value):
    emotion = str(value).strip().lower()
    return "neutral" if emotion == "normal" else emotion


if not FULL_LABELS.exists():
    shutil.copy2(TRAIN_LABELS, FULL_LABELS)

all_labels = pd.read_csv(FULL_LABELS)
# Do not introduce identical recordings into both the training and test sets.
existing_hashes = {
    sha256(path)
    for path in TEST_DATASET.rglob("*.wav")
    if path.is_file()
}

rng = random.Random(SEED)
previous_manifest = pd.read_csv(MANIFEST) if MANIFEST.exists() else pd.DataFrame()
selected = []
selected_hashes = set()
for emotion, group in all_labels.groupby(
    all_labels["emotion"].map(canonical_emotion), sort=True
):
    rows = group.to_dict("records")
    previous_rows = (
        previous_manifest[
            previous_manifest["emotion"].map(canonical_emotion) == emotion
        ].to_dict("records")
        if not previous_manifest.empty
        else []
    )
    count = max(1, round(len(rows) * HOLDOUT_FRACTION))
    ordered = previous_rows + rng.sample(rows, len(rows))
    chosen = []
    for row in ordered:
        if len(chosen) >= count:
            break
        source = DATASET / emotion / str(row["filename"])
        if not source.exists():
            continue
        source_hash = sha256(source)
        target = TEST_DATASET / emotion / f"holdout_{source.name}"
        is_our_existing_holdout = (
            target.exists() and sha256(target) == source_hash
        )
        if source_hash in selected_hashes:
            continue
        if source_hash in existing_hashes and not is_our_existing_holdout:
            continue
        chosen.append({**row, "emotion": emotion})
        selected_hashes.add(source_hash)
    if len(chosen) != count:
        raise ValueError(
            f"Could only select {len(chosen)} of {count} held-out files for {emotion}."
        )
    selected.extend(chosen)

manifest = pd.DataFrame(selected)
manifest.to_csv(MANIFEST, index=False)

for row in manifest.to_dict("records"):
    emotion = canonical_emotion(row["emotion"])
    source_emotion = canonical_emotion(row["emotion"])
    source = DATASET / source_emotion / str(row["filename"])
    if not source.exists():
        # Labels may preserve title case or a legacy Normal label.
        original_emotion = str(row.get("source_emotion", row["emotion"])).lower()
        source = DATASET / original_emotion / str(row["filename"])
    if not source.exists():
        raise FileNotFoundError(f"Held-out source audio is missing: {source}")

    source_hash = sha256(source)
    target_dir = TEST_DATASET / emotion
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"holdout_{source.name}"
    if target.exists():
        if sha256(target) != source_hash:
            raise ValueError(f"Existing held-out audio differs: {target}")
        continue
    if source_hash in existing_hashes:
        raise ValueError(f"Duplicate audio found in test set: {source.name}")
    shutil.copy2(source, target)
    existing_hashes.add(source_hash)

heldout_keys = {
    (canonical_emotion(row["emotion"]), str(row["filename"]))
    for row in manifest.to_dict("records")
}
train_labels = all_labels[
    [
        (canonical_emotion(row.emotion), str(row.filename)) not in heldout_keys
        for row in all_labels.itertuples(index=False)
    ]
].copy()
train_labels.to_csv(TRAIN_LABELS, index=False)

print(f"Training audio labels: {len(train_labels)}")
print(f"Held-out source clips: {len(manifest)}")
print("Added held-out clips by class:")
print(manifest.groupby(manifest["emotion"].map(canonical_emotion)).size())
