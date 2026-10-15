
from datasets import load_dataset

print("Loading FLEURS Filipino training dataset...")
print("The first download may take some time.")

dataset = load_dataset("google/fleurs", "fil_ph", split="train")

print(f"\nDataset loaded successfully!")
print(f"Number of recordings: {len(dataset)}")
print("\nFirst recording:")
print("ID:", dataset[0]["id"])
print("Transcript:", dataset[0]["transcription"])
print("Audio sample rate:", dataset[0]["audio"]["sampling_rate"])
