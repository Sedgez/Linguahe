import os
import random
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from app import ra_sbert_embedding


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DATASET_PATH = "diction_dataset_1500_balanced.csv"
MODEL_DIR = "diction_ai/models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "diction_ann.keras"
)

ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "label_encoder.pkl"
)

HISTORY_PATH = os.path.join(
    MODEL_DIR,
    "training_history.csv"
)

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

EXPECTED_CLASSES = {"Plain", "Polite", "Formal"}
FEATURE_DIM = 384


# ============================================================
# LOAD DATASET
# ============================================================

print("\n========================================")
print("LINGUAHE DICTION ANN TRAINING")
print("========================================")

print("\n--- Loading Dataset ---")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}\n"
        "Place diction_dataset_1500_balanced.csv "
        "in the project root folder."
    )

df = pd.read_csv(DATASET_PATH)

required_columns = ["sentence", "diction_label"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Missing required column: {column}"
        )

df = df.dropna(
    subset=["sentence", "diction_label"]
).copy()

df["sentence"] = (
    df["sentence"].astype(str).str.strip()
)

df["diction_label"] = (
    df["diction_label"].astype(str).str.strip().str.title()
)

df = df[
    (df["sentence"] != "")
    & (df["diction_label"].isin(EXPECTED_CLASSES))
].copy()

print(f"Rows loaded: {len(df)}")

print("\nOriginal class distribution:")
print(df["diction_label"].value_counts())

# ------------------------------------------------------------
# Prevent the same sentence from appearing in multiple splits.
# Sentences assigned conflicting labels are excluded because
# they cannot be safely treated as unambiguous training examples.
# ------------------------------------------------------------

label_counts = df.groupby("sentence")["diction_label"].nunique()

conflicting_sentences = label_counts[
    label_counts > 1
].index

if len(conflicting_sentences) > 0:
    print(
        "\nWarning: Excluding "
        f"{len(conflicting_sentences)} sentences "
        "that have conflicting labels."
    )

df = df[
    ~df["sentence"].isin(conflicting_sentences)
].copy()

before_deduplication = len(df)

df = df.drop_duplicates(
    subset=["sentence"]
).reset_index(drop=True)

duplicates_removed = before_deduplication - len(df)

print(f"Duplicate rows removed: {duplicates_removed}")
print(f"Unique sentences available: {len(df)}")

print("\nUnique sentence distribution:")
print(df["diction_label"].value_counts())

if set(df["diction_label"].unique()) != EXPECTED_CLASSES:
    raise ValueError(
        "The cleaned dataset must contain all three classes: "
        "Plain, Polite, and Formal."
    )

if df["diction_label"].value_counts().min() < 3:
    raise ValueError(
        "At least three unique sentences per class are "
        "required for the stratified train/validation/test split."
    )


# ============================================================
# ENCODE LABELS
# ============================================================

print("\n--- Encoding Labels ---")

label_encoder = LabelEncoder()

y_all = label_encoder.fit_transform(
    df["diction_label"].to_numpy()
)

X_text = df["sentence"].to_numpy()

class_names = label_encoder.classes_
num_classes = len(class_names)

print("Classes:", list(class_names))


# ============================================================
# SPLIT UNIQUE SENTENCES BEFORE TRAINING
# ============================================================

print("\n--- Creating Dataset Splits ---")

# 70% training, 30% temporary
text_train, text_temp, y_train, y_temp = train_test_split(
    X_text,
    y_all,
    test_size=0.30,
    random_state=SEED,
    stratify=y_all
)

# Split the remaining 30% equally:
# 15% validation and 15% testing
text_val, text_test, y_val, y_test = train_test_split(
    text_temp,
    y_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=y_temp
)

print(f"Training sentences:   {len(text_train)}")
print(f"Validation sentences: {len(text_val)}")
print(f"Testing sentences:    {len(text_test)}")


# ============================================================
# GENERATE RA-SBERT FEATURES
# ============================================================

print("\n--- Generating RA-SBERT Features ---")
print("Each unique sentence is encoded only once.")

all_text = np.concatenate(
    [text_train, text_val, text_test]
)

features_by_sentence = {}

for index, sentence in enumerate(all_text, start=1):

    print(
        f"\rProcessing {index}/{len(all_text)}",
        end="",
        flush=True
    )

    vector, token_details = ra_sbert_embedding(sentence)

    if vector is None:
        raise ValueError(
            f"\nRA-SBERT could not encode this sentence:\n"
            f"{sentence}"
        )

    # Convert PyTorch tensor to NumPy when necessary.
    if hasattr(vector, "detach"):
        vector = vector.detach().cpu().numpy()

    vector = np.asarray(
        vector,
        dtype=np.float32
    ).reshape(-1)

    if vector.size != FEATURE_DIM:
        raise ValueError(
            f"\nExpected {FEATURE_DIM} features, "
            f"got {vector.size} for:\n{sentence}"
        )

    if not np.isfinite(vector).all():
        raise ValueError(
            f"\nNon-finite feature values found for:\n{sentence}"
        )

    features_by_sentence[sentence] = vector

print("\nRA-SBERT feature generation complete.")


def make_feature_matrix(sentences):
    return np.stack([
        features_by_sentence[sentence]
        for sentence in sentences
    ]).astype(np.float32)


X_train = make_feature_matrix(text_train)
X_val = make_feature_matrix(text_val)
X_test = make_feature_matrix(text_test)

print("Training feature shape:", X_train.shape)
print("Validation feature shape:", X_val.shape)
print("Testing feature shape:", X_test.shape)

if (
    X_train.shape[1] != FEATURE_DIM
    or X_val.shape[1] != FEATURE_DIM
    or X_test.shape[1] != FEATURE_DIM
):
    raise ValueError("RA-SBERT feature dimension is incorrect.")


# ============================================================
# BALANCE TRAINING SET ONLY
# ============================================================

print("\n--- Balancing Training Set ---")

rng = np.random.default_rng(SEED)

train_indices_by_class = {
    class_id: np.flatnonzero(y_train == class_id)
    for class_id in range(num_classes)
}

target_count = max(
    len(indices)
    for indices in train_indices_by_class.values()
)

balanced_indices = []

for class_id, indices in train_indices_by_class.items():

    if len(indices) == 0:
        raise ValueError(
            f"No training examples for {class_names[class_id]}"
        )

    # Keep all original training examples.
    selected = list(indices)

    # Oversample only this training class if necessary.
    if len(indices) < target_count:
        extra = rng.choice(
            indices,
            size=target_count - len(indices),
            replace=True
        )
        selected.extend(extra.tolist())

    balanced_indices.extend(selected)

balanced_indices = np.asarray(
    balanced_indices,
    dtype=np.int64
)

rng.shuffle(balanced_indices)

X_train_balanced = X_train[balanced_indices]
y_train_balanced = y_train[balanced_indices]

print("Training rows after balancing:")
for class_id, class_name in enumerate(class_names):
    count = int(np.sum(y_train_balanced == class_id))
    print(f"  {class_name}: {count}")

print(
    "\nValidation and test sets remain unoversampled."
)


# ============================================================
# BUILD DICTION ANN
# ============================================================

print("\n--- Building Diction ANN ---")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(FEATURE_DIM,)),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.20),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.20),

    tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ============================================================
# TRAIN
# ============================================================

print("\n--- Training Diction ANN ---")

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train_balanced,
    y_train_balanced,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=[early_stopping],
    verbose=1
)


# ============================================================
# EVALUATE ON HELD-OUT TEST SET
# ============================================================

print("\n--- Evaluating Model ---")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

y_probability = model.predict(
    X_test,
    verbose=0
)

y_prediction = np.argmax(
    y_probability,
    axis=1
)

print(
    f"Independent test accuracy: "
    f"{accuracy_score(y_test, y_prediction) * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n--- Classification Report ---")

print(
    classification_report(
        y_test,
        y_prediction,
        labels=np.arange(num_classes),
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n--- Confusion Matrix ---")

cm = confusion_matrix(
    y_test,
    y_prediction,
    labels=np.arange(num_classes)
)

print(
    pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    )
)


# ============================================================
# SAVE MODEL AND ARTIFACTS
# ============================================================

print("\n--- Saving Model ---")

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

model.save(MODEL_PATH)

joblib.dump(
    label_encoder,
    ENCODER_PATH
)

pd.DataFrame(
    history.history
).to_csv(
    HISTORY_PATH,
    index=False
)

print(f"Model saved to: {MODEL_PATH}")
print(f"Label encoder saved to: {ENCODER_PATH}")
print(f"Training history saved to: {HISTORY_PATH}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("DICTION ANN TRAINING COMPLETE")
print("========================================")

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print("Classes:", list(class_names))
print(f"Model: {MODEL_PATH}")
print(f"Encoder: {ENCODER_PATH}")

print("\nNext step:")
print("Integrate diction_ann.keras into Linguahe /analyze.")