import os
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

# Import the RA-SBERT function from your existing system
from app import ra_sbert_embedding


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "diction_dataset_600_balanced.csv"
MODEL_DIR = "diction_ai/models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "diction_ann.keras"
)

ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "label_encoder.pkl"
)


# ============================================================
# LOAD DATASET
# ============================================================

print("\n========================================")
print("LINGUAHE DICTION ANN TRAINING")
print("========================================")

print("\n--- Loading Dataset ---")

df = pd.read_csv(DATASET_PATH)

required_columns = [
    "sentence",
    "diction_label"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Missing required column: {column}"
        )

df = df.dropna(
    subset=["sentence", "diction_label"]
)

df["sentence"] = (
    df["sentence"]
    .astype(str)
    .str.strip()
)

df["diction_label"] = (
    df["diction_label"]
    .astype(str)
    .str.strip()
)

print(f"Total sentences: {len(df)}")

print("\nClass distribution:")
print(
    df["diction_label"]
    .value_counts()
)


# ============================================================
# GENERATE RA-SBERT FEATURES
# ============================================================

print("\n--- Generating RA-SBERT Features ---")
print("This is the slowest part of the training process.")

features = []
labels = []

for index, row in df.iterrows():

    sentence = row["sentence"]
    label = row["diction_label"]

    print(
        f"\rProcessing "
        f"{index + 1}/{len(df)}",
        end=""
    )

    vector, token_details = (
        ra_sbert_embedding(sentence)
    )

    if vector is None:
        print(
            f"\nWarning: Could not encode: "
            f"{sentence}"
        )
        continue

    # Convert PyTorch tensor → NumPy
    vector = (
        vector.detach()
        .cpu()
        .numpy()
    )

    features.append(vector)
    labels.append(label)

print("\n")

X = np.asarray(
    features,
    dtype=np.float32
)

y_text = np.asarray(
    labels
)

print(
    f"Feature matrix shape: {X.shape}"
)


# ============================================================
# VALIDATE FEATURE DIMENSION
# ============================================================

if X.shape[1] != 384:

    raise ValueError(
        f"Expected 384-dimensional "
        f"RA-SBERT vectors, got "
        f"{X.shape[1]}"
    )

print(
    "RA-SBERT dimension verified: 384"
)


# ============================================================
# ENCODE LABELS
# ============================================================

print("\n--- Encoding Labels ---")

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    y_text
)

print(
    "Classes:",
    list(label_encoder.classes_)
)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n--- Creating Dataset Splits ---")

# First split:
# 70% training
# 30% temporary
X_train, X_temp, y_train, y_temp = (
    train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )
)

# Second split:
# 15% validation
# 15% testing
X_val, X_test, y_val, y_test = (
    train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )
)

print(
    f"Training:   {len(X_train)}"
)

print(
    f"Validation: {len(X_val)}"
)

print(
    f"Testing:    {len(X_test)}"
)


# ============================================================
# BUILD DICTION ANN
# ============================================================

print("\n--- Building Diction ANN ---")

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(384,)
    ),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.20
    ),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.20
    ),

    tf.keras.layers.Dense(
        3,
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

    metrics=[
        "accuracy"
    ]
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

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=100,

    batch_size=16,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


# ============================================================
# TEST
# ============================================================

print("\n--- Evaluating Model ---")

test_loss, test_accuracy = (
    model.evaluate(
        X_test,
        y_test,
        verbose=0
    )
)

print(
    f"\nTest Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

y_probability = model.predict(
    X_test,
    verbose=0
)

y_prediction = np.argmax(
    y_probability,
    axis=1
)

print("\n--- Classification Report ---")

print(
    classification_report(
        y_test,
        y_prediction,
        target_names=label_encoder.classes_,
        digits=4
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n--- Confusion Matrix ---")

cm = confusion_matrix(
    y_test,
    y_prediction
)

print(
    pd.DataFrame(
        cm,
        index=label_encoder.classes_,
        columns=label_encoder.classes_
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\n--- Saving Model ---")

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

model.save(
    MODEL_PATH
)

joblib.dump(
    label_encoder,
    ENCODER_PATH
)

print(
    f"Model saved to: {MODEL_PATH}"
)

print(
    f"Label encoder saved to: "
    f"{ENCODER_PATH}"
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_path = os.path.join(
    MODEL_DIR,
    "training_history.csv"
)

pd.DataFrame(
    history.history
).to_csv(
    history_path,
    index=False
)

print(
    f"Training history saved to: "
    f"{history_path}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("DICTION ANN TRAINING COMPLETE")
print("========================================")

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    "Classes:",
    list(label_encoder.classes_)
)

print(
    f"Model: {MODEL_PATH}"
)

print(
    f"Encoder: {ENCODER_PATH}"
)

print("\nNext step:")
print(
    "Integrate diction_ann.keras "
    "into Linguahe /analyze."
)