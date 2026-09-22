import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    Concatenate
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "intonation_dataset.csv"

MODEL_DIR = "intonation_ai/models"
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42

TEST_SIZE = 0.10
VALIDATION_SIZE = 0.10

BATCH_SIZE = 16
EPOCHS = 100


# ============================================================
# LOAD DATASET
# ============================================================

print("\n============================================================")
print("SMALL RA-SBERT + ACOUSTIC INTONATION MODEL")
print("============================================================")

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

required_columns = [
    "transcript",
    "normalized_final_pitch_change",
    "normalized_final_pitch_slope",
    "normalized_final_pitch_range",
    "intonation_label"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df = df.dropna(
    subset=required_columns
).reset_index(drop=True)

print(f"Dataset rows: {len(df)}")


# ============================================================
# LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    df["intonation_label"].astype(str)
)

print("\nClasses:")

for index, label in enumerate(
    label_encoder.classes_
):
    print(f"  {index}: {label}")


# ============================================================
# ACOUSTIC FEATURES
# ============================================================

acoustic_columns = [
    "normalized_final_pitch_change",
    "normalized_final_pitch_slope",
    "normalized_final_pitch_range"
]

A = df[
    acoustic_columns
].astype(
    np.float32
).values

print("\nAcoustic dimensions:", A.shape[1])


# ============================================================
# COMPUTE INTONATION RA-SBERT
# ============================================================

print("\n============================================================")
print("COMPUTING INTONATION RA-SBERT")
print("============================================================")

from app import intonation_ra_sbert_embedding

ra_sbert_vectors = []

for index, transcript in enumerate(
    df["transcript"].astype(str)
):

    vector, token_details = (
        intonation_ra_sbert_embedding(
            transcript
        )
    )

    vector = np.asarray(
        vector,
        dtype=np.float32
    )

    if vector.shape[0] != 384:
        raise ValueError(
            f"Expected 384-D RA-SBERT vector, "
            f"got {vector.shape[0]}"
        )

    ra_sbert_vectors.append(vector)

    if (
        (index + 1) % 25 == 0
        or index == len(df) - 1
    ):
        print(
            f"Processed {index + 1}/{len(df)}"
        )


V = np.vstack(
    ra_sbert_vectors
)

print(
    "\nRA-SBERT dimensions:",
    V.shape[1]
)


# ============================================================
# SAME DATA SPLIT
# ============================================================

indices = np.arange(
    len(df)
)

train_indices, temp_indices = train_test_split(
    indices,
    test_size=TEST_SIZE + VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

relative_test_size = (
    TEST_SIZE /
    (TEST_SIZE + VALIDATION_SIZE)
)

validation_indices, test_indices = (
    train_test_split(
        temp_indices,
        test_size=relative_test_size,
        random_state=RANDOM_STATE,
        stratify=y[temp_indices]
    )
)


print("\n============================================================")
print("DATA SPLIT")
print("============================================================")

print(
    f"Training:   {len(train_indices)}"
)

print(
    f"Validation: {len(validation_indices)}"
)

print(
    f"Test:       {len(test_indices)}"
)


# ============================================================
# SPLIT DATA
# ============================================================

V_train = V[train_indices]
V_validation = V[validation_indices]
V_test = V[test_indices]

A_train = A[train_indices]
A_validation = A[validation_indices]
A_test = A[test_indices]

y_train = y[train_indices]
y_validation = y[validation_indices]
y_test = y[test_indices]


# ============================================================
# SCALE ACOUSTIC FEATURES
# ============================================================

acoustic_scaler = StandardScaler()

A_train = acoustic_scaler.fit_transform(
    A_train
)

A_validation = acoustic_scaler.transform(
    A_validation
)

A_test = acoustic_scaler.transform(
    A_test
)


# ============================================================
# RA-SBERT PROJECTION
# ============================================================

print("\n============================================================")
print("BUILDING MODEL")
print("============================================================")

print("384-D RA-SBERT → 32-D projection")
print("32-D linguistic + 3-D acoustic → 35-D fusion")


# ============================================================
# MODEL INPUTS
# ============================================================

sbert_input = Input(
    shape=(384,),
    name="ra_sbert_input"
)

acoustic_input = Input(
    shape=(3,),
    name="acoustic_input"
)


# ============================================================
# RA-SBERT PROJECTION
# ============================================================

sbert_projection = Dense(
    32,
    activation="relu",
    name="sbert_projection"
)(
    sbert_input
)

sbert_projection = BatchNormalization()(
    sbert_projection
)

sbert_projection = Dropout(
    0.20
)(
    sbert_projection
)


# ============================================================
# FUSION
# ============================================================

combined = Concatenate(
    name="linguistic_acoustic_fusion"
)([
    sbert_projection,
    acoustic_input
])


print(
    "Combined representation: 35-D"
)


# ============================================================
# SMALL ANN CLASSIFIER
# ============================================================

x = Dense(
    32,
    activation="relu"
)(
    combined
)

x = BatchNormalization()(
    x
)

x = Dropout(
    0.20
)(
    x
)

x = Dense(
    16,
    activation="relu"
)(
    x
)

x = Dropout(
    0.15
)(
    x
)

output = Dense(
    len(label_encoder.classes_),
    activation="softmax",
    name="intonation_output"
)(
    x
)


# ============================================================
# MODEL
# ============================================================

model = Model(
    inputs=[
        sbert_input,
        acoustic_input
    ],
    outputs=output
)


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=12,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6
)


# ============================================================
# TRAIN
# ============================================================

print("\n============================================================")
print("TRAINING")
print("============================================================")

history = model.fit(

    [
        V_train,
        A_train
    ],

    y_train,

    validation_data=(
        [
            V_validation,
            A_validation
        ],
        y_validation
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# EVALUATE
# ============================================================

print("\n============================================================")
print("EVALUATING SMALL FUSION MODEL")
print("============================================================")

probabilities = model.predict(
    [
        V_test,
        A_test
    ],
    verbose=0
)

predictions = np.argmax(
    probabilities,
    axis=1
)


accuracy = accuracy_score(
    y_test,
    predictions
)


print(
    f"\nTest Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_,
        digits=4
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "intonation_ann_small_fusion.keras"
)

model.save(
    model_path
)

print(
    f"\nModel saved: {model_path}"
)


# ============================================================
# SAVE ACOUSTIC SCALER
# ============================================================

scaler_path = os.path.join(
    MODEL_DIR,
    "intonation_acoustic_scaler.pkl"
)

with open(
    scaler_path,
    "wb"
) as file:

    pickle.dump(
        acoustic_scaler,
        file
    )

print(
    f"Acoustic scaler saved: {scaler_path}"
)


# ============================================================
# SAVE LABEL ENCODER
# ============================================================

encoder_path = os.path.join(
    MODEL_DIR,
    "intonation_small_fusion_label_encoder.pkl"
)

with open(
    encoder_path,
    "wb"
) as file:

    pickle.dump(
        label_encoder,
        file
    )

print(
    f"Label encoder saved: {encoder_path}"
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_path = os.path.join(
    MODEL_DIR,
    "intonation_small_fusion_history.csv"
)

pd.DataFrame(
    history.history
).to_csv(
    history_path,
    index=False
)

print(
    f"Training history saved: {history_path}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n============================================================")
print("FINAL RESULT")
print("============================================================")

print(
    f"RA-SBERT input:       384-D"
)

print(
    f"RA-SBERT projection:   32-D"
)

print(
    f"Acoustic input:         3-D"
)

print(
    f"Fusion representation: 35-D"
)

print(
    f"Test accuracy: "
    f"{accuracy * 100:.2f}%"
)

print("\nPrevious baselines:")

print(
    "Acoustic-only: 82.14%"
)

print(
    "RA-SBERT-only: 67.86%"
)

print(
    "Previous 387-D fusion: 50.00%"
)

print("\n============================================================")
