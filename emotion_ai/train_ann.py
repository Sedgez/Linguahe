import json
import os
import zipfile

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2


data = pd.read_csv("features/features.csv")
X = data.drop(columns=["filename", "speaker_id", "emotion"]).to_numpy(dtype=np.float32)
labels = data["emotion"].astype(str)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

experiments = [
    ((64, 32), 0.25, 0.001),
    ((128, 64, 32), 0.25, 0.001),
    ((256, 128, 64), 0.30, 0.001),
    ((128, 64), 0.20, 0.0005),
]
best_accuracy = -1.0
best_loss = float("inf")
best_epoch = 0
best_model = None

for experiment_index, (widths, dropout_rate, regularization) in enumerate(experiments):
    for seed in (17, 42):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        layers = [Input(shape=(X_train.shape[1],))]
        for layer_index, width in enumerate(widths):
            layers.append(Dense(width, activation="relu", kernel_regularizer=l2(regularization)))
            if layer_index < len(widths) - 1:
                layers.append(Dropout(dropout_rate))
        layers.append(Dense(len(label_encoder.classes_), activation="softmax"))
        model = Sequential(layers)
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_valid, y_valid),
            epochs=250,
            batch_size=32,
            callbacks=[
                EarlyStopping(
                    monitor="val_accuracy",
                    mode="max",
                    patience=30,
                    restore_best_weights=True,
                ),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=10,
                    min_lr=1e-5,
                ),
            ],
            verbose=0,
        )
        loss, accuracy = model.evaluate(X_valid, y_valid, verbose=0)
        epoch = int(np.argmax(history.history["val_accuracy"]) + 1)
        print(
            f"Trial {experiment_index + 1}, seed {seed}: "
            f"val_accuracy={accuracy:.4f}, val_loss={loss:.4f}, best_epoch={epoch}"
        )
        if accuracy > best_accuracy or (accuracy == best_accuracy and loss < best_loss):
            best_accuracy = float(accuracy)
            best_loss = float(loss)
            best_epoch = epoch
            best_model = model

model = best_model
print(f"Selected validation accuracy: {best_accuracy:.4f}")
print(f"Selected validation loss: {best_loss:.4f}")
print(f"Selected best epoch: {best_epoch}")

# Save the model and preprocessing artifacts together for app inference.
os.makedirs("models", exist_ok=True)
model_path = "models/emotion_ann_v2.keras"
weights_path = "models/emotion_ann_v2.weights.h5"
model.save_weights(weights_path)
with zipfile.ZipFile(model_path, mode="w", compression=zipfile.ZIP_STORED) as archive:
    archive.writestr("metadata.json", json.dumps({"keras_version": "3"}))
    archive.writestr("config.json", model.to_json())
    archive.write(weights_path, arcname="model.weights.h5")
os.remove(weights_path)

joblib.dump(scaler, "models/scaler_v2.pkl")
joblib.dump(label_encoder, "models/label_encoder_v2.pkl")
