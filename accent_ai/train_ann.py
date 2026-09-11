import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ------------------------
# Load dataset
# ------------------------

data = pd.read_csv(
    "features/features.csv"
)

print(data.head())

# ------------------------
# Split X and y
# ------------------------

X = data.drop(
    columns=[
        "filename",
        "speaker_id",
        "accent"
    ]
)

y = data["accent"]

# ------------------------
# Encode labels
# ------------------------

encoder = LabelEncoder()

y = encoder.fit_transform(y)

print("Classes:")
print(encoder.classes_)

# ------------------------
# Train/Test split
# ------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------
# Normalize
# ------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ------------------------
# Build ANN
# ------------------------

model = Sequential()

model.add(
    Dense(
        64,
        activation="relu",
        input_shape=(X_train.shape[1],)
    )
)

model.add(
    Dropout(0.30)
)

model.add(
    Dense(
        32,
        activation="relu"
    )
)

model.add(
    Dropout(0.30)
)

model.add(
    Dense(
        len(encoder.classes_),
        activation="softmax"
    )
)

# ------------------------
# Compile
# ------------------------

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)

# ------------------------
# Train
# ------------------------

history = model.fit(

    X_train,

    y_train,

    validation_split=0.20,

    epochs=50,

    batch_size=8
)

# ------------------------
# Evaluate
# ------------------------

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print(f"\nAccuracy: {accuracy*100:.2f}%")

# ------------------------
# Save
# ------------------------

os.makedirs(
    "models",
    exist_ok=True
)

model.save(
    "models/accent_ann.keras"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

joblib.dump(
    encoder,
    "models/label_encoder.pkl"
)

print("\nTraining Complete!")

print("\nSaved Files:")

print("accent_ann.keras")

print("scaler.pkl")

print("label_encoder.pkl")