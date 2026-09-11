import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

data = pd.read_csv(
    "features/features.csv"
)

X = data.drop(
    columns=[
        "filename",
        "speaker_id",
        "emotion"
    ]
)

y = data["emotion"]

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

model = Sequential()

model.add(
    Dense(
        64,
        activation="relu",
        input_shape=(18,)
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
        len(label_encoder.classes_),
        activation="softmax"
    )
)

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

history = model.fit(

    X_train,

    y_train,

    validation_data=(

        X_test,

        y_test

    ),

    epochs=100,

    batch_size=8,

    verbose=1

)

loss, accuracy = model.evaluate(

    X_test,

    y_test,

    verbose=0

)

print()

print("Accuracy:", accuracy)

os.makedirs(
    "models",
    exist_ok=True
)

model.save(
    "models/emotion_ann.keras"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

joblib.dump(
    label_encoder,
    "models/label_encoder.pkl"
)

print()

print("Emotion ANN saved!")