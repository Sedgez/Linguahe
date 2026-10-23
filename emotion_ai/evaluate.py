
import joblib
import pandas as pd
from pathlib import Path

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent

data = pd.read_csv(
    BASE_DIR / "features" / "features.csv"
)

X = data.drop(
    columns=[
        "filename",
        "speaker_id",
        "emotion"
    ]
)

y = data["emotion"]

encoder = joblib.load(
    BASE_DIR / "models" / "label_encoder_v2.pkl"
)

scaler = joblib.load(
    BASE_DIR / "models" / "scaler_v2.pkl"
)

model = load_model(
    BASE_DIR / "models" / "emotion_ann_v2.keras"
)

print("Emotion classes:", encoder.classes_)

X = scaler.transform(X)

prediction = model.predict(
    X,
    verbose=0
)

prediction = prediction.argmax(axis=1)

truth = encoder.transform(y)

print()

print(classification_report(
    truth,
    prediction,
    labels=range(len(encoder.classes_)),
    target_names=encoder.classes_,
    zero_division=0
))

print()

print(confusion_matrix(
    truth,
    prediction,
    labels=range(len(encoder.classes_))
))
