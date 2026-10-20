import joblib
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import load_model

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

encoder = joblib.load(
    "models/label_encoder_v2.pkl"
)

scaler = joblib.load(
    "models/scaler_v2.pkl"
)

model = load_model(
    "models/emotion_ann_v2.keras"
)

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
    target_names=encoder.classes_
))

print()

print(confusion_matrix(
    truth,
    prediction
))
