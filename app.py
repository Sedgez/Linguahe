from flask import Flask, render_template, request, jsonify
import whisper
import os
import re
import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 1. WHISPER MODEL
# -----------------------------
print(f"--- Initializing Whisper on {device} ---")

try:
    whisper_model = whisper.load_model("turbo", device=device)
except Exception as e:
    print(f"Turbo failed: {e}")
    whisper_model = whisper.load_model("medium", device=device)

# -----------------------------
# 2. SBERT MODEL
# -----------------------------
print("--- Loading SBERT ---")
sbert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# -----------------------------
# 3. PROVINCES (FIXED)
# -----------------------------
provinces = ["batangas", "laguna", "cavite", "rizal", "quezon"]

# -----------------------------
# 4. LOAD PROTOTYPES (sentence, dialect)
# -----------------------------
def load_sbert_prototypes():
    path = "reference_sentences.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        df["dialect"] = df["dialect"].astype(str).str.lower().str.strip()
        df["sentence"] = df["sentence"].astype(str)
        return df
    return pd.DataFrame()

proto_df = load_sbert_prototypes()

# Precompute SBERT embeddings per province
proto_embeddings = {}

for prov in provinces:
    sentences = proto_df[proto_df["dialect"] == prov]["sentence"].tolist()

    if len(sentences) == 0:
        proto_embeddings[prov] = None
        continue

    proto_embeddings[prov] = sbert_model.encode(
        sentences,
        convert_to_tensor=True
    )

# -----------------------------
# 5. LOAD RISK WORDS
# -----------------------------
def load_risk_db():
    path = "risk_words.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        return df.fillna("")
    return pd.DataFrame()

risk_df = load_risk_db()

# -----------------------------
# 6. AUDIO CLEANING
# -----------------------------
def clean_audio(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        y = librosa.util.normalize(y)
        y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.7)
        sf.write(audio_path, y, sr)
    except:
        pass

# -----------------------------
# 7. INTONATION ANALYSIS
# -----------------------------
def analyze_intonation(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        pitches, mags = librosa.piptrack(y=y, sr=sr)

        pitch_vals = pitches[mags > np.median(mags)]
        pitch_vals = pitch_vals[pitch_vals > 0]

        if len(pitch_vals) < 5:
            return "Flat / Neutral"

        std = np.std(pitch_vals)

        if std > 45:
            return "Highly Expressive"
        elif std > 25:
            return "Moderate Variation"
        return "Flat / Formal Tone"

    except:
        return "Analysis Unavailable"

# -----------------------------
# 8. TOKENIZATION (SAFE)
# -----------------------------
def tokenize(text):
    return re.findall(r"[a-zA-ZÀ-ÿ']+", text.lower())

# -----------------------------
# 9. SBERT SCORING (MULTI-CLASS FIXED)
# -----------------------------
def sbert_similarity(text):
    if not proto_embeddings:
        return {p: 1 / len(provinces) for p in provinces}

    emb_input = sbert_model.encode(text, convert_to_tensor=True)

    scores = {}

    for prov in provinces:
        if proto_embeddings[prov] is None:
            scores[prov] = 0.0
            continue

        # IMPORTANT FIX: use MAX similarity (not mean)
        scores[prov] = util.cos_sim(
            emb_input,
            proto_embeddings[prov]
        ).max().item()

    total = sum(scores.values())

    if total == 0:
        return {p: 1 / len(provinces) for p in provinces}

    return {k: v / total for k, v in scores.items()}

# -----------------------------
# 10. RULE ENGINE (MULTI-PROVINCE SAFE)
# -----------------------------
def rule_engine(text):
    tokens = tokenize(text)

    scores = {p: 0 for p in provinces}
    detected = []
    risks = []

    if risk_df.empty:
        return scores, detected, risks

    for _, row in risk_df.iterrows():
        token = str(row["token"]).lower().strip()
        tag = str(row.get("dialect_tag", "")).lower()

        if token in tokens:
            detected.append(token)

            if tag in scores:
                scores[tag] += 1

            risks.append({
                "word": token,
                "category": row.get("category", ""),
                "meaning": row.get("meaning", ""),
                "tag": tag
            })

    return scores, detected, risks

# -----------------------------
# 11. FUSION ENGINE
# -----------------------------
def process_text_analysis(text):

    rule_scores, markers, risks = rule_engine(text)
    sbert_scores = sbert_similarity(text)

    final_scores = {}

    for p in provinces:
        final_scores[p] = (
            rule_scores[p] * 0.6 +
            sbert_scores[p] * 0.4
        )

    best = max(final_scores, key=final_scores.get)
    confidence = final_scores[best] * 100

    return (
        f"{best.capitalize()} Dialect",
        round(confidence, 2),
        markers,
        risks
    )

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():

    audio_path = os.path.join(UPLOAD_FOLDER, "audio.wav")
    audio = request.files.get("audio")
    use_noise = request.form.get("noise_suppression") == "true"

    if not audio:
        return jsonify({"error": "No audio"}), 400

    try:
        audio.save(audio_path)

        if use_noise:
            clean_audio(audio_path)

        result = whisper_model.transcribe(
            audio_path,
            fp16=(device == "cuda"),
            language=None,
            beam_size=5
        )

        text = result["text"].strip()
        intonation = analyze_intonation(audio_path)

        return jsonify({
            "original_text": text,
            "intonation": intonation
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.get_json()
    text = data.get("conversation", "")
    intonation = data.get("intonation", "Unknown")

    if not text:
        return jsonify({"explanation": "Empty text."})

    dialect, confidence, markers, risks = process_text_analysis(text)

    report = [
        f"<div><strong>Sentence:</strong> {text}</div>",
        f"<div><strong>Speech:</strong> {intonation}</div>",
        f"<div><strong>Detected:</strong> <span style='color:green'>{dialect}</span></div>",
        f"<div><strong>Confidence:</strong> {confidence:.2f}%</div>",
        "<hr>"
    ]

    if risks:
        report.append("<div><strong>Evidence:</strong></div>")
        for r in risks:
            report.append(
                f"<div style='margin-bottom:8px;border-left:3px solid #3498db;padding-left:6px;'>"
                f"<b>{r['word']}</b> ({r['category']}) [{r['tag']}]"
                f"</div>"
            )
    else:
        report.append("<div>No markers detected.</div>")

    return jsonify({"explanation": "".join(report)})

# -----------------------------
if __name__ == "__main__":
    app.run(debug=False, port=8080)