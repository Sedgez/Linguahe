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

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 1. LOAD WHISPER (TURBO)
# -----------------------------
print(f"--- Initializing Whisper Turbo on {device} ---")
try:
    whisper_model = whisper.load_model("turbo", device=device)
except Exception as e:
    print(f"Turbo failed, using medium: {e}")
    whisper_model = whisper.load_model("medium", device=device)

# -----------------------------
# 2. LOAD SBERT (MULTILINGUAL)
# -----------------------------
print("--- Loading SBERT ---")
sbert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# -----------------------------
# 3. LOAD SBERT PROTOTYPES
# -----------------------------
def load_sbert_prototypes():
    path = "reference_sentences.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        return df
    return pd.DataFrame()

proto_df = load_sbert_prototypes()

# Split prototypes
batangas_proto = proto_df[proto_df["dialect"] == "batangas"]["sentence"].tolist()
laguna_proto = proto_df[proto_df["dialect"] == "laguna"]["sentence"].tolist()

# Precompute embeddings (FAST)
batangas_embeddings = sbert_model.encode(batangas_proto, convert_to_tensor=True) if batangas_proto else None
laguna_embeddings = sbert_model.encode(laguna_proto, convert_to_tensor=True) if laguna_proto else None

# -----------------------------
# 4. LOAD RISK CSV
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
# 5. AUDIO CLEANING
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
# 6. INTONATION ANALYSIS
# -----------------------------
def analyze_intonation(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        pitch_values = pitch_values[pitch_values > 0]

        if len(pitch_values) < 5:
            return "Flat / Neutral"

        pitch_std = np.std(pitch_values)

        if pitch_std > 45:
            return "Highly Expressive"
        elif pitch_std > 25:
            return "Moderate Variation"
        else:
            return "Flat / Formal Tone"
    except:
        return "Analysis Unavailable"

# -----------------------------
# 7. SBERT SIMILARITY (MULTI)
# -----------------------------
def sbert_similarity(text):
    if batangas_embeddings is None or laguna_embeddings is None:
        return 0.5, 0.5

    emb_input = sbert_model.encode(text, convert_to_tensor=True)

    sim_batangas = util.cos_sim(emb_input, batangas_embeddings).mean().item()
    sim_laguna = util.cos_sim(emb_input, laguna_embeddings).mean().item()

    total = sim_batangas + sim_laguna
    if total == 0:
        return 0.5, 0.5

    return sim_batangas / total, sim_laguna / total

# -----------------------------
# 8. RULE + SBERT FUSION
# -----------------------------
def process_text_analysis(text):
    batangas_score = 0
    laguna_score = 0
    detected_markers = []
    risks = []

    if risk_df.empty:
        return "Unknown", 0, [], []

    for _, row in risk_df.iterrows():
        token = str(row['token']).lower().strip()
        dialect_tag = str(row.get('dialect_tag', '')).lower()

        if re.search(rf'\b{token}\b', text.lower()):
            detected_markers.append(token)

            if dialect_tag == "batangas":
                batangas_score += 1
            elif dialect_tag == "laguna":
                laguna_score += 1

            risks.append({
                "word": token,
                "category": row.get("category", ""),
                "batangas": row.get("meaning_batangas", ""),
                "laguna": row.get("meaning_laguna", ""),
                "tag": dialect_tag
            })

    # Normalize rule score
    total_rule = batangas_score + laguna_score
    if total_rule == 0:
        rule_batangas = 0.5
        rule_laguna = 0.5
    else:
        rule_batangas = batangas_score / total_rule
        rule_laguna = laguna_score / total_rule

    # SBERT score
    sbert_batangas, sbert_laguna = sbert_similarity(text)

    #
    final_batangas = (rule_batangas * 0.6) + (sbert_batangas * 0.4)
    final_laguna = (rule_laguna * 0.6) + (sbert_laguna * 0.4)

    # Final decision
    if final_batangas > final_laguna:
        dialect = "Batangas Dialect"
        confidence = final_batangas * 100
    else:
        dialect = "Laguna Dialect"
        confidence = final_laguna * 100

    return dialect, round(confidence, 2), detected_markers, risks

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, "audio.wav"))
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
            language="tl",
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
        f"<div><strong>Speech Delivery:</strong> {intonation}</div>",
        f"<div><strong>Detected Dialect:</strong> <span style='color:#27ae60;'>{dialect}</span></div>",
        f"<div><strong>Confidence:</strong> {confidence:.2f}%</div>",
        "<hr>"
    ]

    if risks:
        report.append("<div><strong>Linguistic Evidence:</strong></div>")
        for r in risks:
            report.append(
                f"<div style='margin-bottom:10px; padding:5px; border-left:3px solid #3498db;'>"
                f"• <strong>{r['word'].upper()}</strong> ({r['category']})<br>"
                f"<small>Batangas: {r['batangas']} | Laguna: {r['laguna']}</small>"
                f"</div>"
            )
    else:
        report.append("<div>No markers detected.</div>")

    return jsonify({"explanation": "".join(report)})

# -----------------------------
if __name__ == "__main__":
    app.run(debug=False, port=8080)