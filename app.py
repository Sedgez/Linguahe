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

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"

# Ensure folders exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 1. LOAD WHISPER TURBO
# -----------------------------
print(f"--- Initializing Whisper Turbo on {device} ---")
try:
    whisper_model = whisper.load_model("turbo", device=device)
except Exception as e:
    print(f"Turbo failed, using medium: {e}")
    whisper_model = whisper.load_model("medium", device=device)

# -----------------------------
# 2. LOAD CSV DATABASE
# -----------------------------
def load_risk_db():
    path = "risk_words.csv"
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            # Clean columns: token, category, meaning_batangas, meaning_laguna, dialect_tag
            df.columns = df.columns.str.strip().str.lower()
            return df.fillna("")
        except Exception as e:
            print(f"CSV Load Error: {e}")
    return pd.DataFrame()

risk_df = load_risk_db()

# -----------------------------
# 3. AUDIO PRE-PROCESSING
# -----------------------------
def clean_audio(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        y = librosa.util.normalize(y)
        y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.7)
        sf.write(audio_path, y, sr)
    except Exception as e:
        print(f"Cleaning error: {e}")

# -----------------------------
# 4. INTONATION (ANN LOGIC)
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
        if pitch_std > 45: return "Highly Expressive"
        if pitch_std > 25: return "Moderate Variation"
        return "Flat / Formal Tone"
    except:
        return "Analysis Unavailable"

# -----------------------------
# 5. CORE ANALYSIS LOGIC
# -----------------------------
def process_text_analysis(text):
    """
    Combined function to handle dialect scoring and risk identification.
    """
    batangas_score = 0
    laguna_score = 0
    detected_markers = []
    risks = []

    if risk_df.empty:
        return "Unknown", [], []

    for _, row in risk_df.iterrows():
        # Match using the 'token' column from your CSV
        token = str(row['token']).lower().strip()
        tag = str(row.get('dialect_tag', '')).lower()
        
        # Regex for whole-word matching
        if re.search(rf'\b{token}\b', text.lower()):
            detected_markers.append(token)
            
            # Update statistical score
            if tag == "batangas": batangas_score += 1
            elif tag == "laguna": laguna_score += 1
            
            # Map into the Risk object for the UI
            risks.append({
                "token": token,
                "category": row.get("category", "general"),
                "batangas": row.get("meaning_batangas", ""),
                "laguna": row.get("meaning_laguna", ""),
                "tag": tag
            })

    # Decision Logic
    if batangas_score > laguna_score:
        dialect = "Batangas Dialect"
    elif laguna_score > batangas_score:
        dialect = "Laguna Dialect"
    else:
        dialect = "Neutral / Standard Tagalog"

    return dialect, detected_markers, risks

# -----------------------------
# 6. ROUTES
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, "audio.wav"))
    audio = request.files.get("audio")
    use_noise = request.form.get("noise_suppression") == "true"

    if not audio: return jsonify({"error": "No audio"}), 400

    try:
        audio.save(audio_path)
        if use_noise: clean_audio(audio_path)

        # Build prompt from CSV tokens to help Whisper's accuracy
        csv_prompt = ", ".join(risk_df['token'].head(10).tolist()) if not risk_df.empty else ""

        result = whisper_model.transcribe(
            audio_path,
            fp16=(device == "cuda"),
            language="tl",
            initial_prompt=f"ala eh, ga, dine, {csv_prompt}",
            beam_size=5
        )

        text = result["text"].strip()
        intonation = analyze_intonation(audio_path)

        return jsonify({"original_text": text, "intonation": intonation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(audio_path): os.remove(audio_path)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("conversation", "")
    intonation = data.get("intonation", "Unknown")

    if not text: return jsonify({"explanation": "Empty text."})

    # Run the combined analysis
    dialect, markers, risks = process_text_analysis(text)

    # Build HTML Report
    report = [
        f"<div style='margin-bottom:8px;'><strong>Transcript:</strong> {text}</div>",
        f"<div style='margin-bottom:8px;'><strong>Tone:</strong> {intonation}</div>",
        f"<div style='margin-bottom:15px;'><strong>Detected Dialect:</strong> <span style='color:#27ae60; font-weight:bold;'>{dialect}</span></div>",
        "<hr style='border:0; border-top:1px dotted #ccc;'>"
    ]

    if risks:
        report.append("<div style='margin-top:10px; font-weight:bold; color:#2c3e50;'>Linguistic Analysis:</div>")
        for r in risks:
            # Color code based on tag
            border_color = "#e74c3c" if r['tag'] == "batangas" else "#3498db"
            bg_color = "#fdf2f2" if r['tag'] == "batangas" else "#f2f7fd"
            
            report.append(
                f"<div style='margin: 10px 0; padding: 10px; border-left: 4px solid {border_color}; background: {bg_color}; border-radius: 4px;'>"
                f"<span style='font-weight:bold;'>• {r['token'].upper()}</span> "
                f"<span style='font-size:0.75rem; color:#7f8c8d;'>[{r['category']}]</span><br>"
                f"<div style='font-size:0.9rem; margin-top:5px;'>"
                f"<strong>Batangas:</strong> {r['batangas']}<br>"
                f"<strong>Laguna:</strong> {r['laguna']}</div>"
                f"</div>"
            )
    else:
        report.append("<div style='color:gray; font-style:italic; margin-top:10px;'>No lexical markers found.</div>")

    return jsonify({"explanation": "".join(report)})

if __name__ == "__main__":
    app.run(debug=False, port=8080)