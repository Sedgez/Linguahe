from flask import Flask, render_template, request, jsonify
import whisper
import os
from transformers import pipeline
import pandas as pd
import torch
import re
import noisereduce as nr
import librosa
import soundfile as sf
import numpy as np
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
REPORTS_FOLDER = "reports"

for folder in [UPLOAD_FOLDER, REPORTS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# --- Load AI Models ---
print("--- Loading AI Engines ---")
try:
    # Speech Engine (Using "tiny" for demo speed)
    whisper_model = whisper.load_model("base") 
except Exception as e:
    print(f"Model Load Error: {e}")
    whisper_model = None

# --- Analysis Modules ---

def analyze_intonation(audio_path):
    try:
        # sr=16000 prevents the "PySoundFile failed" warning in most cases
        y, sr = librosa.load(audio_path, sr=16000)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        pitch_values = pitch_values[pitch_values > 0] 

        if len(pitch_values) < 5:
            return "Flat / Neutral"

        pitch_std = np.std(pitch_values)
        if pitch_std > 45: return "Highly Expressive"
        elif pitch_std > 25: return "Moderate Variation"
        else: return "Flat / Formal Tone"
    except:
        return "Analysis Unavailable"

def detect_batangas(text):
    # Added common Batangas markers for better detection
    markers = ["ala eh", "ga", "ba ga", "dine", "nakain", "naulan", "mabanas", "liban"]
    return any(m in text.lower() for m in markers)

# --- Routes ---

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, "audio.wav"))
    if 'audio' not in request.files:
        return jsonify({"error": "No audio"}), 400

    audio = request.files["audio"]
    use_noise = request.form.get("noise_suppression") == "true"

    try:
        audio.save(audio_path)
        if use_noise:
            y, sr = librosa.load(audio_path, sr=None)
            reduced = nr.reduce_noise(y=y, sr=sr)
            sf.write(audio_path, reduced, sr)

        result = whisper_model.transcribe(audio_path, fp16=False, language="tl")
        text = result["text"].strip()

        # Acoustic extraction for punto
        intonation = analyze_intonation(audio_path)

        return jsonify({
            "original_text": text,
            "intonation": intonation
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Keep file temporarily for analysis if needed, but usually safe to clean
        pass

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("conversation", "").lower()
    intonation = data.get("intonation", "Unknown")

    if not text:
        return jsonify({"explanation": "No speech detected."})

    # 1. Dialect Identification
    dialect = "Laguna Tagalog"
    if detect_batangas(text):
        dialect = "Batangas Dialect Detected"

    # 2. Misunderstood Words Logic (Cross-Cultural Focus)
    risk_map = {
        "nakain": "Southern Tagalog: 'Eating' | Manila: 'Being eaten'.",
        "liban": "Regional: 'To cross' | Standard: 'To be absent'.",
        "ga": "Regional question marker; changes sentence intent.",
        "mabanas": "Regional term for 'hot/humid'; often misunderstood as 'irritated'.",
        "dine": "Regional for 'here'; confusing for non-native speakers."
    }
    
    found_risks = [f"<strong>{word}</strong>: {desc}" for word, desc in risk_map.items() if word in text]

    # 3. Build Simplified Report
    report = [
        f"<div><strong>Tone / Diction:</strong> {intonation}</div>",
        f"<div><strong>Detected Dialect:</strong> {dialect}</div>",
        "<hr>"
    ]

    if found_risks:
        report.append("<div><strong>Possible Misunderstood Words:</strong></div>")
        for risk in found_risks:
            report.append(f"<div style='color: #ef4444; font-size: 0.85rem; margin-top: 5px;'>• {risk}</div>")
    else:
        report.append("<div style='color: #6b7280; font-style: italic;'>No high-risk regional markers detected.</div>")

    return jsonify({"explanation": "".join(report)})

if __name__ == "__main__":
    # Use port 8080 as per your previous setup
    app.run(debug=False, port=8080)