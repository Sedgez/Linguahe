from flask import Flask, render_template, request, jsonify, send_file
import whisper
import os
import re
import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr
import pandas as pd
import torch
import uuid
import joblib
import io
from gtts import gTTS
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# -----------------------------
# DIRECTORY SETUP
# -----------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# DEVICE INITIALIZATION
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n=== SYSTEM DEVICE: {device.upper()} ===")

# -----------------------------
# MODEL INITIALIZATIONS
# -----------------------------
print("--- Loading Accent ANN Models ---")
try:
    accent_model = load_model("accent_ai/models/accent_ann.keras")
    accent_scaler = joblib.load("accent_ai/models/scaler.pkl")
    accent_encoder = joblib.load("accent_ai/models/label_encoder.pkl")
    print("Accent ANN Loaded Successfully")
except Exception as e:
    print(f"Warning: Failed to load Accent ANN models: {e}")
    accent_model, accent_scaler, accent_encoder = None, None, None

print("--- Initializing Whisper Model ---")
try:
    whisper_model = whisper.load_model("turbo", device=device)
    print("Loaded Whisper Turbo Model")
except Exception as e:
    print(f"Turbo unavailable -> fallback to medium: {e}")
    whisper_model = whisper.load_model("medium", device=device)

print("--- Loading Multilingual SBERT Model ---")
sbert_model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2",
    device=device
)

# -----------------------------
# TARGET RESEARCH PROVINCES
# -----------------------------
provinces = ["batangas", "laguna", "cavite", "rizal", "quezon"]

# -----------------------------
# SAFE DATA LOADERS
# -----------------------------
def safe_read_csv(path):
    encodings = ["utf-8", "cp1252", "latin-1"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except:
            continue
    return pd.DataFrame()

def load_reference_sentences():
    path = "reference_sentences.csv"
    if not os.path.exists(path):
        print("reference_sentences.csv not found")
        return pd.DataFrame()

    df = safe_read_csv(path)
    if df.empty:
        return df

    df.columns = df.columns.str.strip().str.lower()
    required_cols = ["dialect", "sentence"]

    for col in required_cols:
        if col not in df.columns:
            raise Exception(f"Missing required column: {col}")

    df["dialect"] = df["dialect"].astype(str).str.lower().str.strip()
    df["sentence"] = df["sentence"].astype(str).str.strip()
    return df

def load_risk_words():
    path = "risk_words.csv"
    if not os.path.exists(path):
        print("risk_words.csv not found")
        return pd.DataFrame()

    df = safe_read_csv(path)
    if df.empty:
        return df

    df.columns = df.columns.str.strip().str.lower()
    
    if "dielect_tag" in df.columns and "dialect_tag" not in df.columns:
        df = df.rename(columns={"dielect_tag": "dialect_tag"})
    if "context_meaning" in df.columns and "meaning" not in df.columns:
        df = df.rename(columns={"context_meaning": "meaning"})

    if "token" not in df.columns:
        raise Exception("Missing required column: token")

    return df.fillna("")

proto_df = load_reference_sentences()
risk_df = load_risk_words()

# -----------------------------
# PRECOMPUTE SBERT EMBEDDINGS
# -----------------------------
print("--- Building Province Embeddings ---")
proto_embeddings = {}

if not proto_df.empty:
    with torch.no_grad():
        for prov in provinces:
            sentences = proto_df[proto_df["dialect"] == prov]["sentence"].tolist()
            if len(sentences) == 0:
                proto_embeddings[prov] = None
                continue

            embeddings = sbert_model.encode(
                sentences,
                convert_to_tensor=True,
                device=device
            )
            proto_embeddings[prov] = embeddings
print("Province embeddings complete")

# -----------------------------
# AUDIO PROCESSING PIPELINE (OPTIMIZED)
# -----------------------------
def process_audio_pipeline(audio_path, use_noise=False):

    y, sr = librosa.load(audio_path, sr=16000)
    if len(y) == 0:
        return y, sr

    y = librosa.util.normalize(y)

    if use_noise and len(y) > sr * 0.5:  # Run only if audio is longer than 0.5s
        try:
            y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.7)
        except Exception as e:
            print(f"Noise reduction skipped: {e}")

    y, _ = librosa.effects.trim(y, top_db=20)
 
    sf.write(audio_path, y, sr)
    return y, sr

# -----------------------------
# FEATURE EXTRACTION ENGINE
# -----------------------------
def predict_accent_in_memory(y_16k, sr_16k):
    if accent_model is None:
        return "Model Unavailable", 0.0

    if len(y_16k) == 0:
        return "Unknown (Empty Audio)", 0.0

    # Resample in-memory from 16kHz to 22050Hz to match standard trained features
    y = librosa.resample(y_16k, orig_sr=sr_16k, target_sr=22050)
    sr = 22050

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    features = np.concatenate([mfcc, [zcr], [rms], [centroid], [bandwidth], [rolloff]]).reshape(1, -1)
    features = accent_scaler.transform(features)

    prediction = accent_model.predict(features, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)
    accent = accent_encoder.inverse_transform([predicted_class])[0]

    return accent, confidence

def analyze_audio_features_in_memory(y, sr):
    if len(y) == 0:
        return {
            "duration": 0, "speech_rate": 0, "intonation": "Analysis Unavailable",
            "pitch_variation": 0, "emotion": "Unknown"
        }

    try:
        duration = librosa.get_duration(y=y, sr=sr)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        
        if magnitudes.size > 0 and np.max(magnitudes) > 0:
            pitch_values = pitches[magnitudes > np.median(magnitudes)]
            pitch_values = pitch_values[pitch_values > 0]
        else:
            pitch_values = np.array([])

        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        speech_rate = len(onset_frames) / max(duration, 1)

        if len(pitch_values) < 5:
            pitch_std = 0
            intonation = "Flat / Neutral"
        else:
            pitch_std = np.std(pitch_values)
            if pitch_std > 45:
                intonation = "Highly Expressive"
            elif pitch_std > 25:
                intonation = "Moderate Variation"
            else:
                intonation = "Flat / Formal Tone"

        if speech_rate > 4 and pitch_std > 40:
            emotion = "Excited / Energetic"
        elif pitch_std < 15:
            emotion = "Serious / Calm"
        else:
            emotion = "Neutral Conversational"

        return {
            "duration": round(duration, 2),
            "speech_rate": round(speech_rate, 2),
            "intonation": intonation,
            "pitch_variation": round(float(pitch_std), 2),
            "emotion": emotion
        }
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return {
            "duration": 0, "speech_rate": 0, "intonation": "Analysis Error",
            "pitch_variation": 0, "emotion": "Unknown"
        }

# -----------------------------
# DIALECT SIMILARITY ENGINES
# -----------------------------
def sbert_similarity(text):
    if not proto_embeddings:
        return {p: 1 / len(provinces) for p in provinces}

    with torch.no_grad():
        emb_input = sbert_model.encode(text, convert_to_tensor=True, device=device)
        scores = {}

        for prov in provinces:
            emb = proto_embeddings.get(prov)
            if Play := (emb is None):
                scores[prov] = 0.0
                continue
            similarity = util.cos_sim(emb_input, emb)
            scores[prov] = similarity.max().item()

    total = sum(scores.values())
    if total == 0:
        return {p: 1 / len(provinces) for p in provinces}

    return {k: v / total for k, v in scores.items()}

def rule_engine(text):
    text_lower = text.lower().strip()
    scores = {p: 0 for p in provinces}
    detected, risks = [], []

    if risk_df.empty:
        return scores, detected, risks

    risk_tokens = sorted(risk_df["token"].unique(), key=len, reverse=True)
    working_text = text_lower

    for token in risk_tokens:
        pattern = r"\b" + re.escape(token) + r"\b"
        if re.search(pattern, working_text):
            detected.append(token)
            matches = risk_df[risk_df["token"] == token]

            for _, row in matches.iterrows():
                dialect_tag = str(row.get("dialect_tag", "")).lower().strip()
                if dialect_tag in scores:
                    scores[dialect_tag] += 1

                risks.append({
                    "word": token,
                    "category": row.get("category", ""),
                    "meaning": row.get("meaning", ""),
                    "tag": dialect_tag
                })
            working_text = re.sub(pattern, " ", working_text)

    return scores, detected, risks

def process_text_analysis(text):
    rule_scores, markers, risks = rule_engine(text)
    sbert_scores = sbert_similarity(text)
    final_scores = {}

    for province in provinces:
        final_scores[province] = (rule_scores[province] * 0.6) + (sbert_scores[province] * 0.4)

    total_sum = sum(final_scores.values())
    if total_sum > 0:
        for province in final_scores:
            final_scores[province] /= total_sum
    else:
        final_scores = {p: 1 / len(provinces) for p in provinces}

    best_match = max(final_scores, key=final_scores.get)
    confidence = final_scores[best_match] * 100
    normalized_scores = {k: round(v * 100, 2) for k, v in final_scores.items()}

    return {
        "dialect": f"{best_match.capitalize()} Dialect",
        "confidence": round(confidence, 2),
        "markers": markers,
        "risks": risks,
        "scores": normalized_scores
    }

# -----------------------------
# CORE APP ROUTING
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio = request.files.get("audio")
    use_noise = request.form.get("noise_suppression") == "true"

    if not audio:
        return jsonify({"error": "No audio uploaded"}), 400

    unique_name = f"{uuid.uuid4()}.wav"
    audio_path = os.path.join(UPLOAD_FOLDER, unique_name)

    try:
        audio.save(audio_path)

        # Process, normalize and trim the audio once in disk and cache array in RAM
        y_cached, sr_cached = process_audio_pipeline(audio_path, use_noise=use_noise)

        # Run model transcriptions under thread safe evaluation configurations
        with torch.no_grad():
            result = whisper_model.transcribe(
                audio_path,
                fp16=(device == "cuda"),
                beam_size=5,
                language="tl"
            )
        text = result["text"].strip()

        # Execute subsequent downstream models entirely from RAM cache
        accent, accent_confidence = predict_accent_in_memory(y_cached, sr_cached)
        features = analyze_audio_features_in_memory(y_cached, sr_cached)

        return jsonify({
            "original_text": text,
            "accent": accent,
            "accent_confidence": round(accent_confidence, 2),
            "duration_seconds": features["duration"],
            "speech_rate": features["speech_rate"],
            "intonation": features["intonation"],
            "pitch_variation": features["pitch_variation"],
            "emotion": features["emotion"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json() or {}
    text = data.get("conversation", "").strip()
    intonation_input = data.get("intonation", "Unknown")
    emotion_input = data.get("Emotional", "Normal")
    speech_rate = data.get("speech_rate", 0)

    if not text:
        return jsonify({"explanation": "<div style='color:#64748b;'>No text received.</div>"})

    result = process_text_analysis(text)
    dialect = result["dialect"]
    confidence = result["confidence"]
    risks = result["risks"]
    scores = result["scores"]
    markers = result["markers"]

    word_count = len(text.split())
    verbal_style = "Formal / Structured" if word_count > 18 else "Casual Conversational"

    # --- Structured String Building for UI Report Component ---
    diction_html = f"""
    <div style='margin-bottom:16px; background:#eff6ff; border-left:5px solid #0284c7; padding:12px; border-radius:8px;'>
        <div style='font-weight:bold; color:#0369a1; margin-bottom:6px;'>Diction Patterns</div>
        <div style='font-size:14px; color:#334155;'>
            • Words count: <strong>{word_count}</strong> words detected<br>
            • Regional Marker Count: <strong>{len(markers)}</strong> dialect-sensitive expressions<br>
            • Verbal Style: <strong>{verbal_style}</strong>
        </div>
    </div>
    """

    sentence_particles = ["ba", "naman", "eh", "nga", "po", "ho", "diba", "kasi"]
    tokens = text.lower().split()
    particles_found = [p for p in sentence_particles if p in tokens]
    particle_display = ", ".join(particles_found) if particles_found else "No particles detected"

    prosody_html = f"""
    <div style='margin-bottom:16px; background:#fefce8; border-left:5px solid #ca8a04; padding:12px; border-radius:8px;'>
        <div style='font-weight:bold; color:#a16207; margin-bottom:6px;'>Intonation Markers and Prosodic Signals</div>
        <div style='font-size:14px; color:#334155;'>
            • Sentence-Final Particles: <strong>{particle_display}</strong><br>
            • Prosodic Signal: <strong>{intonation_input}</strong><br>
        </div>
    </div>
    """

    emotion_html = f"""
    <div style='margin-bottom:16px; background:#fdf2f8; border-left:5px solid #db2777; padding:12px; border-radius:8px;'>
        <div style='font-weight:bold; color:#be185d; margin-bottom:6px;'>Emotional Expression</div>
        <div style='font-size:14px; color:#334155;'>
            • Detected Emotional Delivery: <strong>{emotion_input}</strong>
        </div>
    </div>
    """

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranking_html = "".join([
        f"<div style='margin-bottom:6px;'>{prov.capitalize()}: <strong>{score:.2f}%</strong></div>"
        for prov, score in sorted_scores
    ])

    report = [
        diction_html, prosody_html, emotion_html,
        f"<div style='margin-bottom:12px;'><strong>Closest Province/Region:</strong> <span style='color:#0284c7; font-weight:bold;'>{dialect}</span></div>",
        f"<div style='margin-bottom:16px;'><strong>System Confidence Level:</strong> <strong>{confidence:.2f}%</strong></div>",
        "<div style='margin-bottom:12px;'><strong>Province Similarity Ranking</strong></div>",
        ranking_html,
        "<hr style='border:0; border-top:1px solid #e2e8f0; margin:16px 0;'>"
    ]

    if risks:
        report.append("<div style='font-weight:bold; margin-bottom:10px; color:#e11d48;'>Flagged Regional Risk Words</div>")
        for risk in risks:
            report.append(f"""
            <div style='margin-bottom:10px; border-left:4px solid #e11d48; padding:8px; background:#fff1f2; border-radius:6px;'>
                • <strong>{risk.get('word', 'unknown')}</strong> ({risk.get('category', 'Uncategorized')})<br>
                <small style='color:#475569;'>Meaning: {risk.get('meaning', 'No meaning metadata')} [{risk.get('tag', 'UNK').upper()}]</small>
            </div>
            """)
    else:
        report.append("<div style='color:#64748b; font-style:italic;'>No regional misunderstanding or dialect-sensitive terms detected.</div>")

    return jsonify({
        "explanation": "".join(report),
        "raw_data": {
            "dialect": dialect,
            "confidence": confidence,
            "scores": scores,
            "risks": risks
        }
    })

# -----------------------------
# GOOGLE TEXT-TO-SPEECH STREAM ROUTE
# -----------------------------
@app.route("/tts", methods=["POST"])
def text_to_speech():
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
        
    try:
        # Generate speech and stream it back directly via an in-memory byte buffer
        fp = io.BytesIO()
        tts = gTTS(text=text, lang="tl")
        tts.write_to_fp(fp)
        fp.seek(0)
        
        return send_file(fp, mimetype="audio/mp3", as_attachment=False)
    except Exception as e:
        return jsonify({"error": f"TTS Processing Exception: {str(e)}"}), 500

# -----------------------------
# APPLICATION ENTRY
# -----------------------------
if __name__ == "__main__":
    app.run(
        debug=False,
        host="0.0.0.0",
        port=8080
    )