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
import uuid
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
# WHISPER MODEL INITIALIZATION
# -----------------------------
print("--- Initializing Whisper Model ---")

try:
    whisper_model = whisper.load_model("turbo", device=device)
    print("Loaded Whisper Turbo Model")
except Exception as e:
    print(f"Turbo unavailable -> fallback to medium: {e}")
    whisper_model = whisper.load_model("medium", device=device)

# -----------------------------
# SBERT MODEL INITIALIZATION
# -----------------------------
print("--- Loading Multilingual SBERT Model ---")

sbert_model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2",
    device=device
)

# -----------------------------
# TARGET RESEARCH PROVINCES
# -----------------------------
provinces = [
    "batangas",
    "laguna",
    "cavite",
    "rizal",
    "quezon"
]

# -----------------------------
# SAFE CSV LOADER
# -----------------------------
def safe_read_csv(path):

    encodings = ["utf-8", "cp1252", "latin-1"]

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except:
            continue

    return pd.DataFrame()

# -----------------------------
# LOAD SBERT PROTOTYPE DATABASE
# -----------------------------
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

    df["dialect"] = (
        df["dialect"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["sentence"] = (
        df["sentence"]
        .astype(str)
        .str.strip()
    )

    return df

proto_df = load_reference_sentences()

# -----------------------------
# PRECOMPUTE SBERT EMBEDDINGS
# -----------------------------
print("--- Building Province Embeddings ---")

proto_embeddings = {}

if not proto_df.empty:

    for prov in provinces:

        sentences = proto_df[
            proto_df["dialect"] == prov
        ]["sentence"].tolist()

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
# LOAD RISK WORD DATABASE
# -----------------------------
def load_risk_words():

    path = "risk_words.csv"

    if not os.path.exists(path):
        print("risk_words.csv not found")
        return pd.DataFrame()

    df = safe_read_csv(path)

    if df.empty:
        return df

    df.columns = df.columns.str.strip().str.lower()

    # Header auto-fixes
    if "dielect_tag" in df.columns and "dialect_tag" not in df.columns:
        df = df.rename(columns={
            "dielect_tag": "dialect_tag"
        })

    if "context_meaning" in df.columns and "meaning" not in df.columns:
        df = df.rename(columns={
            "context_meaning": "meaning"
        })

    required = ["token"]

    for col in required:
        if col not in df.columns:
            raise Exception(f"Missing required column: {col}")

    return df.fillna("")

risk_df = load_risk_words()

# -----------------------------
# AUDIO CLEANING PIPELINE
# -----------------------------
def clean_audio(audio_path):

    try:
        y, sr = librosa.load(audio_path, sr=16000)

        y = librosa.util.normalize(y)

        y = nr.reduce_noise(
            y=y,
            sr=sr,
            prop_decrease=0.7
        )

        y, _ = librosa.effects.trim(
            y,
            top_db=20
        )

        sf.write(audio_path, y, sr)

    except Exception as e:
        print(f"Audio cleaning error: {e}")

# -----------------------------
# AUDIO FEATURE EXTRACTION
# -----------------------------
def analyze_audio_features(audio_path):

    try:
        y, sr = librosa.load(audio_path, sr=16000)

        duration = librosa.get_duration(y=y, sr=sr)

        # Pitch extraction
        pitches, magnitudes = librosa.piptrack(
            y=y,
            sr=sr
        )

        pitch_values = pitches[
            magnitudes > np.median(magnitudes)
        ]

        pitch_values = pitch_values[pitch_values > 0]

        # Speech rate
        onset_frames = librosa.onset.onset_detect(
            y=y,
            sr=sr
        )

        speech_rate = len(onset_frames) / max(duration, 1)

        # Pitch variation
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

        # Emotion approximation
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
            "duration": 0,
            "speech_rate": 0,
            "intonation": "Analysis Unavailable",
            "pitch_variation": 0,
            "emotion": "Unknown"
        }

# -----------------------------
# TOKENIZATION
# -----------------------------
def tokenize(text):

    return re.findall(
        r"[a-zA-ZÀ-ÿ']+",
        text.lower()
    )

# -----------------------------
# SBERT SIMILARITY ENGINE
# -----------------------------
def sbert_similarity(text):

    if not proto_embeddings:
        return {
            p: 1 / len(provinces)
            for p in provinces
        }

    emb_input = sbert_model.encode(
        text,
        convert_to_tensor=True,
        device=device
    )

    scores = {}

    for prov in provinces:

        emb = proto_embeddings.get(prov)

        if emb is None:
            scores[prov] = 0.0
            continue

        similarity = util.cos_sim(
            emb_input,
            emb
        )

        scores[prov] = similarity.max().item()

    total = sum(scores.values())

    if total == 0:
        return {
            p: 1 / len(provinces)
            for p in provinces
        }

    return {
        k: v / total
        for k, v in scores.items()
    }

# -----------------------------
# RULE ENGINE
# -----------------------------
def rule_engine(text):

    text_lower = text.lower().strip()

    scores = {p: 0 for p in provinces}

    detected = []

    risks = []

    if risk_df.empty:
        return scores, detected, risks

    risk_tokens = sorted(
        risk_df["token"].unique(),
        key=len,
        reverse=True
    )

    working_text = text_lower

    for token in risk_tokens:

        pattern = r"\b" + re.escape(token) + r"\b"

        if re.search(pattern, working_text):

            detected.append(token)

            matches = risk_df[
                risk_df["token"] == token
            ]

            for _, row in matches.iterrows():

                dialect_tag = str(
                    row.get("dialect_tag", "")
                ).lower().strip()

                if dialect_tag in scores:
                    scores[dialect_tag] += 1

                risks.append({
                    "word": token,
                    "category": row.get("category", ""),
                    "meaning": row.get("meaning", ""),
                    "tag": dialect_tag
                })

            working_text = re.sub(
                pattern,
                " ",
                working_text
            )

    return scores, detected, risks

# -----------------------------
# HYBRID ANALYSIS ENGINE
# -----------------------------
def process_text_analysis(text):

    rule_scores, markers, risks = rule_engine(text)

    sbert_scores = sbert_similarity(text)

    final_scores = {}

    for province in provinces:

        final_scores[province] = (
            (rule_scores[province] * 0.6) +
            (sbert_scores[province] * 0.4)
        )

    total_sum = sum(final_scores.values())

    if total_sum > 0:
        for province in final_scores:
            final_scores[province] /= total_sum
    else:
        final_scores = {
            p: 1 / len(provinces)
            for p in provinces
        }

    best_match = max(
        final_scores,
        key=final_scores.get
    )

    confidence = (
        final_scores[best_match] * 100
    )

    normalized_scores = {
        k: round(v * 100, 2)
        for k, v in final_scores.items()
    }

    return {
        "dialect": f"{best_match.capitalize()} Dialect",
        "confidence": round(confidence, 2),
        "markers": markers,
        "risks": risks,
        "scores": normalized_scores
    }

# -----------------------------
# ROOT ROUTE
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------
# TRANSCRIBE ROUTE
# -----------------------------
@app.route("/transcribe", methods=["POST"])
def transcribe():

    audio = request.files.get("audio")

    use_noise = (
        request.form.get("noise_suppression")
        == "true"
    )

    if not audio:
        return jsonify({
            "error": "No audio uploaded"
        }), 400

    unique_name = f"{uuid.uuid4()}.wav"

    audio_path = os.path.join(
        UPLOAD_FOLDER,
        unique_name
    )

    try:

        audio.save(audio_path)

        if use_noise:
            clean_audio(audio_path)

        result = whisper_model.transcribe(
            audio_path,
            fp16=(device == "cuda"),
            beam_size=5,
            language="tl"
        )

        text = result["text"].strip()

        features = analyze_audio_features(audio_path)

        return jsonify({
            "original_text": text,
            "duration_seconds": features["duration"],
            "speech_rate": features["speech_rate"],
            "intonation": features["intonation"],
            "pitch_variation": features["pitch_variation"],
            "emotion": features["emotion"]
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

    finally:

        if os.path.exists(audio_path):
            os.remove(audio_path)

# -----------------------------
# ANALYZE ROUTE
# -----------------------------
@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.get_json()

    text = data.get("conversation", "").strip()

    intonation_input = data.get(
        "intonation",
        "Unknown"
    )

    emotion_input = data.get(
        "emotion",
        "Unknown"
    )

    speech_rate = data.get(
        "speech_rate",
        0
    )

    if not text:

        return jsonify({
            "explanation":
            "<div style='color:#64748b;'>No text received.</div>"
        })

    result = process_text_analysis(text)

    dialect = result["dialect"]

    confidence = result["confidence"]

    risks = result["risks"]

    scores = result["scores"]

    markers = result["markers"]

    # -----------------------------
    # DICTION PATTERN ANALYSIS
    # -----------------------------
    word_count = len(text.split())

    if word_count > 18:
        verbal_style = "Formal / Structured"
    else:
        verbal_style = "Casual Conversational"

    diction_html = f"""
    <div style='
        margin-bottom:16px;
        background:#eff6ff;
        border-left:5px solid #0284c7;
        padding:12px;
        border-radius:8px;
    '>

        <div style='
            font-weight:bold;
            color:#0369a1;
            margin-bottom:6px;
        '>
            Diction Patterns in Verbal Communication
        </div>

        <div style='font-size:14px;color:#334155;'>

            • Lexical Density:
            <strong>{word_count}</strong>
            words detected
            <br>

            • Regional Marker Count:
            <strong>{len(markers)}</strong>
            dialect-sensitive expressions
            <br>

            • Verbal Style:
            <strong>{verbal_style}</strong>

        </div>

    </div>
    """

    # -----------------------------
    # SENTENCE FINAL PARTICLES
    # -----------------------------
    particles_found = []

    sentence_particles = [
        "ba",
        "naman",
        "eh",
        "nga",
        "po",
        "ho",
        "diba",
        "kasi"
    ]

    tokens = text.lower().split()

    for particle in sentence_particles:

        if particle in tokens:
            particles_found.append(particle)

    particle_display = (
        ", ".join(particles_found)
        if particles_found
        else "No major particles detected"
    )

    # -----------------------------
    # PROSODIC ANALYSIS
    # -----------------------------
    prosody_html = f"""
    <div style='
        margin-bottom:16px;
        background:#fefce8;
        border-left:5px solid #ca8a04;
        padding:12px;
        border-radius:8px;
    '>

        <div style='
            font-weight:bold;
            color:#a16207;
            margin-bottom:6px;
        '>
            Intonation Markers and Prosodic Signals
        </div>

        <div style='font-size:14px;color:#334155;'>

            • Sentence-Final Particles:
            <strong>{particle_display}</strong>
            <br>

            • Prosodic Signal:
            <strong>{intonation_input}</strong>
            <br>

            • Speech Rhythm:
            <strong>{speech_rate}</strong>
            rhythm units/sec
            <br>

            • Pitch Interpretation:
            <strong>
                {
                    "Dynamic Speech Flow"
                    if "Highly" in intonation_input
                    else "Moderate Conversational Rhythm"
                }
            </strong>

        </div>

    </div>
    """

    # -----------------------------
    # EMOTIONAL EXPRESSION PANEL
    # -----------------------------
    emotion_html = f"""
    <div style='
        margin-bottom:16px;
        background:#fdf2f8;
        border-left:5px solid #db2777;
        padding:12px;
        border-radius:8px;
    '>

        <div style='
            font-weight:bold;
            color:#be185d;
            margin-bottom:6px;
        '>
            Emotional Expression in Spoken Interaction
        </div>

        <div style='font-size:14px;color:#334155;'>

            • Detected Emotional Delivery:
            <strong>{emotion_input}</strong>

        </div>

    </div>
    """

    # -----------------------------
    # PROVINCE RANKINGS
    # -----------------------------
    ranking_html = ""

    sorted_scores = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for province, score in sorted_scores:

        ranking_html += (
            f"<div style='margin-bottom:6px;'>"
            f"{province.capitalize()}: "
            f"<strong>{score:.2f}%</strong>"
            f"</div>"
        )

    # -----------------------------
    # MAIN REPORT
    # -----------------------------
    report = [

        diction_html,

        prosody_html,

        emotion_html,

        f"""
        <div style='margin-bottom:12px;'>

            <strong>Closest Province/Region:</strong>

            <span style='
                color:#0284c7;
                font-weight:bold;
            '>
                {dialect}
            </span>

        </div>
        """,

        f"""
        <div style='margin-bottom:16px;'>

            <strong>System Confidence Level:</strong>

            <span style='font-weight:bold;'>
                {confidence:.2f}%
            </span>

        </div>
        """,

        """
        <div style='margin-bottom:12px;'>
            <strong>Province Similarity Ranking</strong>
        </div>
        """,

        ranking_html,

        "<hr style='border:0;border-top:1px solid #e2e8f0;margin:16px 0;'>"
    ]

    # -----------------------------
    # RISK WORDS
    # -----------------------------
    if risks:

        report.append(
            """
            <div style='
                font-weight:bold;
                margin-bottom:10px;
                color:#e11d48;
            '>

                Flagged Regional Risk Words

            </div>
            """
        )

        for risk in risks:

            word = risk.get(
                "word",
                "unknown"
            )

            category = risk.get(
                "category",
                "Uncategorized"
            )

            meaning = risk.get(
                "meaning",
                "No meaning metadata"
            )

            tag = risk.get(
                "tag",
                "UNK"
            ).upper()

            report.append(
                f"""
                <div style='
                    margin-bottom:10px;
                    border-left:4px solid #e11d48;
                    padding:8px;
                    background:#fff1f2;
                    border-radius:6px;
                '>

                    • <strong>{word}</strong>
                    ({category})<br>

                    <small style='color:#475569;'>

                        Meaning:
                        {meaning}
                        [{tag}]

                    </small>

                </div>
                """
            )

    else:

        report.append(
            """
            <div style='
                color:#64748b;
                font-style:italic;
            '>

                No regional misunderstanding or
                dialect-sensitive terms detected.

            </div>
            """
        )

    return jsonify({
        "explanation": "".join(report)
    })

# -----------------------------
# APPLICATION ENTRY
# -----------------------------
if __name__ == "__main__":

    app.run(
        debug=False,
        host="0.0.0.0",
        port=8080
    )