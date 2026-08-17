import io
import os
import re
import uuid
from flask import Flask, jsonify, render_template, request, send_file
from gtts import gTTS
import joblib
import librosa
import noisereduce as nr
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import soundfile as sf
from tensorflow.keras.models import load_model
import torch
import whisper

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

print("--- Loading Emotion ANN Models ---")
try:
  emotion_model = load_model("emotion_ai/models/emotion_ann.keras")
  emotion_scaler = joblib.load("emotion_ai/models/scaler.pkl")
  emotion_encoder = joblib.load("emotion_ai/models/label_encoder.pkl")
  print("Emotion ANN Loaded Successfully")
except Exception as e:
  print(f"Warning: Failed to load Emotion ANN models: {e}")
  emotion_model, emotion_scaler, emotion_encoder = None, None, None

print("--- Initializing Whisper Model ---")
try:
  whisper_model = whisper.load_model("turbo", device=device)
  print("Loaded Whisper Turbo Model")
except Exception as e:
  print(f"Turbo unavailable -> fallback to medium: {e}")
  whisper_model = whisper.load_model("medium", device=device)

print("--- Loading Multilingual SBERT Model ---")
sbert_model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2", device=device
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
          sentences, convert_to_tensor=True, device=device
      )
      proto_embeddings[prov] = embeddings
print("Province embeddings complete")


# -----------------------------
# AUDIO PROCESSING PIPELINE
# -----------------------------
def process_audio_pipeline(audio_path, use_noise=False):
    # -----------------------------------------
    # LOAD ORIGINAL AUDIO
    # -----------------------------------------
    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    if len(y) == 0:
        print("=== EMPTY AUDIO FILE ===")
        return y, sr, False

    original_y = y.copy()

    # -----------------------------------------
    # RAW AUDIO LEVEL DETECTION
    # IMPORTANT: DO THIS BEFORE NORMALIZATION
    # -----------------------------------------
    rms = librosa.feature.rms(
        y=original_y,
        frame_length=2048,
        hop_length=512
    )[0]

    if len(rms) == 0:
        print("=== NO AUDIO FRAMES ===")
        return y, sr, False

    mean_rms = float(np.mean(rms))
    max_rms = float(np.max(rms))

    peak = float(np.max(np.abs(original_y)))

    duration = librosa.get_duration(
        y=original_y,
        sr=sr
    )

    print("\n================ AUDIO CHECK ================")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Mean RMS: {mean_rms:.6f}")
    print(f"Max RMS:  {max_rms:.6f}")
    print(f"Peak:     {peak:.6f}")
    print("=============================================\n")

    # -----------------------------------------
    # SILENCE / NO-AUDIO THRESHOLDS
    # -----------------------------------------
    MIN_MEAN_RMS = 0.003
    MIN_MAX_RMS = 0.008
    MIN_PEAK = 0.01

    if (
        mean_rms < MIN_MEAN_RMS
        or max_rms < MIN_MAX_RMS
        or peak < MIN_PEAK
    ):
        print("=== NO MEANINGFUL AUDIO DETECTED ===")
        return original_y, sr, False

    # -----------------------------------------
    # OPTIONAL NOISE REDUCTION
    # -----------------------------------------
    y = original_y.copy()

    if use_noise and len(y) > sr * 0.5:
        try:
            y = nr.reduce_noise(
                y=y,
                sr=sr,
                prop_decrease=0.7
            )
        except Exception as e:
            print(f"Noise reduction skipped: {e}")

    # -----------------------------------------
    # TRIM SILENCE
    # -----------------------------------------
    y, _ = librosa.effects.trim(
        y,
        top_db=20
    )

    if len(y) == 0:
        print("=== AUDIO EMPTY AFTER TRIMMING ===")
        return y, sr, False

    # -----------------------------------------
    # NORMALIZE ONLY AFTER AUDIO DETECTION
    # -----------------------------------------
    y = librosa.util.normalize(y)

    # -----------------------------------------
    # SAVE PROCESSED AUDIO
    # -----------------------------------------
    sf.write(audio_path, y, sr)

    print("=== AUDIO DETECTED ===")

    return y, sr, True


# -----------------------------
# FEATURE EXTRACTION ENGINE
# -----------------------------
def predict_accent_in_memory(y_16k, sr_16k):
  if accent_model is None:
    return "Model Unavailable", 0.0

  if len(y_16k) == 0:
    return "Unknown (Empty Audio)", 0.0

  y = librosa.resample(y_16k, orig_sr=sr_16k, target_sr=22050)
  sr = 22050

  mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
  zcr = np.mean(librosa.feature.zero_crossing_rate(y))
  rms = np.mean(librosa.feature.rms(y=y))
  centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
  bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
  rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

  features = np.concatenate(
      [mfcc, [zcr], [rms], [centroid], [bandwidth], [rolloff]]
  ).reshape(1, -1)
  features = accent_scaler.transform(features)

  prediction = accent_model.predict(features, verbose=0)
  predicted_class = np.argmax(prediction)
  confidence = float(np.max(prediction) * 100)
  accent = accent_encoder.inverse_transform([predicted_class])[0]

  return accent, confidence


def predict_emotion_in_memory(y_16k, sr_16k):
  if emotion_model is None:
    return "Emotion Model Unavailable", 0.0

  if len(y_16k) == 0:
    return "Emotion Cannot predict", 0.0

  y = librosa.resample(y_16k, orig_sr=sr_16k, target_sr=22050)
  sr = 22050

  mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
  zcr = np.mean(librosa.feature.zero_crossing_rate(y))
  rms = np.mean(librosa.feature.rms(y=y))
  centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
  bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
  rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

  features = np.concatenate(
      [mfcc, [zcr], [rms], [centroid], [bandwidth], [rolloff]]
  ).reshape(1, -1)
  features = emotion_scaler.transform(features)

  prediction = emotion_model.predict(features, verbose=0)
  predicted = np.argmax(prediction)
  confidence = float(np.max(prediction) * 100)
  emotion = emotion_encoder.inverse_transform([predicted])[0]

  if emotion.lower() == "normal":
       emotion = "Neutral"

  return emotion, confidence


def analyze_audio_features_in_memory(y, sr, transcript=""):
    if len(y) == 0:
        return {
            "duration": 0,
            "speech_rate": 0,
            "intonation": "Analysis Unavailable",
            "pitch_variation": 0,
        }

    try:
        # -----------------------------------------
        # BASIC AUDIO INFORMATION
        # -----------------------------------------
        duration = librosa.get_duration(y=y, sr=sr)

        # -----------------------------------------
        # REMOVE TRAILING / LEADING SILENCE AGAIN
        # -----------------------------------------
        y_trimmed, _ = librosa.effects.trim(
            y,
            top_db=25
        )

        if len(y_trimmed) == 0:
            return {
                "duration": round(duration, 2),
                "speech_rate": 0,
                "intonation": "Undetermined",
                "pitch_variation": 0,
            }

        # -----------------------------------------
        # SPEECH RATE
        # -----------------------------------------
        onset_frames = librosa.onset.onset_detect(
            y=y_trimmed,
            sr=sr
        )

        speech_rate = len(onset_frames) / max(
            librosa.get_duration(y=y_trimmed, sr=sr),
            1
        )

        # -----------------------------------------
        # PITCH DETECTION
        # -----------------------------------------
        pitch_values = librosa.yin(
            y_trimmed,
            fmin=75,
            fmax=300,
            sr=sr
        )

        pitch_values = np.asarray(pitch_values)

        # -----------------------------------------
        # REMOVE INVALID PITCH VALUES
        # -----------------------------------------
        pitch_values = pitch_values[
            np.isfinite(pitch_values)
        ]

        # -----------------------------------------
        # TOO FEW PITCH VALUES
        # -----------------------------------------
        if len(pitch_values) < 10:
            return {
                "duration": round(duration, 2),
                "speech_rate": round(speech_rate, 2),
                "intonation": "Undetermined",
                "pitch_variation": 0,
            }

        # -----------------------------------------
        # REMOVE EXTREME PITCH OUTLIERS
        # -----------------------------------------
        lower_limit = np.percentile(
            pitch_values,
            10
        )

        upper_limit = np.percentile(
            pitch_values,
            90
        )

        filtered_pitch = pitch_values[
            (pitch_values >= lower_limit) &
            (pitch_values <= upper_limit)
        ]

        if len(filtered_pitch) < 10:
            filtered_pitch = pitch_values

        pitch_std = np.std(filtered_pitch)

        # -----------------------------------------
        # QUESTION DETECTION
        # -----------------------------------------
        text = transcript.strip().lower()

        question_words = [
            "ano",
            "bakit",
            "paano",
            "saan",
            "kailan",
            "sino",
            "alin",
            "magkano",
            "gaano",
            "pwede ba",
            "puwede ba",
            "maaari ba",
            "okay ba",
            "tama ba",
            "ganun ba",
            "ganon ba",
            "ito ba",
            "iyan ba",
            "iyon ba",
            "ba",
        ]

        is_question = False

        # -----------------------------------------
        # EXPLICIT QUESTION MARK
        # -----------------------------------------
        if text.endswith("?"):
            is_question = True

        # -----------------------------------------
        # QUESTION WORD / PARTICLE DETECTION
        # -----------------------------------------
        for phrase in question_words:

            if phrase in text:
                is_question = True
                break

        # -----------------------------------------
        # ANALYZE ONLY THE FINAL SPOKEN PORTION
        # -----------------------------------------
        total_pitch = len(filtered_pitch)

        final_start = int(total_pitch * 0.65)

        final_pitch = filtered_pitch[final_start:]

        if len(final_pitch) < 8:

            intonation = "Undetermined"

        else:

            # -----------------------------------------
            # REMOVE VERY LAST PITCH FRAMES
            # -----------------------------------------
            if len(final_pitch) > 12:
                final_pitch = final_pitch[:-3]

            # -----------------------------------------
            # SPLIT FINAL PITCH INTO TWO PARTS
            # -----------------------------------------
            split_point = len(final_pitch) // 2

            first_half = final_pitch[:split_point]
            second_half = final_pitch[split_point:]

            if (
                len(first_half) < 3
                or len(second_half) < 3
            ):
                intonation = "Undetermined"

            else:

                first_mean = float(
                    np.median(first_half)
                )

                second_mean = float(
                    np.median(second_half)
                )

                pitch_change = (
                    second_mean - first_mean
                )

                # -----------------------------------------
                # NORMALIZED PITCH CHANGE
                # -----------------------------------------
                normalized_change = (
                    pitch_change /
                    max(first_mean, 1.0)
                )

                # -----------------------------------------
                # ADDITIONAL FINAL PITCH TREND
                # -----------------------------------------
                x = np.arange(len(final_pitch))

                slope = np.polyfit(
                    x,
                    final_pitch,
                    1
                )[0]

                normalized_slope = (
                    slope /
                    max(first_mean, 1.0)
                )

                # -----------------------------------------
                # DEBUG INFORMATION
                # -----------------------------------------
                print(
                    "\n========== INTONATION ANALYSIS =========="
                )
                print(
                    f"Transcript: {transcript}"
                )
                print(
                    f"Question detected: {is_question}"
                )
                print(
                    f"Pitch samples: {len(final_pitch)}"
                )
                print(
                    f"First final pitch: "
                    f"{first_mean:.2f} Hz"
                )
                print(
                    f"Second final pitch: "
                    f"{second_mean:.2f} Hz"
                )
                print(
                    f"Pitch change: "
                    f"{pitch_change:.2f} Hz"
                )
                print(
                    f"Normalized change: "
                    f"{normalized_change:.4f}"
                )
                print(
                    f"Normalized slope: "
                    f"{normalized_slope:.6f}"
                )
                print(
                    "==========================================\n"
                )

                # -----------------------------------------
                # INTONATION DECISION
                # -----------------------------------------
                #
                # 3% relative pitch change is used as the
                # main threshold.
                #
                RISING_THRESHOLD = 0.03
                FALLING_THRESHOLD = -0.03

                if normalized_change > RISING_THRESHOLD:

                    intonation = "Rising Intonation"

                elif normalized_change < FALLING_THRESHOLD:

                    intonation = "Falling Intonation"

                else:

                    intonation = "Level Intonation"

        return {
            "duration": round(duration, 2),
            "speech_rate": round(speech_rate, 2),
            "intonation": intonation,
            "pitch_variation": round(
                float(pitch_std),
                2
            ),
        }

    except Exception as e:

        print(
            f"Feature extraction error: {e}"
        )

        return {
            "duration": 0,
            "speech_rate": 0,
            "intonation": "Analysis Error",
            "pitch_variation": 0,
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
      if emb is None:
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
            "tag": dialect_tag,
        })
      working_text = re.sub(pattern, " ", working_text)

  return scores, detected, risks


def process_text_analysis(text):
  rule_scores, markers, risks = rule_engine(text)
  sbert_scores = sbert_similarity(text)
  final_scores = {}

  for province in provinces:
    final_scores[province] = (rule_scores[province] * 0.6) + (
        sbert_scores[province] * 0.4
    )

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
      "scores": normalized_scores,
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
        return jsonify({
            "error": "No audio uploaded",
            "message": "No audio detected. Please speak and try recording again."
        }), 400

    unique_name = f"{uuid.uuid4()}.wav"
    audio_path = os.path.join(UPLOAD_FOLDER, unique_name)

    try:
        audio.save(audio_path)

        # Process audio pipeline
        y_cached, sr_cached, audio_detected = process_audio_pipeline(
            audio_path,
            use_noise=use_noise
        )

        # -----------------------------------------
        # STOP IF NO MEANINGFUL AUDIO IS DETECTED
        # -----------------------------------------
        if not audio_detected:
            print("=== NO AUDIO DETECTED - SKIPPING WHISPER ===")

            return jsonify({
                "audio_detected": False,
                "original_text": "",
                "message": "No audio detected. Please speak and try recording again.",
                "accent": "Not Available",
                "accent_confidence": 0,
                "duration_seconds": 0,
                "speech_rate": 0,
                "intonation": "Not Available",
                "pitch_variation": 0,
                "emotion": "Not Available",
                "emotion_confidence": 0
            })

        # -----------------------------------------
        # ONLY RUN WHISPER IF AUDIO IS DETECTED
        # -----------------------------------------
        with torch.no_grad():
            result = whisper_model.transcribe(
                audio_path,
                fp16=(device == "cuda"),
                beam_size=5,
                language="tl"
            )

        text = result["text"].strip()

        # -----------------------------------------
        # MAXIMUM WORD LIMIT
        # -----------------------------------------
        tokens = re.findall(r"\b[\w']+\b", text)
        token_count = len(tokens)

        print(f"Token count: {token_count}")

        if token_count > 30:
            print("=== MAXIMUM TOKEN LIMIT EXCEEDED ===")

            return jsonify({
                "audio_detected": True,
                "original_text": text,
                "message": "Maximum of 20 tokens/words only.",
                "error_type": "TOKEN_LIMIT_EXCEEDED",
                "token_count": token_count,
                "max_tokens": 30,
                "accent": "Not Available",
                "accent_confidence": 0,
                "duration_seconds": 0,
                "speech_rate": 0,
                "intonation": "Not Available",
                "pitch_variation": 0,
                "emotion": "Not Available",
                "emotion_confidence": 0
            }), 400
         
        # -----------------------------------------
        # RUN FEATURE PREDICTIONS
        # -----------------------------------------
        accent, accent_confidence = predict_accent_in_memory(
            y_cached,
            sr_cached
        )

        emotion, emotion_confidence = predict_emotion_in_memory(
            y_cached,
            sr_cached
        )

        features = analyze_audio_features_in_memory(
            y_cached,
            sr_cached,
            text
        )

        return jsonify({
            "audio_detected": True,
            "original_text": text,
            "message": "Audio successfully detected.",
            "accent": accent,
            "accent_confidence": round(accent_confidence, 2),
            "duration_seconds": features["duration"],
            "speech_rate": features["speech_rate"],
            "intonation": features["intonation"],
            "pitch_variation": features["pitch_variation"],
            "emotion": emotion,
            "emotion_confidence": round(emotion_confidence, 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


@app.route("/analyze", methods=["POST"])
def analyze():
  data = request.get_json() or {}
  text = data.get("conversation", "").strip()
  intonation_input = data.get("intonation", "Unknown")
  emotion_input = data.get("emotion", "Emotion Cannot predict")
  accent_input = data.get("accent", "Unknown")

  try:
    speech_rate = float(data.get("speech_rate", 0))
  except (ValueError, TypeError):
    speech_rate = 0.0

  if not text:
    return jsonify({
        "explanation": "<div style='color:#64748b;'>No text received.</div>"
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
  tokens = re.findall(r"\b[\w']+\b", text.lower())
  word_count = len(tokens)

  # Honorifics
  honorifics = ["po", "opo", "ho", "oho"]
  honorific_found = [
      token for token in tokens if token in honorifics and token
  ]
  honorific_found = list(dict.fromkeys(honorific_found))  # Unique list

  honorific_display = (
      ", ".join(honorific_found) if honorific_found else "None"
  )
  verbal_style = "Polite" if honorific_found else "Plain"

  diction_html = f"""
    <div style='margin-bottom:16px; background:#eff6ff; border-left:5px solid #0284c7; padding:12px; border-radius:8px;'>
        <div style='font-weight:bold; color:#0369a1; margin-bottom:6px;'>
            Diction Patterns
        </div>
        <div style='font-size:14px; color:#334155;'>
            • Total Words: <strong>{word_count}</strong><br>
            • Regional Marker Count: <strong>{len(markers)}</strong><br>
            • Honorific Usage: <strong>{honorific_display}</strong><br>
            • Speech Register: <strong>{verbal_style}</strong>
        </div>
    </div>
    """

  sentence_particles = [
      "ba",
      "nga",
      "po",
      "ho",
      "naman",
      "eh",
      "kasi",
      "diba",
      "pala",
      "na",
      "pa",
      "lang",
      "din",
      "rin",
      "daw",
      "raw",
  ]
  sentences = re.split(r"[.!?]+", text.lower())
  particles_found = []

  for sentence in sentences:
    words = re.findall(r"\b[\w']+\b", sentence)
    if not words:
      continue
    last_word = words[-1]
    if last_word in sentence_particles and last_word not in particles_found:
      particles_found.append(last_word)

  particle_display = ", ".join(particles_found) if particles_found else "None"

  prosody_html = f"""
    <div style='margin-bottom:16px; background:#fefce8; border-left:5px solid #ca8a04; padding:12px; border-radius:8px;'>
        <div style='font-weight:bold; color:#a16207; margin-bottom:6px;'>
            Intonation Markers and Prosodic Signals
        </div>
        <div style='font-size:14px; color:#334155;'>
            • Sentence-final Particles: <strong>{particle_display}</strong><br>
            • Intonation Pattern: <strong>{intonation_input}</strong><br>
            • Speech Rate: <strong>{speech_rate:.2f} units/sec</strong>
        </div>
    </div>
    """

  sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
  ranking_html = "".join([
      f"<div style='margin-bottom:6px;'>{prov.capitalize()}:"
      f" <strong>{score:.2f}%</strong></div>"
      for prov, score in sorted_scores
  ])

  ra_html = f"""
    <div style='margin-bottom:18px; background:#eef8ff; border-left:6px solid #0284c7; padding:14px; border-radius:8px;'>
        <div style='font-size:18px; font-weight:bold; color:#0369a1; margin-bottom:10px;'>
            Rule-Augmented
        </div>
        {diction_html}
        {prosody_html}
    </div>
    """

  sbert_html = f"""
    <div style='margin-bottom:18px; background:#f8fafc; border-left:6px solid #6366f1; padding:14px; border-radius:8px;'>
        <div style='font-size:18px; font-weight:bold; color:#4338ca; margin-bottom:10px;'>
            SBERT Semantic Similarity
        </div>
        <div style='font-size:14px; color:#334155;'>
            • Closest Province/Region: <strong>{dialect}</strong><br><br>
            • System Confidence Level: <strong>{confidence:.2f}%</strong>
        </div>
        <hr style="margin:12px 0">
        <div style='font-weight:bold; margin-bottom:6px;'>
            Province Similarity Ranking
        </div>
        {ranking_html}
    </div>
    """

  ann_html = f"""
    <div style='margin-bottom:18px; background:#fff7ed; border-left:6px solid #ea580c; padding:14px; border-radius:8px;'>
        <div style='font-size:18px; font-weight:bold; color:#c2410c; margin-bottom:10px;'>
            Artificial Neural Network (ANN)
        </div>
        <div style='font-size:14px; color:#334155;'>
            • Predicted Accent: <strong>{accent_input}</strong><br><br>
            • Predicted Emotion: <strong>{emotion_input}</strong>
        </div>
    </div>
    """

  report = [ra_html, sbert_html, ann_html]

  if risks:
    report.append(
        "<div style='font-weight:bold; margin-bottom:10px; color:#e11d48;'>Flagged"
        " Regional Risk Words</div>"
    )
    for risk in risks:
      report.append(f"""
            <div style='margin-bottom:10px; border-left:4px solid #e11d48; padding:8px; background:#fff1f2; border-radius:6px;'>
                • <strong>{risk.get('word', 'unknown')}</strong> ({risk.get('category', 'Uncategorized')})<br>
                <small style='color:#475569;'>Meaning: {risk.get('meaning', 'No meaning metadata')} [{risk.get('tag', 'UNK').upper()}]</small>
            </div>
            """)
  else:
    report.append(
        "<div style='color:#64748b; font-style:italic;'>No regional"
        " misunderstanding or dialect-sensitive terms detected.</div>"
    )

  return jsonify({
      "explanation": "".join(report),
      "raw_data": {
          "dialect": dialect,
          "confidence": confidence,
          "scores": scores,
          "risks": risks,
      },
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
  app.run(debug=False, host="0.0.0.0", port=8080)