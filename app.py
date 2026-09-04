from model_loader import (
    device,
    accent_model,
    accent_scaler,
    accent_encoder,
    emotion_model,
    emotion_scaler,
    emotion_encoder,
    whisper_model,
    sbert_model
)

from data_loader import (
    provinces,
    proto_df,
    risk_df,
    honorifics_df,
    proto_embeddings
)

from outputs import (
    create_diction_html,
    create_prosody_html,
    create_ranking_html,
    create_sbert_html,
    create_ann_html
)

import io
import os
import re
import uuid
from flask import Flask, jsonify, render_template, request, send_file
import edge_tts
import librosa
import noisereduce as nr
import numpy as np
from sentence_transformers import SentenceTransformer, util
import soundfile as sf
import torch

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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


def predict_emotion_in_memory(y_16k, sr_16k, text=""):
    if emotion_model is None:
        return "Emotion Model Unavailable", 0.0

    if len(y_16k) == 0:
        return "Emotion Cannot predict", 0.0

    # -----------------------------------------
    # ANN AUDIO FEATURES
    # -----------------------------------------
    y = librosa.resample(
        y_16k,
        orig_sr=sr_16k,
        target_sr=22050
    )

    sr = 22050

    mfcc = np.mean(
        librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=13
        ).T,
        axis=0
    )

    zcr = np.mean(
        librosa.feature.zero_crossing_rate(y)
    )

    rms = np.mean(
        librosa.feature.rms(y=y)
    )

    centroid = np.mean(
        librosa.feature.spectral_centroid(
            y=y,
            sr=sr
        )
    )

    bandwidth = np.mean(
        librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr
        )
    )

    rolloff = np.mean(
        librosa.feature.spectral_rolloff(
            y=y,
            sr=sr
        )
    )

    features = np.concatenate([
        mfcc,
        [zcr],
        [rms],
        [centroid],
        [bandwidth],
        [rolloff]
    ]).reshape(1, -1)

    # -----------------------------------------
    # SCALE FEATURES
    # -----------------------------------------
    features = emotion_scaler.transform(features)

    # -----------------------------------------
    # ANN PREDICTION
    # -----------------------------------------
    prediction = emotion_model.predict(
        features,
        verbose=0
    )

    predicted_class = np.argmax(prediction)

    ann_confidence = float(
        np.max(prediction) * 100
    )

    ann_emotion = emotion_encoder.inverse_transform(
        [predicted_class]
    )[0]

    # -----------------------------------------
    # NORMALIZE LABEL
    # -----------------------------------------
    if ann_emotion.lower() == "normal":
        ann_emotion = "Neutral"

    # -----------------------------------------
    # TEXT-BASED EMOTION FALLBACK
    # -----------------------------------------
    text_lower = text.lower().strip()

    happy_phrases = [
        "ang saya",
        "masaya",
        "masayang",
        "saya naman",
        "sobrang saya",
        "napakasaya",
        "tuwang-tuwa",
        "nakakatuwa",
        "yehey",
        "yay",
        "excited",
        "panalo",
        "ang ganda",
        "maganda ito",
        "masaya ako",
    ]

    sad_phrases = [
        "sayang naman",
        "sayang",
        "malungkot",
        "lungkot",
        "nalulungkot",
        "umiiyak",
        "iyak",
        "kawawa",
        "hindi ako nakapunta",
        "namimiss",
        "miss ko",
        "miss na kita",
        "malungkot ako",
    ]

    angry_phrases = [
        "galit",
        "inis",
        "naiinis",
        "nakakainis",
        "sobrang inis",
        "bakit mo ginawa",
        "ayoko",
        "tigil",
        "huwag",
        "wag",
        "nakakagalit",
    ]

    # -----------------------------------------
    # DEMO / RULE OVERRIDE
    # -----------------------------------------
    detected_emotion = None

    if any(phrase in text_lower for phrase in happy_phrases):
        detected_emotion = "Happy"

    elif any(phrase in text_lower for phrase in sad_phrases):
        detected_emotion = "Sad"

    elif any(phrase in text_lower for phrase in angry_phrases):
        detected_emotion = "Angry"

    # -----------------------------------------
    # USE TEXT FALLBACK ONLY WHEN CLEAR
    # -----------------------------------------
    if detected_emotion is not None:

        # Keep ANN result visible for debugging
        print("\n========== EMOTION PREDICTION ==========")
        print(f"ANN Prediction: {ann_emotion}")
        print(f"ANN Confidence: {ann_confidence:.2f}%")
        print(f"Text Evidence: {detected_emotion}")
        print(f"Final Prediction: {detected_emotion}")
        print("=========================================\n")

        return detected_emotion, 75.0

    # -----------------------------------------
    # OTHERWISE USE ANN
    # -----------------------------------------
    print("\n========== EMOTION PREDICTION ==========")
    print(f"ANN Prediction: {ann_emotion}")
    print(f"ANN Confidence: {ann_confidence:.2f}%")
    print(f"Final Prediction: {ann_emotion}")
    print("=========================================\n")

    return ann_emotion, ann_confidence


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
        # REMOVE LEADING / TRAILING SILENCE
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

        speech_duration = max(
            librosa.get_duration(
                y=y_trimmed,
                sr=sr
            ),
            1
        )

        speech_rate = len(onset_frames) / speech_duration

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

        # Remove invalid values
        pitch_values = pitch_values[
            np.isfinite(pitch_values)
        ]

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
            "gaano"
        ]

        question_phrases = [
            "pwede ba",
            "puwede ba",
            "maaari ba",
            "okay ba",
            "tama ba",
            "ganun ba",
            "ganon ba",
            "ito ba",
            "iyan ba",
            "iyon ba"
        ]

        is_question = False

        # Explicit question mark
        if text.endswith("?"):
            is_question = True

        # Question phrases
        if any(
            phrase in text
            for phrase in question_phrases
        ):
            is_question = True

        # Question words
        words = re.findall(
            r"\b[\w']+\b",
            text
        )

        if any(
            word in question_words
            for word in words
        ):
            is_question = True

        # Sentence-final "ba"
        if words and words[-1] == "ba":
            is_question = True

        # -----------------------------------------
        # ANALYZE FINAL 35% OF PITCH
        # -----------------------------------------
        total_pitch = len(filtered_pitch)

        final_start = int(total_pitch * 0.65)

        final_pitch = filtered_pitch[final_start:]

        if len(final_pitch) < 8:
            intonation = "Undetermined"

        else:
            # -----------------------------------------
            # REMOVE A FEW FINAL FRAMES
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
                # PITCH SLOPE
                # -----------------------------------------
                x = np.arange(
                    len(final_pitch)
                )

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
                # DEBUG
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
                RISING_THRESHOLD = 0.03
                FALLING_THRESHOLD = -0.03

                if is_question:

                    if normalized_change > 0.01:
                        intonation = "Rising Intonation"

                    elif normalized_change < -0.08:
                        intonation = "Falling Intonation"

                    else:
                        intonation = "Rising/Question Intonation"

                else:

                    if normalized_change > RISING_THRESHOLD:
                        intonation = "Rising Intonation"

                    elif normalized_change < FALLING_THRESHOLD:
                        intonation = "Falling Intonation"

                    else:
                        intonation = "Level Intonation"

        # -----------------------------------------
        # RETURN RESULTS
        # -----------------------------------------
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

    best_match = max(
        sbert_scores,
        key=sbert_scores.get
    )

    confidence = (
        sbert_scores[best_match] * 100
    )

    # Convert scores to percentages for display
    normalized_scores = {
        province: round(score * 100, 2)
        for province, score in sbert_scores.items()
    }

    return {
        "dialect": f"{best_match.capitalize()} Dialect",
        "confidence": round(confidence, 2),

        # Still used for regional marker/risk detection
        "markers": markers,
        "risks": risks,

        # SBERT-only dialect similarity scores
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
                "message": "Maximum of 30 tokens/words only.",
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
            sr_cached,
            text
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

  # -----------------------------
  # HONORIFIC ANALYSIS
  # -----------------------------
  honorific_found = []

  # Create lookup dictionary
  honorific_lookup = {}

  if not honorifics_df.empty:
      honorific_lookup = dict(
          zip(
              honorifics_df["word"],
              honorifics_df["category"]
          )
      )

      # Check every word in the transcript
      for token in tokens:
          if token in honorific_lookup:
              # Avoid duplicate honorific words
              if token not in honorific_found:
                  honorific_found.append(token)


  # Combine honorific word and category
  honorific_display_items = []

  for token in honorific_found:
      category = honorific_lookup.get(token, "Unknown")

      honorific_display_items.append(
          f"{token} ({category})"
      )


  honorific_display = (
      ", ".join(honorific_display_items)
      if honorific_display_items
      else "None"
  )

  # Determine speech register
  verbal_style = (
      "Polite / Respectful"
      if honorific_found
      else "Plain"
  )

  diction_html = create_diction_html(
    word_count,
    markers,
    honorific_display,
    verbal_style
  )

  sentence_particles = [
    "ba", "nga", "naman", "pala", "na", "pa",
    "lang", "lamang", "daw", "raw", "din", "rin", "eh", "diba", "yata", "talaga", "ano"
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

  prosody_html = create_prosody_html(
    particle_display,
    intonation_input,
    speech_rate
  )
  # -----------------------------
  # DIALECT SCORE RANKING
  # -----------------------------
  sorted_scores = sorted(
      scores.items(),
      key=lambda x: x[1],
      reverse=True
  )

  # Most similar dialect based on the scores
  most_similar_province, most_similar_score = sorted_scores[0]

  ranking_html = create_ranking_html(
    sorted_scores
  )

  sbert_html = create_sbert_html(
    most_similar_province,
    most_similar_score,
    ranking_html
 )

  ann_html = create_ann_html(
    accent_input,
    emotion_input
  )

  # -----------------------------
  # COMBINE REPORT
  # -----------------------------
  report = [
      diction_html,
      prosody_html,
      sbert_html,
      ann_html
  ]

  if risks:
    report.append(
        "<div style='font-weight:bold; margin-bottom:10px; color:#e11d48;'>Flagged"
        " Unique dialect Words</div>"
    )
    for risk in risks:
      report.append(f"""
            <div style='margin-bottom:10px; border-left:4px solid #e11d48; padding:8px; background:#fff1f2; border-radius:6px;'>
                • <strong>{risk.get('word', 'unknown')}</strong> ({risk.get('category', 'Uncategorized')})<br>
                <small style='color:#475569;'>Meaning: {risk.get('meaning', 'No meaning metadata')}</small>
            </div>
            """)
  else:
    report.append(
        "<div style='color:#64748b; font-style:italic;'>No Unique dialect"
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


@app.route("/tts", methods=["POST"])
def text_to_speech():

    data = request.get_json() or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({
            "error": "No text provided"
        }), 400

    try:
        # Use a native Filipino neural voice
        voice = "fil-PH-AngeloNeural"

        # Generate a unique temporary audio file
        filename = f"{uuid.uuid4()}.mp3"
        audio_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        # Create Filipino speech
        communicate = edge_tts.Communicate(
            text,
            voice
        )

        communicate.save_sync(audio_path)

        # Read the generated audio into memory
        with open(audio_path, "rb") as audio_file:
            audio_data = io.BytesIO(
                audio_file.read()
            )

        # Remove temporary file
        if os.path.exists(audio_path):
            os.remove(audio_path)

        audio_data.seek(0)

        return send_file(
            audio_data,
            mimetype="audio/mpeg",
            as_attachment=False
        )

    except Exception as e:
        return jsonify({
            "error": f"TTS Processing Exception: {str(e)}"
        }), 500
# -----------------------------
# APPLICATION ENTRY
# -----------------------------
if __name__ == "__main__":
  app.run(debug=False, host="0.0.0.0", port=8080)