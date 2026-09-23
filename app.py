from model_loader import (
    device,

    accent_model,
    accent_scaler,
    accent_encoder,

    emotion_model,
    emotion_scaler,
    emotion_encoder,

    intonation_model,
    intonation_acoustic_scaler,
    intonation_encoder,

    diction_model,
    diction_encoder,

    whisper_model,
    sbert_model
)

from data_loader import (
    provinces,
    proto_df,
    risk_df,
    honorifics_df,
    diction_rules_df,
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

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file
)

import edge_tts
import librosa
import noisereduce as nr
import numpy as np
from sentence_transformers import SentenceTransformer, util
import soundfile as sf
import torch


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# =========================================================
# AUDIO PROCESSING PIPELINE
# =========================================================

def process_audio_pipeline(
    audio_path,
    use_noise=False
):

    # -----------------------------------------
    # LOAD ORIGINAL AUDIO
    # -----------------------------------------

    y, sr = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    if len(y) == 0:

        print(
            "=== EMPTY AUDIO FILE ==="
        )

        return y, sr, False

    original_y = y.copy()

    # -----------------------------------------
    # RAW AUDIO LEVEL DETECTION
    # IMPORTANT:
    # DO THIS BEFORE NORMALIZATION
    # -----------------------------------------

    rms = librosa.feature.rms(
        y=original_y,
        frame_length=2048,
        hop_length=512
    )[0]

    if len(rms) == 0:

        print(
            "=== NO AUDIO FRAMES ==="
        )

        return y, sr, False

    mean_rms = float(
        np.mean(rms)
    )

    max_rms = float(
        np.max(rms)
    )

    peak = float(
        np.max(
            np.abs(original_y)
        )
    )

    duration = librosa.get_duration(
        y=original_y,
        sr=sr
    )

    print(
        "\n================ AUDIO CHECK ================"
    )

    print(
        f"Duration: {duration:.2f} seconds"
    )

    print(
        f"Mean RMS: {mean_rms:.6f}"
    )

    print(
        f"Max RMS:  {max_rms:.6f}"
    )

    print(
        f"Peak:     {peak:.6f}"
    )

    print(
        "=============================================\n"
    )

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

        print(
            "=== NO MEANINGFUL AUDIO DETECTED ==="
        )

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

            print(
                f"Noise reduction skipped: {e}"
            )

    # -----------------------------------------
    # TRIM SILENCE
    # -----------------------------------------

    y, _ = librosa.effects.trim(
        y,
        top_db=20
    )

    if len(y) == 0:

        print(
            "=== AUDIO EMPTY AFTER TRIMMING ==="
        )

        return y, sr, False

    # -----------------------------------------
    # NORMALIZE ONLY AFTER AUDIO DETECTION
    # -----------------------------------------

    y = librosa.util.normalize(y)

    # -----------------------------------------
    # SAVE PROCESSED AUDIO
    # -----------------------------------------

    sf.write(
        audio_path,
        y,
        sr
    )

    print(
        "=== AUDIO DETECTED ==="
    )

    return y, sr, True


# =========================================================
# ACCENT ANN
# =========================================================

def predict_accent_in_memory(
    y_16k,
    sr_16k
):

    if accent_model is None:

        return (
            "Model Unavailable",
            0.0
        )

    if len(y_16k) == 0:

        return (
            "Unknown (Empty Audio)",
            0.0
        )

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

    features = np.concatenate(
        [
            mfcc,
            [zcr],
            [rms],
            [centroid],
            [bandwidth],
            [rolloff]
        ]
    ).reshape(1, -1)

    features = accent_scaler.transform(
        features
    )

    prediction = accent_model.predict(
        features,
        verbose=0
    )

    predicted_class = np.argmax(
        prediction
    )

    confidence = float(
        np.max(prediction) * 100
    )

    accent = accent_encoder.inverse_transform(
        [predicted_class]
    )[0]

    return accent, confidence


# =========================================================
# EMOTION ANN
# =========================================================

def predict_emotion_in_memory(
    y_16k,
    sr_16k,
    text=""
):

    if emotion_model is None:

        return (
            "Emotion Model Unavailable",
            0.0
        )

    if len(y_16k) == 0:

        return (
            "Emotion Cannot predict",
            0.0
        )

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

    features = np.concatenate(
        [
            mfcc,
            [zcr],
            [rms],
            [centroid],
            [bandwidth],
            [rolloff]
        ]
    ).reshape(1, -1)

    # -----------------------------------------
    # SCALE FEATURES
    # -----------------------------------------

    features = emotion_scaler.transform(
        features
    )

    # -----------------------------------------
    # ANN PREDICTION
    # -----------------------------------------

    prediction = emotion_model.predict(
        features,
        verbose=0
    )

    predicted_class = np.argmax(
        prediction
    )

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
        "masaya ako"
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
        "malungkot ako"
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
        "nakakagalit"
    ]

    # -----------------------------------------
    # DEMO / RULE OVERRIDE
    # -----------------------------------------

    detected_emotion = None

    if any(
        phrase in text_lower
        for phrase in happy_phrases
    ):

        detected_emotion = "Happy"

    elif any(
        phrase in text_lower
        for phrase in sad_phrases
    ):

        detected_emotion = "Sad"

    elif any(
        phrase in text_lower
        for phrase in angry_phrases
    ):

        detected_emotion = "Angry"

    # -----------------------------------------
    # USE TEXT FALLBACK ONLY WHEN CLEAR
    # -----------------------------------------

    if detected_emotion is not None:

        print(
            "\n========== EMOTION PREDICTION =========="
        )

        print(
            f"ANN Prediction: {ann_emotion}"
        )

        print(
            f"ANN Confidence: {ann_confidence:.2f}%"
        )

        print(
            f"Text Evidence: {detected_emotion}"
        )

        print(
            f"Final Prediction: {detected_emotion}"
        )

        print(
            "=========================================\n"
        )

        return (
            detected_emotion,
            75.0
        )

    # -----------------------------------------
    # OTHERWISE USE ANN
    # -----------------------------------------

    print(
        "\n========== EMOTION PREDICTION =========="
    )

    print(
        f"ANN Prediction: {ann_emotion}"
    )

    print(
        f"ANN Confidence: {ann_confidence:.2f}%"
    )

    print(
        f"Final Prediction: {ann_emotion}"
    )

    print(
        "=========================================\n"
    )

    return (
        ann_emotion,
        ann_confidence
    )


# =========================================================
# AUDIO / INTONATION FEATURE EXTRACTION
# =========================================================

def analyze_audio_features_in_memory(
    y,
    sr,
    transcript=""
):

    if len(y) == 0:

        return {
            "duration": 0,
            "speech_rate": 0,
            "intonation": "Analysis Unavailable",
            "pitch_variation": 0,

            "normalized_final_pitch_change": 0,
            "normalized_final_pitch_slope": 0,
            "normalized_final_pitch_range": 0,
            "voiced_frame_ratio": 0
        }

    try:

        # -----------------------------------------
        # BASIC AUDIO INFORMATION
        # -----------------------------------------

        duration = librosa.get_duration(
            y=y,
            sr=sr
        )

        # -----------------------------------------
        # REMOVE LEADING / TRAILING SILENCE
        # -----------------------------------------

        y_trimmed, _ = librosa.effects.trim(
            y,
            top_db=25
        )

        if len(y_trimmed) == 0:

            return {
                "duration": round(
                    duration,
                    2
                ),
                "speech_rate": 0,
                "intonation": "Undetermined",
                "pitch_variation": 0,

                "normalized_final_pitch_change": 0,
                "normalized_final_pitch_slope": 0,
                "normalized_final_pitch_range": 0,
                "voiced_frame_ratio": 0
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

        speech_rate = (
            len(onset_frames)
            / speech_duration
        )

        # -----------------------------------------
        # PITCH / F0 DETECTION
        # -----------------------------------------

        pitch_values = librosa.yin(
            y_trimmed,
            fmin=75,
            fmax=300,
            sr=sr
        )

        pitch_values = np.asarray(
            pitch_values
        )

        # -----------------------------------------
        # REMOVE INVALID PITCH VALUES
        # -----------------------------------------

        valid_pitch_mask = np.isfinite(
            pitch_values
        )

        valid_pitch_values = pitch_values[
            valid_pitch_mask
        ]

        if len(valid_pitch_values) < 10:

            return {
                "duration": round(
                    duration,
                    2
                ),
                "speech_rate": round(
                    speech_rate,
                    2
                ),
                "intonation": "Undetermined",
                "pitch_variation": 0,

                "normalized_final_pitch_change": 0,
                "normalized_final_pitch_slope": 0,
                "normalized_final_pitch_range": 0,
                "voiced_frame_ratio": 0
            }

        # -----------------------------------------
        # VOICED FRAME RATIO
        #
        # NOTE:
        # This is retained from your current
        # implementation. librosa.yin() generally
        # returns finite F0 values, so this may
        # remain close to 1.0.
        #
        # We are NOT changing this yet because
        # the Intonation ANN has not been trained.
        # -----------------------------------------

        total_pitch_frames = len(
            pitch_values
        )

        voiced_frames = len(
            valid_pitch_values
        )

        voiced_frame_ratio = (
            voiced_frames
            /
            max(total_pitch_frames, 1)
        )

        # -----------------------------------------
        # REMOVE EXTREME PITCH OUTLIERS
        # -----------------------------------------

        lower_limit = np.percentile(
            valid_pitch_values,
            10
        )

        upper_limit = np.percentile(
            valid_pitch_values,
            90
        )

        filtered_pitch = valid_pitch_values[
            (
                valid_pitch_values
                >= lower_limit
            )
            &
            (
                valid_pitch_values
                <= upper_limit
            )
        ]

        if len(filtered_pitch) < 10:

            filtered_pitch = (
                valid_pitch_values
            )

        # -----------------------------------------
        # GENERAL PITCH VARIATION
        # -----------------------------------------

        pitch_std = np.std(
            filtered_pitch
        )

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

        if text.endswith("?"):

            is_question = True

        if any(
            phrase in text
            for phrase in question_phrases
        ):

            is_question = True

        words = re.findall(
            r"\b[\w']+\b",
            text
        )

        if any(
            word in question_words
            for word in words
        ):

            is_question = True

        if (
            words
            and words[-1] == "ba"
        ):

            is_question = True

        # -----------------------------------------
        # ANALYZE FINAL 35% OF PITCH
        # -----------------------------------------

        total_pitch = len(
            filtered_pitch
        )

        final_start = int(
            total_pitch * 0.65
        )

        final_pitch = filtered_pitch[
            final_start:
        ]

        if len(final_pitch) < 8:

            intonation = "Undetermined"

            normalized_change = 0.0
            normalized_slope = 0.0
            normalized_range = 0.0

        else:

            # -----------------------------------------
            # REMOVE A FEW FINAL FRAMES
            # -----------------------------------------

            if len(final_pitch) > 12:

                final_pitch = (
                    final_pitch[:-3]
                )

            # -----------------------------------------
            # SPLIT FINAL PITCH INTO TWO PARTS
            # -----------------------------------------

            split_point = (
                len(final_pitch) // 2
            )

            first_half = final_pitch[
                :split_point
            ]

            second_half = final_pitch[
                split_point:
            ]

            if (
                len(first_half) < 3
                or len(second_half) < 3
            ):

                intonation = "Undetermined"

                normalized_change = 0.0
                normalized_slope = 0.0
                normalized_range = 0.0

            else:

                # -----------------------------------------
                # FIRST / SECOND FINAL F0
                # -----------------------------------------

                first_mean = float(
                    np.median(first_half)
                )

                second_mean = float(
                    np.median(second_half)
                )

                # -----------------------------------------
                # RAW PITCH CHANGE
                # -----------------------------------------

                pitch_change = (
                    second_mean
                    - first_mean
                )

                # -----------------------------------------
                # FEATURE 1:
                # NORMALIZED FINAL PITCH CHANGE
                # -----------------------------------------

                normalized_change = (
                    pitch_change
                    /
                    max(first_mean, 1.0)
                )

                # -----------------------------------------
                # FEATURE 2:
                # NORMALIZED FINAL PITCH SLOPE
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
                    slope
                    /
                    max(first_mean, 1.0)
                )

                # -----------------------------------------
                # FEATURE 3:
                # NORMALIZED FINAL PITCH RANGE
                # -----------------------------------------

                pitch_range = (
                    np.max(final_pitch)
                    -
                    np.min(final_pitch)
                )

                normalized_range = (
                    pitch_range
                    /
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
                    f"Normalized range: "
                    f"{normalized_range:.4f}"
                )

                print(
                    f"Voiced frame ratio: "
                    f"{voiced_frame_ratio:.4f}"
                )

                print(
                    "==========================================\n"
                )

                # -----------------------------------------
                # CURRENT RULE-BASED DISPLAY DECISION
                # -----------------------------------------

                RISING_THRESHOLD = 0.03
                FALLING_THRESHOLD = -0.03

                if is_question:

                    if normalized_change > 0.01:

                        intonation = (
                            "Rising Intonation"
                        )

                    elif normalized_change < -0.08:

                        intonation = (
                            "Falling Intonation"
                        )

                    else:

                        intonation = (
                            "Rising/Question Intonation"
                        )

                else:

                    if normalized_change > RISING_THRESHOLD:

                        intonation = (
                            "Rising Intonation"
                        )

                    elif normalized_change < FALLING_THRESHOLD:

                        intonation = (
                            "Falling Intonation"
                        )

                    else:

                        intonation = (
                            "Level Intonation"
                        )

        # -----------------------------------------
        # RETURN RESULTS
        # -----------------------------------------

        return {

            "duration": round(
                duration,
                2
            ),

            "speech_rate": round(
                speech_rate,
                2
            ),

            "intonation":
                intonation,

            "pitch_variation":
                round(
                    float(pitch_std),
                    2
                ),

            "normalized_final_pitch_change":
                round(
                    float(normalized_change),
                    6
                ),

            "normalized_final_pitch_slope":
                round(
                    float(normalized_slope),
                    8
                ),

            "normalized_final_pitch_range":
                round(
                    float(normalized_range),
                    6
                ),

            "voiced_frame_ratio":
                round(
                    float(voiced_frame_ratio),
                    6
                )
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

            "normalized_final_pitch_change": 0,
            "normalized_final_pitch_slope": 0,
            "normalized_final_pitch_range": 0,
            "voiced_frame_ratio": 0
        }


# =========================================================
# DIALECT SIMILARITY ENGINES
# =========================================================

def sbert_similarity(text):

    if not proto_embeddings:

        return {
            p: 1 / len(provinces)
            for p in provinces
        }

    with torch.no_grad():

        emb_input = sbert_model.encode(
            text,
            convert_to_tensor=True,
            device=device
        )

        scores = {}

        for prov in provinces:

            emb = proto_embeddings.get(
                prov
            )

            if emb is None:

                scores[prov] = 0.0
                continue

            similarity = util.cos_sim(
                emb_input,
                emb
            )

            scores[prov] = (
                similarity.max().item()
            )

    total = sum(
        scores.values()
    )

    if total == 0:

        return {
            p: 1 / len(provinces)
            for p in provinces
        }

    return {
        k: v / total
        for k, v in scores.items()
    }


def rule_engine(text):

    text_lower = text.lower().strip()

    scores = {
        p: 0
        for p in provinces
    }

    detected = []
    risks = []

    if risk_df.empty:

        return (
            scores,
            detected,
            risks
        )

    risk_tokens = sorted(
        risk_df["token"].unique(),
        key=len,
        reverse=True
    )

    working_text = text_lower

    for token in risk_tokens:

        pattern = (
            r"\b"
            + re.escape(token)
            + r"\b"
        )

        if re.search(
            pattern,
            working_text
        ):

            detected.append(
                token
            )

            matches = risk_df[
                risk_df["token"] == token
            ]

            for _, row in matches.iterrows():

                dialect_tag = str(
                    row.get(
                        "dialect_tag",
                        ""
                    )
                ).lower().strip()

                if dialect_tag in scores:

                    scores[
                        dialect_tag
                    ] += 1

                risks.append({

                    "word": token,

                    "category": row.get(
                        "category",
                        ""
                    ),

                    "meaning": row.get(
                        "meaning",
                        ""
                    ),

                    "tag": dialect_tag
                })

            working_text = re.sub(
                pattern,
                " ",
                working_text
            )

    return (
        scores,
        detected,
        risks
    )


# =========================================================
# TOKENIZATION
# =========================================================

def tokenize(text):

    return re.findall(
        r"[a-zA-ZÀ-ÿ']+",
        text.lower()
    )


# =========================================================
# DICTION RULE MATCHING
#
# IMPORTANT:
# DO NOT ADD INTONATION RULES HERE.
# =========================================================

def get_diction_rule_matches(tokens):

    if diction_rules_df.empty:

        return {}

    rule_map = dict(
        zip(
            diction_rules_df["token"],
            diction_rules_df["category"]
        )
    )

    matches = {}

    for token in tokens:

        if token in rule_map:

            matches[token] = (
                rule_map[token]
            )

    return matches


# =========================================================
# DICTION RA-SBERT ENGINE
#
# EXISTING DICTION LOGIC
# LEFT SEPARATE FROM INTONATION
# =========================================================

def ra_sbert_embedding(text):

    if not text or not text.strip():

        return None, []

    tokens = tokenize(
        text
    )

    if not tokens:

        return None, []

    tokens = tokens[:30]

    # -----------------------------------------
    # DICTION RULE WEIGHT
    # -----------------------------------------

    ALPHA = 1.5

    diction_rule_matches = (
        get_diction_rule_matches(tokens)
    )

    rule_tokens = set(
        diction_rule_matches.keys()
    )

    # -----------------------------------------
    # SBERT
    # -----------------------------------------

    transformer = (
        sbert_model[0].auto_model
    )

    tokenizer = (
        sbert_model.tokenizer
    )

    encoded = tokenizer(
        " ".join(tokens),
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=False
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():

        outputs = transformer(
            **encoded
        )

        token_embeddings = (
            outputs.last_hidden_state
        )

    # -----------------------------------------
    # MAP ORIGINAL WORDS TO TRANSFORMER TOKENS
    # -----------------------------------------

    word_ids = tokenizer(
        " ".join(tokens),
        truncation=True,
        max_length=128,
        return_tensors=None
    ).word_ids()

    word_vectors = []
    word_weights = []
    token_details = []

    current_word = None
    current_vectors = []

    for index, word_id in enumerate(
        word_ids
    ):

        if word_id is None:
            continue

        if current_word is None:
            current_word = word_id

        if word_id != current_word:

            if current_vectors:

                word_vector = torch.stack(
                    current_vectors
                ).mean(
                    dim=0
                )

                word_vectors.append(
                    word_vector
                )

                original_token = (
                    tokens[current_word]
                )

                if original_token in rule_tokens:

                    weight = ALPHA
                    matched = True

                    rule_category = (
                        diction_rule_matches[
                            original_token
                        ]
                    )

                else:

                    weight = 1.0
                    matched = False
                    rule_category = None

                word_weights.append(
                    weight
                )

                token_details.append({

                    "token": original_token,

                    "weight": weight,

                    "rule_matched": matched,

                    "rule_category":
                        rule_category
                })

            current_vectors = []
            current_word = word_id

        current_vectors.append(
            token_embeddings[
                0,
                index
            ]
        )

    # -----------------------------------------
    # FINAL WORD
    # -----------------------------------------

    if (
        current_vectors
        and current_word is not None
    ):

        word_vector = torch.stack(
            current_vectors
        ).mean(
            dim=0
        )

        word_vectors.append(
            word_vector
        )

        original_token = (
            tokens[current_word]
        )

        if original_token in rule_tokens:

            weight = ALPHA
            matched = True

            rule_category = (
                diction_rule_matches[
                    original_token
                ]
            )

        else:

            weight = 1.0
            matched = False
            rule_category = None

        word_weights.append(
            weight
        )

        token_details.append({

            "token": original_token,

            "weight": weight,

            "rule_matched": matched,

            "rule_category":
                rule_category
        })

    if not word_vectors:

        return None, []

    # -----------------------------------------
    # WEIGHTED SBERT
    # -----------------------------------------

    weighted_vectors = []

    for vector, weight in zip(
        word_vectors,
        word_weights
    ):

        weighted_vector = (
            vector * weight
        )

        weighted_vectors.append(
            weighted_vector
        )

    sentence_vector = torch.stack(
        weighted_vectors
    ).mean(
        dim=0
    )

    sentence_vector = (
        torch.nn.functional.normalize(
            sentence_vector,
            p=2,
            dim=0
        )
    )

    return (
        sentence_vector,
        token_details
    )


# =========================================================
# INTONATION RULES
# =========================================================

INTONATION_ALPHA = 1.5

INTONATION_SENTENCE_FINAL_PARTICLES = {

    "ba",
    "ha",
    "eh",

    "nga",
    "naman",
    "pala",
    "na",
    "pa",
    "lang",
    "lamang",
    "daw",
    "raw",
    "din",
    "rin",
    "diba",
    "yata",
    "talaga",
    "ano"
}


def get_intonation_rule_matches(text):

    """
    Detects sentence-final particles for
    the Intonation RA-SBERT pipeline.

    Only particles occurring at the end
    of a sentence are considered matches.
    """

    if not text or not text.strip():

        return {}

    matches = {}

    sentences = re.split(
        r"[.!?]+",
        text.lower()
    )

    for sentence in sentences:

        words = re.findall(
            r"\b[\w']+\b",
            sentence
        )

        if not words:
            continue

        last_word = words[-1]

        if last_word in (
            INTONATION_SENTENCE_FINAL_PARTICLES
        ):

            matches[last_word] = (
                "intonation"
            )

    return matches


# =========================================================
# INTONATION RA-SBERT ENGINE
# =========================================================
def intonation_ra_sbert_embedding(text):
    """
    Separate RA-SBERT representation for
    Intonation analysis.

    Pipeline:

        Sentence-final particle rules
                    ↓
                  SBERT
                    ↓
                RA weighting
                    ↓
                 384-D V_S
    """

    if not text or not text.strip():
        return None, []

    # -----------------------------------------
    # FORMULA 1:
    # SENTENCE TOKENS
    # -----------------------------------------

    tokens = tokenize(text)

    if not tokens:
        return None, []

    tokens = tokens[:30]

    # -----------------------------------------
    # INTONATION RULE MATCHES
    # -----------------------------------------

    intonation_rule_matches = get_intonation_rule_matches(text)

    rule_tokens = set(
        intonation_rule_matches.keys()
    )

    # -----------------------------------------
    # DIAGNOSTIC OUTPUT
    # -----------------------------------------

    print("\n========== INTONATION RA-SBERT ==========")
    print("Transcript:", text)
    print("Matched sentence-final particles:")

    if intonation_rule_matches:
        for particle, category in intonation_rule_matches.items():
            print(
                f"    {particle} → {category}"
            )
    else:
        print("    None detected")

    # -----------------------------------------
    # SBERT MODEL
    # -----------------------------------------

    transformer = (
        sbert_model[0].auto_model
    )

    tokenizer = (
        sbert_model.tokenizer
    )

    encoded = tokenizer(
        " ".join(tokens),
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=False
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():
        outputs = transformer(
            **encoded
        )

        token_embeddings = (
            outputs.last_hidden_state
        )

    # -----------------------------------------
    # MAP WORDS TO TRANSFORMER TOKENS
    # -----------------------------------------

    word_ids = tokenizer(
        " ".join(tokens),
        truncation=True,
        max_length=128,
        return_tensors=None
    ).word_ids()

    word_vectors = []
    word_weights = []
    token_details = []

    current_word = None
    current_vectors = []

    for index, word_id in enumerate(word_ids):

        if word_id is None:
            continue

        if current_word is None:
            current_word = word_id

        if word_id != current_word:

            if current_vectors:

                word_vector = torch.stack(
                    current_vectors
                ).mean(dim=0)

                word_vectors.append(
                    word_vector
                )

                original_token = (
                    tokens[current_word]
                )

                # -----------------------------------------
                # INTONATION RULE WEIGHT
                # -----------------------------------------

                if original_token in rule_tokens:

                    weight = (
                        INTONATION_ALPHA
                    )

                    matched = True

                    rule_category = (
                        intonation_rule_matches[
                            original_token
                        ]
                    )

                else:

                    weight = 1.0
                    matched = False
                    rule_category = None

                word_weights.append(
                    weight
                )

                token_details.append({
                    "token": original_token,
                    "weight": weight,
                    "rule_matched": matched,
                    "rule_category": rule_category
                })

            current_vectors = []
            current_word = word_id

        current_vectors.append(
            token_embeddings[
                0,
                index
            ]
        )

    # -----------------------------------------
    # ADD FINAL WORD
    # -----------------------------------------

    if (
        current_vectors
        and current_word is not None
    ):

        word_vector = torch.stack(
            current_vectors
        ).mean(dim=0)

        word_vectors.append(
            word_vector
        )

        original_token = (
            tokens[current_word]
        )

        if original_token in rule_tokens:

            weight = (
                INTONATION_ALPHA
            )

            matched = True

            rule_category = (
                intonation_rule_matches[
                    original_token
                ]
            )

        else:

            weight = 1.0
            matched = False
            rule_category = None

        word_weights.append(
            weight
        )

        token_details.append({
            "token": original_token,
            "weight": weight,
            "rule_matched": matched,
            "rule_category": rule_category
        })

    if not word_vectors:
        return None, []

    # -----------------------------------------
    # DIAGNOSTIC TOKEN OUTPUT
    # -----------------------------------------

    print("\nToken details:")

    for detail in token_details:

        print(
            f"    {detail['token']} "
            f"→ weight {detail['weight']} "
            f"| matched={detail['rule_matched']} "
            f"| category={detail['rule_category']}"
        )

    # -----------------------------------------
    # WEIGHTED SBERT VECTORS
    # -----------------------------------------

    weighted_vectors = []

    for vector, weight in zip(
        word_vectors,
        word_weights
    ):

        weighted_vector = (
            vector * weight
        )

        weighted_vectors.append(
            weighted_vector
        )

    # -----------------------------------------
    # 384-D RA-SBERT REPRESENTATION
    # -----------------------------------------

    sentence_vector = torch.stack(
        weighted_vectors
    ).mean(dim=0)

    sentence_vector = (
        torch.nn.functional.normalize(
            sentence_vector,
            p=2,
            dim=0
        )
    )

    print("\nRA-SBERT dimensions:", sentence_vector.shape[0])
    print("==========================================")

    return (
        sentence_vector,
        token_details
    )


# =========================================================
# INTONATION FEATURE COMBINATION
# =========================================================
def build_intonation_features(
    text,
    acoustic_features
):

    """
    Creates the proposed combined
    Intonation ANN input.

    RA-SBERT:
        384 dimensions

    Acoustic:
        4 dimensions

    Combined:
        388 dimensions

    This function prepares the feature
    vector only. It does NOT perform
    ANN prediction yet.
    """

    vector, token_details = (
        intonation_ra_sbert_embedding(
            text
        )
    )

    if vector is None:

        return None, []

    acoustic_vector = np.array(
        [
            acoustic_features.get(
                "normalized_final_pitch_change",
                0.0
            ),

            acoustic_features.get(
                "normalized_final_pitch_slope",
                0.0
            ),

            acoustic_features.get(
                "normalized_final_pitch_range",
                0.0
            ),

            acoustic_features.get(
                "voiced_frame_ratio",
                0.0
            )
        ],
        dtype=np.float32
    )

    sbert_vector = (
        vector.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    combined_vector = np.concatenate(
        [
            sbert_vector,
            acoustic_vector
        ]
    )

    print(
        "\n========== INTONATION FEATURES =========="
    )

    print(
        f"RA-SBERT dimensions: "
        f"{len(sbert_vector)}"
    )

    print(
        f"Acoustic dimensions: "
        f"{len(acoustic_vector)}"
    )

    print(
        f"Combined dimensions: "
        f"{len(combined_vector)}"
    )

    print(
        "==========================================\n"
    )

    return (
        combined_vector.reshape(1, -1),
        token_details
    )

# =========================================================
# INTONATION FEATURE COMBINATION
# =========================================================

def build_intonation_features(
    text,
    acoustic_features
):
    ...
    return (
        combined_vector.reshape(1, -1),
        token_details
    )


# =========================================================
# INTONATION ANN PREDICTION
# =========================================================

def predict_intonation(
    text,
    acoustic_features
):

    if (
        intonation_model is None
        or intonation_acoustic_scaler is None
        or intonation_encoder is None
    ):

        return {
            "label": "Analysis Unavailable",
            "confidence": 0.0,
            "rule_matches": [],
            "token_details": []
        }

    # -----------------------------------------
    # RA-SBERT
    # -----------------------------------------

    ra_vector, token_details = (
        intonation_ra_sbert_embedding(
            text
        )
    )

    if ra_vector is None:

        return {
            "label": "Analysis Unavailable",
            "confidence": 0.0,
            "rule_matches": [],
            "token_details": []
        }

    ra_input = (
        ra_vector
        .detach()
        .cpu()
        .numpy()
        .reshape(1, 384)
        .astype(np.float32)
    )

    # -----------------------------------------
    # 3-D ACOUSTIC FEATURES
    # -----------------------------------------

    acoustic_vector = np.array(
        [
            acoustic_features.get(
                "normalized_final_pitch_change",
                0.0
            ),

            acoustic_features.get(
                "normalized_final_pitch_slope",
                0.0
            ),

            acoustic_features.get(
                "normalized_final_pitch_range",
                0.0
            )
        ],
        dtype=np.float32
    ).reshape(1, 3)

    # -----------------------------------------
    # SCALE ACOUSTIC FEATURES
    # -----------------------------------------

    acoustic_scaled = (
        intonation_acoustic_scaler
        .transform(
            acoustic_vector
        )
        .astype(np.float32)
    )

    # -----------------------------------------
    # SMALL FUSION ANN
    # -----------------------------------------

    prediction = (
        intonation_model.predict(
            [
                ra_input,
                acoustic_scaled
            ],
            verbose=0
        )
    )

    predicted_index = int(
        np.argmax(
            prediction[0]
        )
    )

    predicted_label = (
        intonation_encoder
        .inverse_transform(
            [predicted_index]
        )[0]
    )

    confidence = float(
        np.max(
            prediction[0]
        ) * 100
    )

    # -----------------------------------------
    # RULE MATCHES
    # -----------------------------------------

    rule_matches = [
        detail["token"]
        for detail in token_details
        if detail["rule_matched"]
    ]

    # -----------------------------------------
    # DEBUG
    # -----------------------------------------

    print(
        "\n========== INTONATION ANN =========="
    )

    print(
        f"Prediction: {predicted_label}"
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )

    print(
        "RA-SBERT dimensions: 384"
    )

    print(
        "Acoustic dimensions: 3"
    )

    print(
        "Small Fusion input: 384-D + 3-D"
    )

    print(
        f"Sentence-final particles: "
        f"{rule_matches}"
    )

    print(
        "====================================\n"
    )

    return {
        "label":
            str(predicted_label),

        "confidence":
            round(
                confidence,
                2
            ),

        "rule_matches":
            rule_matches,

        "token_details":
            token_details
    }

# =========================================================
# DICTION ANN CLASSIFIER
# =========================================================

def predict_diction(text):

    if (
        diction_model is None
        or diction_encoder is None
    ):

        return {

            "label":
                "Diction Model Unavailable",

            "confidence":
                0.0,

            "token_details":
                []
        }

    # Generate Diction RA-SBERT representation
    vector, token_details = (
        ra_sbert_embedding(
            text
        )
    )

    if vector is None:

        return {

            "label":
                "Diction Cannot Predict",

            "confidence":
                0.0,

            "token_details":
                []
        }

    features = (
        vector.detach()
        .cpu()
        .numpy()
        .reshape(1, -1)
    )

    prediction = diction_model.predict(
        features,
        verbose=0
    )

    predicted_class = np.argmax(
        prediction,
        axis=1
    )[0]

    confidence = float(
        np.max(prediction) * 100
    )

    predicted_label = (
        diction_encoder.inverse_transform(
            [predicted_class]
        )[0]
    )

    print(
        "\n========== DICTION ANN PREDICTION =========="
    )

    print(
        f"Prediction: {predicted_label}"
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )

    print(
        "============================================\n"
    )

    return {

        "label":
            str(predicted_label),

        "confidence":
            round(
                confidence,
                2
            ),

        "token_details":
            token_details
    }


# =========================================================
# TEXT ANALYSIS
# =========================================================

def process_text_analysis(text):

    rule_scores, markers, risks = (
        rule_engine(text)
    )

    sbert_scores = (
        sbert_similarity(text)
    )

    best_match = max(
        sbert_scores,
        key=sbert_scores.get
    )

    confidence = (
        sbert_scores[
            best_match
        ] * 100
    )

    normalized_scores = {

        province: round(
            score * 100,
            2
        )

        for province, score
        in sbert_scores.items()
    }

    return {

        "dialect":
            f"{best_match.capitalize()} Dialect",

        "confidence":
            round(
                confidence,
                2
            ),

        "markers":
            markers,

        "risks":
            risks,

        "scores":
            normalized_scores
    }


# =========================================================
# CORE APP ROUTING
# =========================================================

@app.route("/")
def home():

    return render_template(
        "landing.html"
    )


@app.route("/analyzer")
def analyzer():

    return render_template(
        "index.html"
    )


# =========================================================
# TRANSCRIBE
# =========================================================

@app.route(
    "/transcribe",
    methods=["POST"]
)
def transcribe():

    audio = request.files.get(
        "audio"
    )

    use_noise = (
        request.form.get(
            "noise_suppression"
        ) == "true"
    )

    if not audio:

        return jsonify({

            "error":
                "No audio uploaded",

            "message":
                "No audio detected. "
                "Please speak and try "
                "recording again."
        }), 400

    unique_name = (
        f"{uuid.uuid4()}.wav"
    )

    audio_path = os.path.join(
        UPLOAD_FOLDER,
        unique_name
    )

    try:

        audio.save(
            audio_path
        )

        y_cached, sr_cached, audio_detected = (
            process_audio_pipeline(
                audio_path,
                use_noise=use_noise
            )
        )

        if not audio_detected:

            return jsonify({

                "audio_detected":
                    False,

                "original_text":
                    "",

                "message":
                    "No audio detected. "
                    "Please speak and try "
                    "recording again.",

                "accent":
                    "Not Available",

                "accent_confidence":
                    0,

                "duration_seconds":
                    0,

                "speech_rate":
                    0,

                "intonation":
                    "Not Available",

                "pitch_variation":
                    0,

                "emotion":
                    "Not Available",

                "emotion_confidence":
                    0
            })

        # -----------------------------------------
        # WHISPER
        # -----------------------------------------

        segments, info = (
            whisper_model.transcribe(
                audio_path,
                beam_size=5,
                language="tl"
            )
        )

        text = " ".join(
            segment.text
            for segment in segments
        ).strip()

        # -----------------------------------------
        # MAXIMUM WORD LIMIT
        # -----------------------------------------

        tokens = re.findall(
            r"\b[\w']+\b",
            text
        )

        token_count = len(
            tokens
        )

        print(
            f"Token count: {token_count}"
        )

        if token_count > 30:

            return jsonify({

                "audio_detected":
                    True,

                "original_text":
                    text,

                "message":
                    "Maximum of 30 "
                    "tokens/words only.",

                "error_type":
                    "TOKEN_LIMIT_EXCEEDED",

                "token_count":
                    token_count,

                "max_tokens":
                    30,

                "accent":
                    "Not Available",

                "accent_confidence":
                    0,

                "duration_seconds":
                    0,

                "speech_rate":
                    0,

                "intonation":
                    "Not Available",

                "pitch_variation":
                    0,

                "emotion":
                    "Not Available",

                "emotion_confidence":
                    0
            }), 400

        # -----------------------------------------
        # FEATURE PREDICTIONS
        # -----------------------------------------

        accent, accent_confidence = (
            predict_accent_in_memory(
                y_cached,
                sr_cached
            )
        )

        emotion, emotion_confidence = (
            predict_emotion_in_memory(
                y_cached,
                sr_cached,
                text
            )
        )

        features = (
            analyze_audio_features_in_memory(
                y_cached,
                sr_cached,
                text
            )
        )

        # -----------------------------------------
        # INTONATION ANN
        #
        # 384-D RA-SBERT
        # +
        # 3 acoustic features
        # =
        # SMALL FUSION ANN
        # -----------------------------------------

        intonation_result = (
            predict_intonation(
                text,
                features
            )
        )

        return jsonify({

            "audio_detected":
                True,

            "original_text":
                text,

            "message":
                "Audio successfully detected.",

            "accent":
                accent,

            "accent_confidence":
                round(
                    accent_confidence,
                    2
                ),

            "duration_seconds":
                features[
                    "duration"
                ],

            "speech_rate":
                features[
                    "speech_rate"
                ],

            "intonation":
                intonation_result[
                    "label"
                ],

            "intonation_confidence":
                intonation_result[
                   "confidence"
                ],

            "pitch_variation":
                features[
                    "pitch_variation"
                ],

            "normalized_final_pitch_change":
                features[
                    "normalized_final_pitch_change"
                ],

            "normalized_final_pitch_slope":
                features[
                    "normalized_final_pitch_slope"
                ],

            "normalized_final_pitch_range":
                features[
                    "normalized_final_pitch_range"
                ],

            "voiced_frame_ratio":
                features[
                    "voiced_frame_ratio"
                ],

            "emotion":
                emotion,

            "emotion_confidence":
                round(
                    emotion_confidence,
                    2
                ),
        
            # -----------------------------------------
            # INTONATION ANN INFORMATION
            # -----------------------------------------

            "intonation_confidence":
                intonation_result[
                    "confidence"
                ],

            "intonation_rule_matches":
                intonation_result[
                    "rule_matches"
                ],

            "intonation_token_details":
                intonation_result[
                    "token_details"
                ],

            "intonation_ra_sbert_dimensions":
                384
        })

    except Exception as e:

        print(
            f"Transcription exception: {e}"
        )

        return jsonify({

            "error":
                str(e)

        }), 500

    finally:

        if os.path.exists(
            audio_path
        ):

            os.remove(
                audio_path
            )


# =========================================================
# TEMPORARY INTONATION DATASET COLLECTION
# =========================================================

@app.route(
    "/dataset/analyze",
    methods=["POST"]
)
def dataset_analyze():

    audio = request.files.get(
        "audio"
    )

    if not audio:

        return jsonify({
            "error":
                "No audio uploaded."
        }), 400

    if not audio.filename:

        return jsonify({
            "error":
                "Invalid audio filename."
        }), 400

    unique_name = (
        f"{uuid.uuid4()}.wav"
    )

    audio_path = os.path.join(
        UPLOAD_FOLDER,
        unique_name
    )

    try:

        audio.save(
            audio_path
        )

        # -----------------------------------------
        # PROCESS AUDIO
        # -----------------------------------------

        y_cached, sr_cached, audio_detected = (
            process_audio_pipeline(
                audio_path,
                use_noise=False
            )
        )

        if not audio_detected:

            return jsonify({
                "error":
                    "No meaningful audio detected."
            }), 400

        # -----------------------------------------
        # WHISPER TRANSCRIPTION
        # -----------------------------------------

        segments, info = (
            whisper_model.transcribe(
                audio_path,
                beam_size=5,
                language="tl"
            )
        )

        text = " ".join(
            segment.text
            for segment in segments
        ).strip()

        # -----------------------------------------
        # PITCH FEATURES
        # -----------------------------------------

        features = (
            analyze_audio_features_in_memory(
                y_cached,
                sr_cached,
                text
            )
        )

        pitch_change = float(
            features[
                "normalized_final_pitch_change"
            ]
        )

        pitch_slope = float(
            features[
                "normalized_final_pitch_slope"
            ]
        )

        pitch_range = float(
            features[
                "normalized_final_pitch_range"
            ]
        )

        # -----------------------------------------
        # PRELIMINARY LABEL
        #
        # IMPORTANT:
        # THIS IS ONLY A SUGGESTION.
        # IT IS NOT THE FINAL ANN LABEL.
        # -----------------------------------------

        LABEL_THRESHOLD = 0.03

        if (
            pitch_change >= LABEL_THRESHOLD
            and pitch_slope > 0
        ):

            suggested_label = "Rising"

            label_reason = (
                "Pitch change is positive "
                "and pitch slope is upward."
            )

        elif (
            pitch_change <= -LABEL_THRESHOLD
            and pitch_slope < 0
        ):

            suggested_label = "Falling"

            label_reason = (
                "Pitch change is negative "
                "and pitch slope is downward."
            )

        else:

            suggested_label = "Uncertain"

            label_reason = (
                "Pitch change and pitch slope "
                "do not provide a clear direction."
            )

        # -----------------------------------------
        # RETURN DATA
        # -----------------------------------------

        return jsonify({

            "success":
                True,

            "filename":
                audio.filename,

            "transcript":
                text,

            "pitch_change":
                round(
                    pitch_change,
                    4
                ),

            "pitch_slope":
                round(
                    pitch_slope,
                    4
                ),

            "pitch_range":
                round(
                    pitch_range,
                    4
                ),

            "suggested_label":
                suggested_label,

            "label_reason":
                label_reason
        })

    except Exception as e:

        return jsonify({

            "error":
                str(e)

        }), 500

    finally:

        if os.path.exists(
            audio_path
        ):

            os.remove(
                audio_path
            )


# =========================================================
# TEXT ANALYSIS
# =========================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    data = (
        request.get_json()
        or {}
    )

    text = (
        data.get(
            "conversation",
            ""
        ).strip()
    )

    intonation_input = (
        data.get(
            "intonation",
            "Unknown"
        )
    )

    emotion_input = (
        data.get(
            "emotion",
            "Emotion Cannot predict"
        )
    )

    accent_input = (
        data.get(
            "accent",
            "Unknown"
        )
    )

    try:

        speech_rate = float(
            data.get(
                "speech_rate",
                0
            )
        )

    except (
        ValueError,
        TypeError
    ):

        speech_rate = 0.0

    if not text:

        return jsonify({

            "explanation":
                "<div style='color:#64748b;'>"
                "No text received."
                "</div>"
        })

    result = (
        process_text_analysis(
            text
        )
    )

    dialect = result[
        "dialect"
    ]

    confidence = result[
        "confidence"
    ]

    risks = result[
        "risks"
    ]

    scores = result[
        "scores"
    ]

    markers = result[
        "markers"
    ]

    # -----------------------------------------
    # DICTION PATTERN ANALYSIS
    # -----------------------------------------

    tokens = re.findall(
        r"\b[\w']+\b",
        text.lower()
    )

    word_count = len(
        tokens
    )

    # -----------------------------------------
    # HONORIFIC ANALYSIS
    # -----------------------------------------

    honorific_found = []

    honorific_lookup = {}

    if not honorifics_df.empty:

        honorific_lookup = dict(
            zip(
                honorifics_df["word"],
                honorifics_df["category"]
            )
        )

        for token in tokens:

            if token in honorific_lookup:

                if token not in honorific_found:

                    honorific_found.append(
                        token
                    )

    honorific_display_items = []

    for token in honorific_found:

        category = (
            honorific_lookup.get(
                token,
                "Unknown"
            )
        )

        honorific_display_items.append(
            f"{token} ({category})"
        )

    honorific_display = (

        ", ".join(
            honorific_display_items
        )

        if honorific_display_items

        else "None"
    )

    # -----------------------------------------
    # DICTION ANN ANALYSIS
    # -----------------------------------------

    diction_result = predict_diction(
        text
    )

    verbal_style = diction_result[
        "label"
    ]

    diction_confidence = diction_result[
        "confidence"
    ]

    diction_token_details = (
        diction_result[
            "token_details"
        ]
    )

    diction_html = (
        create_diction_html(
            word_count,
            markers,
            honorific_display,
            verbal_style
        )
    )

    # -----------------------------------------
    # SENTENCE-FINAL PARTICLES
    #
    # Uses the SAME separate Intonation
    # particle definition.
    # -----------------------------------------

    particle_matches = (
        get_intonation_rule_matches(
            text
        )
    )

    particles_found = list(
        particle_matches.keys()
    )

    particle_display = (

        ", ".join(
            particles_found
        )

        if particles_found

        else "None"
    )

    prosody_html = (
        create_prosody_html(
            particle_display,
            intonation_input,
            speech_rate
        )
    )

    # -----------------------------------------
    # DIALECT SCORE RANKING
    # -----------------------------------------

    sorted_scores = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    most_similar_province, most_similar_score = (
        sorted_scores[0]
    )

    ranking_html = (
        create_ranking_html(
            sorted_scores
        )
    )

    sbert_html = (
        create_sbert_html(
            most_similar_province,
            most_similar_score,
            ranking_html
        )
    )

    ann_html = (
        create_ann_html(
            accent_input,
            emotion_input
        )
    )

    # -----------------------------------------
    # COMBINE REPORT
    # -----------------------------------------

    report = [

        diction_html,

        prosody_html,

        sbert_html,

        ann_html
    ]

    if risks:

        report.append(

            "<div style='font-weight:bold; "
            "margin-bottom:10px; "
            "color:#e11d48;'>"
            "Flagged Unique dialect Words"
            "</div>"
        )

        for risk in risks:

            report.append(

                f"""
                <div style='
                    margin-bottom:10px;
                    border-left:4px solid #e11d48;
                    padding:8px;
                    background:#fff1f2;
                    border-radius:6px;
                '>

                    • <strong>
                        {risk.get(
                            'word',
                            'unknown'
                        )}
                    </strong>

                    (
                    {risk.get(
                        'category',
                        'Uncategorized'
                    )}
                    )

                    <br>

                    <small style='color:#475569;'>

                        Meaning:

                        {risk.get(
                            'meaning',
                            'No meaning metadata'
                        )}

                    </small>

                </div>
                """
            )

    else:

        report.append(

            "<div style='color:#64748b; "
            "font-style:italic;'>"
            "No Unique dialect "
            "misunderstanding or "
            "dialect-sensitive terms "
            "detected."
            "</div>"
        )

    return jsonify({

        "explanation":
            "".join(report),

        "raw_data": {

            "dialect":
                dialect,

            "confidence":
                confidence,

            "scores":
                scores,

            "risks":
                risks,

            "diction": {

                "label":
                    verbal_style,

                "confidence":
                    diction_confidence,

                "token_details":
                    diction_token_details
            }
        }
    })


# =========================================================
# TEXT TO SPEECH
# =========================================================

@app.route(
    "/tts",
    methods=["POST"]
)
def text_to_speech():

    data = (
        request.get_json()
        or {}
    )

    text = (
        data.get(
            "text",
            ""
        ).strip()
    )

    if not text:

        return jsonify({

            "error":
                "No text provided"

        }), 400

    try:

        voice = (
            "fil-PH-AngeloNeural"
        )

        filename = (
            f"{uuid.uuid4()}.mp3"
        )

        audio_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        communicate = (
            edge_tts.Communicate(
                text,
                voice
            )
        )

        communicate.save_sync(
            audio_path
        )

        with open(
            audio_path,
            "rb"
        ) as audio_file:

            audio_data = io.BytesIO(
                audio_file.read()
            )

        if os.path.exists(
            audio_path
        ):

            os.remove(
                audio_path
            )

        audio_data.seek(0)

        return send_file(
            audio_data,
            mimetype="audio/mpeg",
            as_attachment=False
        )

    except Exception as e:

        return jsonify({

            "error":
                f"TTS Processing Exception: "
                f"{str(e)}"

        }), 500


# =========================================================
# APPLICATION ENTRY
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=False,
        host="0.0.0.0",
        port=8080
    )
