import os
import re
import numpy as np
import pandas as pd
import librosa
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf

# Reuse the same Whisper and SBERT models already used by Linguahe.
# This assumes model_loader.py is in the project root.
from model_loader import whisper_model, sbert_model


# ============================================================
# PATHS
# ============================================================

VALIDATION_AUDIO_DIR = os.path.join(
    "validation_audio"
)

LABELS_CSV = "validation_labels.csv"

FUSION_MODEL_PATH = os.path.join(
    "intonation_ai",
    "models",
    "intonation_ann_small_fusion.keras"
)

ACOUSTIC_SCALER_PATH = os.path.join(
    "intonation_ai",
    "models",
    "intonation_acoustic_scaler.pkl"
)

LABEL_ENCODER_PATH = os.path.join(
    "intonation_ai",
    "models",
    "intonation_small_fusion_label_encoder.pkl"
)

OUTPUT_CSV = os.path.join(
    "intonation_ai",
    "models",
    "validation_results.csv"
)


# ============================================================
# SETTINGS
# ============================================================

ALPHA = 1.5

SENTENCE_PARTICLES = [
    "ba",
    "ha",
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
    "eh",
    "diba",
    "yata",
    "talaga",
    "ano",
]

ACOUSTIC_COLUMNS = [
    "normalized_final_pitch_change",
    "normalized_final_pitch_slope",
    "normalized_final_pitch_range",
]


# ============================================================
# LOAD SAVED SMALL FUSION COMPONENTS
# ============================================================

if not os.path.exists(FUSION_MODEL_PATH):
    raise FileNotFoundError(
        f"Fusion model not found:\n{FUSION_MODEL_PATH}"
    )

if not os.path.exists(ACOUSTIC_SCALER_PATH):
    raise FileNotFoundError(
        f"Acoustic scaler not found:\n{ACOUSTIC_SCALER_PATH}"
    )

if not os.path.exists(LABEL_ENCODER_PATH):
    raise FileNotFoundError(
        f"Label encoder not found:\n{LABEL_ENCODER_PATH}"
    )

fusion_model = tf.keras.models.load_model(
    FUSION_MODEL_PATH
)

# joblib is used for the scaler and encoder files
import joblib

acoustic_scaler = joblib.load(
    ACOUSTIC_SCALER_PATH
)

label_encoder = joblib.load(
    LABEL_ENCODER_PATH
)


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):
    return re.findall(
        r"[a-zA-ZÀ-ÿ']+",
        text.lower()
    )[:30]


# ============================================================
# INTONATION RULE MATCHING
# ============================================================

def get_intonation_rule_matches(text):
    """
    Only sentence-final particles are used for the
    Intonation RA-SBERT rule layer.

    A particle is considered a match when it is the
    final word of a sentence.
    """

    matches = set()

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

        final_word = words[-1]

        if final_word in SENTENCE_PARTICLES:
            matches.add(final_word)

    return matches


# ============================================================
# INTONATION RA-SBERT
# ============================================================

def intonation_ra_sbert(text):
    """
    Recreates the same Intonation RA-SBERT representation
    used in the 5-fold Small Fusion experiment.

    Output:
        384-dimensional NumPy vector
    """

    tokens = tokenize(text)

    if not tokens:
        return np.zeros(
            384,
            dtype=np.float32
        )

    matched_particles = (
        get_intonation_rule_matches(text)
    )

    transformer = (
        sbert_model[0].auto_model
    )

    tokenizer = sbert_model.tokenizer

    encoded = tokenizer(
        " ".join(tokens),
        truncation=True,
        max_length=128,
        return_tensors="pt",
        padding=False
    )

    device = next(
        transformer.parameters()
    ).device

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    transformer.eval()

    with torch.no_grad():

        hidden = (
            transformer(**encoded)
            .last_hidden_state[0]
        )

    tokenized = tokenizer(
        " ".join(tokens),
        truncation=True,
        max_length=128,
        return_tensors=None
    )

    word_ids = tokenized.word_ids()

    vectors = []
    weights = []

    current_word = None
    current_vectors = []
    current_weight = 1.0

    for index, word_id in enumerate(word_ids):

        if word_id is None:
            continue

        if word_id != current_word:

            if current_vectors:

                vectors.append(
                    torch.stack(
                        current_vectors
                    ).mean(0)
                )

                weights.append(
                    current_weight
                )

            current_word = word_id
            current_vectors = []

            word = (
                tokens[word_id]
                if word_id < len(tokens)
                else ""
            )

            current_weight = (
                ALPHA
                if word in matched_particles
                else 1.0
            )

        current_vectors.append(
            hidden[index]
        )

    if current_vectors:

        vectors.append(
            torch.stack(
                current_vectors
            ).mean(0)
        )

        weights.append(
            current_weight
        )

    if not vectors:

        return np.zeros(
            384,
            dtype=np.float32
        )

    vectors = torch.stack(
        vectors
    )

    weights = torch.tensor(
        weights,
        dtype=vectors.dtype,
        device=vectors.device
    ).unsqueeze(1)

    sentence_vector = (
        vectors * weights
    ).sum(0) / weights.sum()

    norm = torch.norm(
        sentence_vector
    )

    if norm > 0:

        sentence_vector = (
            sentence_vector / norm
        )

    return (
        sentence_vector
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


# ============================================================
# ACOUSTIC FEATURE EXTRACTION
# ============================================================

def extract_intonation_acoustic_features(
    audio_path
):
    """
    Uses the same pitch-feature procedure used by the
    Intonation dataset pipeline:

    1. Load at 16 kHz
    2. Trim silence
    3. librosa.yin()
    4. Remove 10th/90th percentile pitch outliers
    5. Analyze the final 35% of pitch values
    6. Calculate:
       - normalized final pitch change
       - normalized final pitch slope
       - normalized final pitch range

    voiced_frame_ratio is deliberately NOT used.
    """

    y, sr = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    if len(y) == 0:
        raise ValueError(
            "Audio file is empty."
        )

    # Match the dataset-generation pipeline: remove leading/trailing
    # silence before pitch analysis.  The training pipeline uses top_db=25.
    y_trimmed, trim_indices = (
        librosa.effects.trim(
            y,
            top_db=25
        )
    )

    if len(y_trimmed) == 0:
        raise ValueError(
            "Audio contains no usable speech after trimming."
        )

    trimmed_start = trim_indices[0] / sr
    trimmed_end = trim_indices[1] / sr
    trimmed_duration = len(y_trimmed) / sr

    print(
        "\nSpeech-region trimming:"
    )
    print(
        f"  Original duration: {len(y) / sr:.3f} s"
    )
    print(
        f"  Speech region:     {trimmed_start:.3f} - "
        f"{trimmed_end:.3f} s"
    )
    print(
        f"  Speech duration:   {trimmed_duration:.3f} s"
    )

    pitch_values = librosa.yin(
        y_trimmed,
        fmin=75,
        fmax=300,
        sr=sr
    )

    pitch_values = np.asarray(
        pitch_values
    )

    valid_pitch = pitch_values[
        np.isfinite(pitch_values)
    ]

    if len(valid_pitch) < 10:
        raise ValueError(
            "Not enough valid pitch samples."
        )

    # Same 10th/90th percentile filtering used
    # in the current Linguahe acoustic pipeline.
    lower_limit = np.percentile(
        valid_pitch,
        10
    )

    upper_limit = np.percentile(
        valid_pitch,
        90
    )

    filtered_pitch = valid_pitch[
        (valid_pitch >= lower_limit)
        &
        (valid_pitch <= upper_limit)
    ]

    if len(filtered_pitch) < 10:

        filtered_pitch = valid_pitch

    # Final 35%
    final_start = int(
        len(filtered_pitch) * 0.65
    )

    final_pitch = filtered_pitch[
        final_start:
    ]

    # Very short utterances can legitimately have fewer than 8 pitch
    # frames in the final 35%.  Keep the same "final speech" concept,
    # but use a small minimum end-window so short recordings do not fail
    # solely because of their recording length.
    if len(final_pitch) < 6:

        minimum_window = min(6, len(filtered_pitch))

        if minimum_window < 6:
            raise ValueError(
                "Not enough pitch samples in the final speech region."
            )

        final_pitch = filtered_pitch[-minimum_window:]

        print(
            "  Short-utterance fallback: using last "
            f"{minimum_window} pitch samples."
        )

    elif len(final_pitch) < 8:

        minimum_window = min(8, len(filtered_pitch))
        final_pitch = filtered_pitch[-minimum_window:]

        print(
            "  Short-utterance fallback: using last "
            f"{minimum_window} pitch samples."
        )

    if len(final_pitch) > 12:

        final_pitch = (
            final_pitch[:-3]
        )

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
        raise ValueError(
            "Not enough pitch samples to calculate final pitch change."
        )

    first_mean = float(
        np.median(first_half)
    )

    second_mean = float(
        np.median(second_half)
    )

    pitch_change = (
        second_mean - first_mean
    )

    normalized_change = (
        pitch_change
        /
        max(first_mean, 1.0)
    )

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

    return {
        "normalized_final_pitch_change":
            float(normalized_change),

        "normalized_final_pitch_slope":
            float(normalized_slope),

        "normalized_final_pitch_range":
            float(normalized_range),

        "first_final_pitch_hz":
            first_mean,

        "second_final_pitch_hz":
            second_mean,

        "pitch_samples":
            len(final_pitch),
    }


# ============================================================
# WHISPER TRANSCRIPTION
# ============================================================

def transcribe_audio(audio_path):

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

    if not text:
        raise ValueError(
            "Whisper returned an empty transcript."
        )

    return text


# ============================================================
# VALIDATION
# ============================================================

def main():

    if not os.path.exists(
        LABELS_CSV
    ):
        raise FileNotFoundError(
            f"Validation label CSV not found:\n{LABELS_CSV}"
        )

    if not os.path.isdir(
        VALIDATION_AUDIO_DIR
    ):
        raise FileNotFoundError(
            f"Validation audio folder not found:\n"
            f"{VALIDATION_AUDIO_DIR}"
        )

    labels_df = pd.read_csv(
        LABELS_CSV
    )

    required_columns = {
        "audio_file",
        "actual_label"
    }

    missing = (
        required_columns
        -
        set(labels_df.columns)
    )

    if missing:

        raise ValueError(
            "CSV is missing columns: "
            +
            ", ".join(sorted(missing))
        )

    print("=" * 70)
    print("LINGUAHE INT0NATION UNSEEN-SPEECH VALIDATION")
    print("=" * 70)

    print(
        f"Validation recordings: "
        f"{len(labels_df)}"
    )

    print(
        f"Fusion model: "
        f"{FUSION_MODEL_PATH}"
    )

    print("=" * 70)

    results = []

    for index, row in labels_df.iterrows():

        audio_file = str(
            row["audio_file"]
        ).strip()

        actual_label = str(
            row["actual_label"]
        ).strip()

        audio_path = os.path.join(
            VALIDATION_AUDIO_DIR,
            audio_file
        )

        print("\n" + "=" * 70)

        print(
            f"TEST {index + 1}/"
            f"{len(labels_df)}"
        )

        print(
            f"Audio: {audio_file}"
        )

        print(
            f"Actual label: {actual_label}"
        )

        print("=" * 70)

        result = {
            "audio_file": audio_file,
            "actual_label": actual_label,
            "transcript": "",
            "matched_particles": "",
            "normalized_final_pitch_change": np.nan,
            "normalized_final_pitch_slope": np.nan,
            "normalized_final_pitch_range": np.nan,
            "ra_sbert_dimension": np.nan,
            "acoustic_dimension": 3,
            "fusion_dimension": 35,
            "predicted_label": "",
            "confidence_percent": np.nan,
            "correct": False,
            "status": "FAILED",
            "error": "",
        }

        try:

            if not os.path.exists(
                audio_path
            ):
                raise FileNotFoundError(
                    f"Audio file not found: "
                    f"{audio_path}"
                )

            # ----------------------------------------
            # 1. Whisper
            # ----------------------------------------

            transcript = (
                transcribe_audio(
                    audio_path
                )
            )

            result["transcript"] = (
                transcript
            )

            print(
                f"\nTranscript:\n"
                f"{transcript}"
            )

            # ----------------------------------------
            # 2. Sentence-final particles
            # ----------------------------------------

            particles = (
                get_intonation_rule_matches(
                    transcript
                )
            )

            result[
                "matched_particles"
            ] = ", ".join(
                sorted(particles)
            )

            if particles:

                print(
                    "\nMatched sentence-final "
                    "particle(s): "
                    +
                    ", ".join(
                        sorted(particles)
                    )
                )

            else:

                print(
                    "\nMatched sentence-final "
                    "particle(s): None"
                )

            # ----------------------------------------
            # 3. RA-SBERT
            # ----------------------------------------

            ra_vector = (
                intonation_ra_sbert(
                    transcript
                )
            )

            result[
                "ra_sbert_dimension"
            ] = len(
                ra_vector
            )

            print(
                "\nRA-SBERT dimension: "
                f"{len(ra_vector)}"
            )

            # ----------------------------------------
            # 4. Acoustic features
            # ----------------------------------------

            acoustic = (
                extract_intonation_acoustic_features(
                    audio_path
                )
            )

            acoustic_vector = np.array(
                [
                    acoustic[
                        "normalized_final_pitch_change"
                    ],
                    acoustic[
                        "normalized_final_pitch_slope"
                    ],
                    acoustic[
                        "normalized_final_pitch_range"
                    ],
                ],
                dtype=np.float32
            ).reshape(1, -1)

            result[
                "normalized_final_pitch_change"
            ] = acoustic_vector[0, 0]

            result[
                "normalized_final_pitch_slope"
            ] = acoustic_vector[0, 1]

            result[
                "normalized_final_pitch_range"
            ] = acoustic_vector[0, 2]

            print(
                "\nAcoustic features:"
            )

            print(
                f"  Pitch change: "
                f"{acoustic_vector[0, 0]:.6f}"
            )

            print(
                f"  Pitch slope:  "
                f"{acoustic_vector[0, 1]:.8f}"
            )

            print(
                f"  Pitch range:  "
                f"{acoustic_vector[0, 2]:.6f}"
            )

            # ----------------------------------------
            # 5. Scale acoustic features
            # ----------------------------------------

            acoustic_scaled = (
                acoustic_scaler.transform(
                    acoustic_vector
                ).astype(np.float32)
            )

            # ----------------------------------------
            # 6. Small Fusion ANN
            #
            # The trained model has TWO inputs:
            #   384-D RA-SBERT
            #   3-D acoustic
            #
            # Internally it projects RA-SBERT to
            # 32-D and concatenates it with acoustic
            # features to form the 35-D fusion
            # representation.
            # ----------------------------------------

            ra_input = (
                ra_vector
                .reshape(1, 384)
                .astype(np.float32)
            )

            prediction = (
                fusion_model.predict(
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
                label_encoder.inverse_transform(
                    [predicted_index]
                )[0]
            )

            confidence = float(
                np.max(
                    prediction[0]
                ) * 100
            )

            correct = (
                predicted_label.lower()
                ==
                actual_label.lower()
            )

            result[
                "predicted_label"
            ] = str(
                predicted_label
            )

            result[
                "confidence_percent"
            ] = confidence

            result[
                "correct"
            ] = correct

            result[
                "status"
            ] = "OK"

            print(
                "\nANN:"
            )

            print(
                f"  RA-SBERT: "
                f"{len(ra_vector)}-D"
            )

            print(
                "  Acoustic: 3-D"
            )

            print(
                "  Fusion: 35-D"
            )

            print(
                f"\nPrediction: "
                f"{predicted_label}"
            )

            print(
                f"Confidence: "
                f"{confidence:.2f}%"
            )

            print(
                "Result: "
                +
                (
                    "CORRECT"
                    if correct
                    else "INCORRECT"
                )
            )

        except Exception as e:

            result[
                "error"
            ] = str(e)

            print(
                f"\nERROR: {e}"
            )

        results.append(
            result
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    os.makedirs(
        os.path.dirname(
            OUTPUT_CSV
        ),
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    successful = results_df[
        results_df["status"] == "OK"
    ].copy()

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print(
        f"Successful tests: "
        f"{len(successful)}/{len(results_df)}"
    )

    if len(successful) > 0:

        y_true = (
            successful[
                "actual_label"
            ].astype(str)
        )

        y_pred = (
            successful[
                "predicted_label"
            ].astype(str)
        )

        accuracy = (
            accuracy_score(
                y_true,
                y_pred
            )
            * 100
        )

        print(
            f"\nUnseen-speech accuracy: "
            f"{accuracy:.2f}%"
        )

        print(
            "\nClassification report:"
        )

        print(
            classification_report(
                y_true,
                y_pred,
                labels=label_encoder.classes_,
                zero_division=0
            )
        )

        print(
            "Confusion matrix "
            "[Falling, Rising]:"
        )

        print(
            confusion_matrix(
                y_true,
                y_pred,
                labels=label_encoder.classes_
            )
        )

    else:

        print(
            "\nNo successful validation "
            "recordings were processed."
        )

    print(
        "\nDetailed results saved to:"
    )

    print(
        OUTPUT_CSV
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
