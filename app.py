from flask import Flask, render_template, request, jsonify
import whisper
import os
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import re
import noisereduce as nr
import librosa
import soundfile as sf
import json

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)

#Load AI Models
print("--- Loading Linguahe Diagnostic Engine ---")
model = whisper.load_model("base")
sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

#Load Regional Markers (Symbolic Layer) ---
def load_markers():
    path = 'regional_markers.csv'
    if os.path.exists(path):
        try:
            return pd.read_csv(path).fillna("")
        except Exception as e:
            print(f"Marker Load Error: {e}")
    return pd.DataFrame()

markers_df = load_markers()

#Load Transcription Hint
def load_transcription_hints():
    path = 'transcription_hints.json'
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"JSON Load Error: {e}")
    # Default fallback if file is missing
    return {
        "tl": "Tagalog, po, opo.",
        "ceb": "Cebuano, maayong buntag.",
        "ilo": "Ilocano, naimbag a bigat."
    }

transcription_hints = load_transcription_hints()

#Load Neural Reference Clusters
def load_flores_references():
    path = 'dialects_rules.csv'
    if os.path.exists(path):
        try:
            print("--- Initializing Neural Clusters (Loading all rows)... ---")
            df = pd.read_csv(path)
            lang_map = {"Tagalog": "fil", "Cebuano": "ceb", "Ilocano": "ilo"}
            vectors = {}
            for display_name, iso_code in lang_map.items():
                # Filtering rows by ISO code and grabbing the 'text' column
                subset = df[df['iso_639_3'] == iso_code]['text'].tolist()
                if subset:
                    # Pre-calculate embeddings once at startup
                    vectors[display_name] = sbert_model.encode(subset, convert_to_tensor=True)
            print(f"--- Neural Clusters Ready for: {', '.join(vectors.keys())} ---")
            return vectors
        except Exception as e:
            print(f"Data Load Error: {e}")
    return None

language_vectors = load_flores_references()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, "raw_audio.wav"))
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio = request.files["audio"]
    selected_lang = request.form.get("language", "tl")
    use_noise_reduction = request.form.get("noise_suppression") == "true"

    # Fetching the prompt from externalized JSON
    current_prompt = transcription_hints.get(selected_lang, "Tagalog, Cebuano, Ilocano")

    try:
        audio.save(audio_path)

        if use_noise_reduction:
            y, sr = librosa.load(audio_path, sr=None)
            reduced_y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.75)
            sf.write(audio_path, reduced_y, sr)

        # Whisper Transcription
        result = model.transcribe(
            audio_path, 
            fp16=False, 
            initial_prompt=f"This is a conversation in {selected_lang}. " + current_prompt
        )
        original_text = result["text"].strip()

        # Diction Score calculation
        segments = result.get('segments', [])
        if segments:
            avg_log = sum([s.get('avg_logprob', 0) for s in segments]) / len(segments)
            diction_val = round(max(0, min(100, 100 + (avg_log * 50))), 2)
        else:
            diction_val = 0.0

        # UI Highlighting Logic
        highlighted_text = original_text
        if not markers_df.empty:
            sorted_markers = markers_df.assign(len=markers_df['token'].str.len()).sort_values('len', ascending=False)
            for _, row in sorted_markers.iterrows():
                token = str(row['token']).strip()
                pattern = re.compile(rf'\b({re.escape(token)})\b', re.IGNORECASE)
                css_class = "linguistic-risk" if row.get('is_ambiguous') else "marker-hit"
                highlighted_text = pattern.sub(f'<mark class="{css_class}" title="{row["explanation"]}">\\1</mark>', highlighted_text)

        return jsonify({
            "original_text": original_text,
            "highlighted_text": highlighted_text,
            "diction_score": f"{diction_val}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(audio_path): os.remove(audio_path)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("conversation", "").lower()
    diction = data.get("diction_score", "N/A")

    user_vec = sbert_model.encode(text, convert_to_tensor=True)
    highest_sim = 0
    detected_dialect = "General"

    if language_vectors:
        for lang, ref_vecs in language_vectors.items():
            sim_scores = util.pytorch_cos_sim(user_vec, ref_vecs)
            max_score = torch.max(sim_scores).item()
            
            if max_score > highest_sim:
                highest_sim = max_score
                detected_dialect = lang
    
    report = [f"<strong>Diction Score:</strong> {diction}"]
    report.append(f"<strong>Semantic Match:</strong> {detected_dialect} ({round(highest_sim*100)}%)")
    report.append(f"<em>Verified via FLORES+ Benchmark</em>")
    
    found_markers = []
    if not markers_df.empty:
        for _, row in markers_df.iterrows():
            if str(row['token']).lower() in text:
                prefix = "⚠️ Risk" if row['is_ambiguous'] else "✓ Marker"
                found_markers.append(f"{prefix}: '{row['token']}' ({row['category']})")
    
    if found_markers:
        report.append("<strong>Tokens Identified:</strong>")
        report.extend(found_markers[:5]) 

    return jsonify({
        "explanation": "<br>".join(report),
        "primary_dialect": detected_dialect
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)