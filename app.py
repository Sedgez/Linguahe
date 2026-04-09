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
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
REPORTS_FOLDER = "reports"

# Ensure directories exist
for folder in [UPLOAD_FOLDER, REPORTS_FOLDER]:
    if not os.path.exists(folder): 
        os.makedirs(folder)

# Load AI Models
print("--- Loading Linguahe Diagnostic Engine ---")
model = whisper.load_model("base")
sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# CULTURAL INDICATOR DATABASE
POLITE_MARKERS = ['po', 'opo', 'ho', 'oho', 'paki', 'palihug', 'unta', 'maki', 'tabi']
INDIRECT_MARKERS = ['baka', 'basin', 'parang', 'siguro', 'tingali', 'marahil', 'kaypala']

def load_markers():
    path = 'regional_markers.csv'
    if os.path.exists(path):
        try:
            df = pd.read_csv(path).fillna("")
            df.columns = df.columns.str.strip().str.lower()
            return df
        except Exception as e:
            print(f"Marker Load Error: {e}")
    return pd.DataFrame()

markers_df = load_markers()

def load_sentence_references():
    path = 'sentences.csv'
    if os.path.exists(path):
        try:
            print("--- Initializing Neural Clusters ---")
            df = pd.read_csv(path)
            lang_map = {"Tagalog": "fil", "Cebuano": "ceb", "Ilocano": "ilo"}
            vectors = {}
            for display_name, iso_code in lang_map.items():
                subset = df[df['iso_639_3'] == iso_code]['text'].tolist()
                if subset:
                    vectors[display_name] = sbert_model.encode(subset, convert_to_tensor=True)
            return vectors
        except Exception as e:
            print(f"Data Load Error: {e}")
    return None

language_vectors = load_sentence_references()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, "raw_audio.wav"))
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio = request.files["audio"]
    use_noise_reduction = request.form.get("noise_suppression") == "true"

    try:
        audio.save(audio_path)
        if use_noise_reduction:
            y, sr = librosa.load(audio_path, sr=None)
            reduced_y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.6)
            sf.write(audio_path, reduced_y, sr)

        result = model.transcribe(audio_path, fp16=False, language="tl")
        original_text = result["text"].strip()

        segments = result.get('segments', [])
        diction_val = round(max(0, min(100, 100 + (sum([s.get('avg_logprob', 0) for s in segments]) / len(segments) * 50))), 2) if segments else 0.0

        highlighted_text = original_text
        if not markers_df.empty and original_text:
            sorted_markers = markers_df.assign(len=markers_df['token'].str.len()).sort_values('len', ascending=False)
            for _, row in sorted_markers.iterrows():
                token = str(row['token']).strip()
                pattern = re.compile(rf'\b({re.escape(token)})\b', re.IGNORECASE)
                css_class = "linguistic-risk" if row.get('is_ambiguous') else "marker-hit"
                highlighted_text = pattern.sub(f'<mark class="{css_class}" title="{row.get("explanation", "")}">\\1</mark>', highlighted_text)

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

    if not text:
        return jsonify({"explanation": "No speech detected.", "primary_dialect": "None"})

    # 1. SEMANTIC ANALYSIS
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
    
    # 2. CULTURAL ANALYSIS
    tokens = text.split()
    polite_hits = [w for w in tokens if w in POLITE_MARKERS]
    indirect_hits = [w for w in tokens if w in INDIRECT_MARKERS]

    # Build HTML for UI
    report_lines = [
        f"<strong>Diction Score:</strong> {diction}",
        f"<strong>Semantic Match:</strong> {detected_dialect} ({round(highest_sim*100)}%)",
        "<hr style='margin: 10px 0; border: 0; border-top: 1px solid #eee;'>"
    ]

    if polite_hits:
        report_lines.append(f"✓ <strong>Cultural Encoding:</strong> Politeness detected ({', '.join(polite_hits)})")
    if indirect_hits:
        report_lines.append(f"✓ <strong>Cultural Encoding:</strong> Indirectness detected ({', '.join(indirect_hits)})")

    # 3. REGIONAL MARKERS
    found_markers = []
    if not markers_df.empty:
        for _, row in markers_df.iterrows():
            token = str(row['token']).lower().strip()
            if token in text:
                prefix = "⚠️ <strong>Risk</strong>" if row.get('is_ambiguous') else "✓ <strong>Marker</strong>"
                marker_html = f"{prefix}: '{token}' ({row.get('category', 'General')})<br><small>— {row.get('explanation', '')}</small>"
                found_markers.append(marker_html)
    
    if found_markers:
        report_lines.append("<strong>Detailed Linguistic Analysis:</strong>")
        report_lines.extend(found_markers[:10])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(REPORTS_FOLDER, f"report_{timestamp}.txt")

    clean_text = (
        f"LINGUAHE DIAGNOSTIC REPORT\n"
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Transcript: {text}\n"
        f"Diction Score: {diction}\n"
        f"Semantic Match: {detected_dialect} ({round(highest_sim*100)}%)\n"
        f"Structural Status: Verified\n"
        "------------------------------------------\n"
    )
    if polite_hits: clean_text += f"Politeness Markers: {', '.join(polite_hits)}\n"
    if indirect_hits: clean_text += f"Indirectness Markers: {', '.join(indirect_hits)}\n"
    
    if found_markers:
        clean_text += "Detailed Linguistic Analysis:\n"
        for m in found_markers:
            # This Regex strips all HTML tags like <strong> and <small>
            clean_m = re.sub(r'<[^>]+>', '', m).replace('&nbsp;', ' ')
            clean_text += f"{clean_m.strip()}\n"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(clean_text)

    return jsonify({
        "explanation": "".join([f"<div>{line}</div>" for line in report_lines]),
        "primary_dialect": detected_dialect
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)