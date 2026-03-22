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

def load_markers():
    path = 'regional_markers.csv'
    if os.path.exists(path):
        try:
            # Loading CSV and ensuring column names are clean
            df = pd.read_csv(path).fillna("")
            df.columns = df.columns.str.strip().str.lower()
            return df
        except Exception as e:
            print(f"Marker Load Error: {e}")
    return pd.DataFrame()

markers_df = load_markers()

def load_transcription_hints():
    path = 'transcription_hints.json'
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"JSON Load Error: {e}")
    return {
        "tl": "Tagalog, po, opo.",
        "ceb": "Cebuano, maayong buntag.",
        "ilo": "Ilocano, naimbag a bigat."
    }

transcription_hints = load_transcription_hints()

def load_flores_references():
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
    current_prompt = transcription_hints.get(selected_lang, "")

    try:
        audio.save(audio_path)

        if use_noise_reduction:
            y, sr = librosa.load(audio_path, sr=None)
            reduced_y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.6)
            sf.write(audio_path, reduced_y, sr)

        # Whisper Transcription
        result = model.transcribe(
            audio_path, 
            fp16=False, 
            language="tl", 
            initial_prompt=current_prompt,
            temperature=0.0,
            no_speech_threshold=0.6
        )
        
        original_text = result["text"].strip()
        
        if original_text.lower() in current_prompt.lower() or len(original_text) < 2:
            original_text = ""

        segments = result.get('segments', [])
        if segments and original_text:
            avg_log = sum([s.get('avg_logprob', 0) for s in segments]) / len(segments)
            diction_val = round(max(0, min(100, 100 + (avg_log * 50))), 2)
        else:
            diction_val = 0.0

        highlighted_text = original_text
        if not markers_df.empty and original_text:
            # Highlight Logic
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
    
    # Build HTML Report
    report_lines = [
        f"<strong>Diction Score:</strong> {diction}",
        f"<strong>Semantic Match:</strong> {detected_dialect} ({round(highest_sim*100)}%)",
        "<hr style='margin: 10px 0; border: 0; border-top: 1px solid #eee;'>"
    ]
    
    found_markers = []
    if not markers_df.empty:
        for _, row in markers_df.iterrows():
            token = str(row['token']).lower().strip()
            # Searching for the marker in the text
            if token in text:
                prefix = "⚠️ <strong>Risk</strong>" if row.get('is_ambiguous') else "✓ <strong>Marker</strong>"
                category = row.get('category', 'General')
                explanation = row.get('explanation', 'No details available.')
                
                # Format: Category, Token, and the Detail (Explanation)
                marker_html = (
                    f"{prefix}: '{token}' ({category})<br>"
                    f"<small style='color: #64748b; display: block; margin-bottom: 8px;'>— {explanation}</small>"
                )
                found_markers.append(marker_html)
    
    if found_markers:
        report_lines.append("<strong>Detailed Analysis:</strong>")
        report_lines.extend(found_markers[:10])
    else:
        report_lines.append("<p style='color: #94a3b8; font-style: italic;'>No specific regional markers or ambiguous homonyms detected in this sample.</p>")

    # --- SAVE TO SERVER FOLDER ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{timestamp}.txt"
    filepath = os.path.join(REPORTS_FOLDER, filename)
    
    # Clean version for the .txt file (no HTML tags)
    clean_text_report = f"LINGUAHE DIAGNOSTIC - {datetime.now()}\n"
    clean_text_report += f"Transcript: {text}\n"
    clean_text_report += "\n".join(report_lines).replace("<strong>", "").replace("</strong>", "").replace("<br>", "\n").replace("<hr>", "---").replace("<small>", "").replace("</small>", "").replace("<em>", "").replace("</em>", "").replace("<p>", "").replace("</p>", "")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(clean_text_report)

    return jsonify({
        "explanation": "".join([f"<div>{line}</div>" for line in report_lines]),
        "primary_dialect": detected_dialect
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
