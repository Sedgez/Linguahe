from flask import Flask, render_template, request, jsonify
import whisper
import os

app = Flask(__name__)

# Ensure the uploads folder exists
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load Whisper model once
print("Loading AI Model... please wait.")
model = whisper.load_model("base")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio = request.files["audio"]
    # Get the language constraint from the dropdown
    selected_lang = request.form.get("language", "tl") 
    
    audio_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, "temp_audio.wav"))
    audio.save(audio_path)

    try:
        # Use the language constraint and a prompt to guide the AI
        result = model.transcribe(
            audio_path, 
            fp16=False, 
            language=selected_lang,
            initial_prompt="Tagalog, Ilocano, Bisaya, po, opo, naimbag, maayong"
        )
        return jsonify({"text": result["text"]})
    except Exception as e:
        print(f"Transcription Error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("conversation", "").lower()

    risks = []
    
    # Cultural Specialization Rules
    if "po" in text or "opo" in text:
        risks.append("<strong>Respect Marker:</strong> High politeness detected (Tagalog context).")
    
    if "naimbag" in text:
        risks.append("<strong>Ilocano detected:</strong> Greeting used. Context: Formal/Friendly.")
        
    if "yawa" in text or "piste" in text:
        risks.append("<strong>High Risk:</strong> Profanity detected (Bisaya context). Potential communication barrier.")

    explanation = "<br>".join(risks) if risks else "General conversation detected. No high-level cultural risks found."

    return jsonify({"explanation": explanation})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)