from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    conversation = request.form["conversation"]

    if conversation.strip() == "":
        result = "Please enter a conversation."
    else:
        # Placeholder analysis logic
        result = """
        Potential Cross-Cultural Misunderstanding Detected:<br>
        - Possible homonym usage.<br>
        - Indirect refusal detected.<br>
        - Politeness marker varies across dialect.
        """

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)