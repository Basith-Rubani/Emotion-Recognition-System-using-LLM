import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        if "video" not in request.files:
            error = "No file uploaded"
        else:
            file = request.files["video"]

            if file.filename == "":
                error = "No file selected"

            elif allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

                # Ensure upload directory exists
                os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
                file.save(filepath)

                # 🔥 LAZY IMPORT (critical for Render)
                from emotion_model import predict_emotion

                result = predict_emotion(filepath)

    return render_template(
        "index.html",
        prediction=result["emotion"] if result else None,
        transcription=result["transcription"] if result else None,
        description=result["description"] if result else None,
        error=error
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
