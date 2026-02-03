# 🎥 LLM‑Based Video Emotion Detection Web App

This project is a **Flask‑based web application** that analyzes a video to:

- 🎙 Transcribe spoken audio using **Whisper**
- 🧠 Generate a semantic description using a **Large Language Model (GPT‑2)**
- 🙂 Detect the **dominant facial emotion** using **DeepFace**

The system combines **computer vision**, **speech processing**, and **LLM‑based text generation** into a single end‑to‑end pipeline.

---

## 🚀 Features

- Upload and preview a video in the browser
- Speech‑to‑text transcription (Whisper)
- Emotion detection from video frames (DeepFace)
- Text generation using GPT‑2 (LLM)
- Clean, professional UI built with HTML + CSS
- Emoji‑enhanced emotion output for easy understanding

---

## 🧠 LLM Usage (Core Concept)

The **LLM component** of this project is isolated in `emotion_model.py`:

- **Whisper**: Converts speech in the video to text (foundation model)
- **GPT‑2**: Generates a semantic description from the transcription
- **Tokenization & generation** are handled using HuggingFace Transformers

Flask (`app.py`) is used **only for orchestration and serving** the web interface.

---

## 🗂 Project Structure

```text
LLM-Video-Emotion-Detection/
│
├── app.py                  # Flask application (routes & orchestration)
├── emotion_model.py        # ML + LLM pipeline (CORE FILE)
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignored files/folders
├── README.md               # Project documentation
│
├── templates/
│   └── index.html          # UI template
│
├── static/
│   └── uploads/            # Temporary uploaded videos (ignored in git)
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone <your-repo-url>
cd LLM-Video-Emotion-Detection
```

### 2️⃣ Create and activate virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ Make sure you are using **Python 3.10 or 3.11**

---

## ▶️ Run the Application

```bash
python app.py
```

Then open your browser and go to:

```
http://127.0.0.1:5000
```

---

## ⏱ Performance Notes

- Initial startup may take **30–90 seconds**
- This is expected because multiple **pretrained deep learning models** are loaded:
  - Whisper
  - GPT‑2
  - DeepFace

- Once loaded, inference works normally

> This project is intended for **demonstration and academic purposes**.

---

## ⚠️ Warnings

- TensorFlow and HuggingFace warnings during startup are **normal**
- Flask development server is used (not for production deployment)

---

## 🧪 Dataset & Training

- No dataset is required at runtime
- The system uses **pretrained models only**
- Any experimental files (CSV / PKL) are excluded from the final project

---

## 🛠 Tech Stack

- **Backend**: Flask
- **LLM**: GPT‑2 (HuggingFace Transformers)
- **Speech Model**: Whisper
- **Emotion Detection**: DeepFace (TensorFlow)
- **Frontend**: HTML, CSS, JavaScript

---

## 🎯 Use Cases

- Human emotion analysis
- Multimodal AI demonstrations
- LLM + CV academic projects
- Interview / portfolio showcase

---

## 📌 Disclaimer

This project is built for **learning and demonstration purposes** and is not optimized for production use.

---

## Deployment Note
Due to the memory requirements of Whisper, DeepFace, and LLM models, this project is intended to be run locally. Free-tier cloud platforms may run out of memory.


## 👤 Author

**Basith Rubani**

---

✨ _A multimodal AI system combining vision, speech, and large language models._

