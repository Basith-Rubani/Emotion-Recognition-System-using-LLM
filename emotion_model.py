import os
import cv2
import ffmpeg
import torch

# =============================
# GLOBAL LAZY MODEL CACHE
# =============================
_whisper_model = None
_gpt_tokenizer = None
_gpt_model = None
_deepface_loaded = False


# =============================
# MODEL LOADERS (MINIMAL)
# =============================
def load_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model("tiny")  # 🔥 smallest
    return _whisper_model


def load_gpt():
    global _gpt_tokenizer, _gpt_model
    if _gpt_tokenizer is None or _gpt_model is None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        _gpt_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        _gpt_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        _gpt_model.eval()
    return _gpt_tokenizer, _gpt_model


def load_deepface():
    global _deepface_loaded
    if not _deepface_loaded:
        from deepface import DeepFace
        _deepface_loaded = True


# =============================
# UTILITY FUNCTIONS
# =============================
def extract_audio_from_video(video_path, audio_path="temp_audio.wav"):
    ffmpeg.input(video_path).output(audio_path).run(overwrite_output=True, quiet=True)


def transcribe_audio(audio_path):
    model = load_whisper()
    result = model.transcribe(audio_path)
    return result["text"]


def get_highest_non_neutral_emotion(emotion_dict):
    emotion_dict = emotion_dict.copy()
    emotion_dict.pop("neutral", None)
    dominant_emotion = max(emotion_dict, key=emotion_dict.get)
    return dominant_emotion, emotion_dict[dominant_emotion]


def analyze_video_frames(video_path):
    load_deepface()
    from deepface import DeepFace

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = max(total_frames // 2, 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "unknown", 0.0

    try:
        result = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",  # 🔥 lightest backend
        )

        return get_highest_non_neutral_emotion(result[0]["emotion"])

    except Exception:
        return "unknown", 0.0


def generate_description(text):
    tokenizer, model = load_gpt()

    inputs = tokenizer.encode(text[:300], return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=120,
            do_sample=True,
            top_k=40,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# =============================
# MAIN ENTRY (Flask calls)
# =============================
def predict_emotion(video_path):
    audio_path = "temp_audio.wav"

    extract_audio_from_video(video_path, audio_path)

    transcription = transcribe_audio(audio_path)
    dominant_emotion, _ = analyze_video_frames(video_path)
    description = generate_description(transcription)

    if os.path.exists(audio_path):
        os.remove(audio_path)

    return {
        "transcription": transcription,
        "emotion": dominant_emotion,
        "description": f"{description}\n\nDetected dominant emotion: {dominant_emotion}",
    }
