import os
import cv2
import ffmpeg
import torch

# =============================
# GLOBAL MODEL CACHES (lazy)
# =============================
_whisper_model = None
_gpt_tokenizer = None
_gpt_model = None
_deepface_loaded = False


# =============================
# MODEL LOADERS (lazy)
# =============================
def load_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def load_gpt2():
    global _gpt_tokenizer, _gpt_model
    if _gpt_tokenizer is None or _gpt_model is None:
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        _gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        _gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
        _gpt_model.eval()
    return _gpt_tokenizer, _gpt_model


def load_deepface():
    global _deepface_loaded
    if not _deepface_loaded:
        from deepface import DeepFace
        _deepface_loaded = True
    return


# =============================
# UTILITY FUNCTIONS
# =============================
def extract_audio_from_video(video_path, audio_path="temp_audio.wav"):
    ffmpeg.input(video_path).output(audio_path).run(overwrite_output=True)


def transcribe_audio(audio_path):
    whisper_model = load_whisper()
    result = whisper_model.transcribe(audio_path)
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
    emotions_across_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False
            )

            dominant_emotion, emotion_prob = get_highest_non_neutral_emotion(
                result[0]["emotion"]
            )

            emotions_across_frames.append((dominant_emotion, emotion_prob))

        except Exception:
            continue

    cap.release()

    if emotions_across_frames:
        return max(emotions_across_frames, key=lambda x: x[1])

    return "unknown", 0.0


def generate_description(text):
    tokenizer, gpt_model = load_gpt2()

    inputs = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        outputs = gpt_model.generate(
            inputs,
            max_length=150,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# =============================
# MAIN FUNCTION (Flask calls)
# =============================
def predict_emotion(video_path):
    audio_path = "temp_audio.wav"

    extract_audio_from_video(video_path, audio_path)
    transcription = transcribe_audio(audio_path)

    dominant_emotion, _ = analyze_video_frames(video_path)
    description = generate_description(transcription)

    enhanced_description = (
        f"{description}\n\n"
        f"Detected dominant emotion: {dominant_emotion}"
    )

    if os.path.exists(audio_path):
        os.remove(audio_path)

    return {
        "transcription": transcription,
        "emotion": dominant_emotion,
        "description": enhanced_description
    }
