import whisper
import sounddevice as sd
import soundfile as sf
import numpy as np
import subprocess
import tempfile
import os

whisper_model = whisper.load_model("small")

VOICE_MODEL = os.path.expanduser("~/tars/voices/fr_FR-gilles-low.onnx")
LENGTH_SCALE = "0.7"
SAMPLE_RATE = 16000
DURATION = 5

def record_audio():
    print("Écoute...")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    return audio

def transcribe(audio):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, SAMPLE_RATE)
        result = whisper_model.transcribe(f.name, language="fr")
        os.unlink(f.name)
    return result["text"].strip()

def clean_for_tts(text):
    replacements = {
        "TARS": "TARSE",
        "tars": "tarse",
        "Tars": "Tarse",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def speak(text):
    text = clean_for_tts(text)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        subprocess.run([
            "piper",
            "--model", VOICE_MODEL,
            "--length_scale", LENGTH_SCALE,
            "--output_file", f.name
        ], input=text.encode(), check=True)
        subprocess.run(["afplay", f.name], check=True)
        os.unlink(f.name)