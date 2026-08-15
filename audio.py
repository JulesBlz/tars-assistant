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

# Voix Claude : voix système macOS "Thomas" (masculine française)
CLAUDE_VOICE = "Thomas"
CLAUDE_RATE = "180"  # mots par minute, défaut ~200


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


def speak_tars(text):
    """Voix TARS via Piper (fr_FR-gilles-low, rapide)."""
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


def speak_claude(text):
    """Voix Claude via macOS say (Thomas, masculine française)."""
    text = clean_for_tts(text)
    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
        subprocess.run([
            "say",
            "-v", CLAUDE_VOICE,
            "-r", CLAUDE_RATE,
            "-o", f.name,
            text
        ], check=True)
        subprocess.run(["afplay", f.name], check=True)
        os.unlink(f.name)


def speak(text, source="tars"):
    """Point d'entrée : route vers la bonne voix selon la source."""
    if source == "claude":
        speak_claude(text)
    else:
        speak_tars(text)


def generate_audio_bytes(text, source="tars"):
    """
    Génère l'audio et retourne les bytes WAV (pour l'API HTTP /tts).
    Ne joue rien, retourne juste les données.
    """
    text = clean_for_tts(text)
    if source == "claude":
        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
            subprocess.run([
                "say",
                "-v", CLAUDE_VOICE,
                "-r", CLAUDE_RATE,
                "-o", f.name,
                text
            ], check=True)
            # Convertir AIFF en WAV pour le navigateur
            wav_path = f.name.replace(".aiff", ".wav")
            subprocess.run([
                "afconvert", "-f", "WAVE", "-d", "LEI16", f.name, wav_path
            ], check=True)
            with open(wav_path, "rb") as audio_file:
                audio_data = audio_file.read()
            os.unlink(f.name)
            os.unlink(wav_path)
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            subprocess.run([
                "piper",
                "--model", VOICE_MODEL,
                "--length_scale", LENGTH_SCALE,
                "--output_file", f.name
            ], input=text.encode(), check=True)
            with open(f.name, "rb") as audio_file:
                audio_data = audio_file.read()
            os.unlink(f.name)
    return audio_data