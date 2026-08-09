import httpx
import asyncio
from audio import record_audio, transcribe, speak

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "tars"

conversation_history = []

async def chat(text):
    conversation_history.append({
        "role": "user",
        "content": text
    })

    payload = {
        "model": MODEL,
        "messages": conversation_history,
        "stream": False
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(OLLAMA_URL, json=payload)
        data = response.json()

    reply = data["message"]["content"]

    conversation_history.append({
        "role": "assistant",
        "content": reply
    })

    return reply

async def main():
    print("TARS vocal actif. Ctrl+C pour quitter.")
    speak("TARS opérationnel. Je t'écoute, Jules.")

    try:
        while True:
            audio = record_audio()
            text = transcribe(audio)

            if not text or len(text) < 2:
                continue

            if any(phrase in text.lower() for phrase in ["fin de session"]):
                speak("A plus Jules.")
                break

            print(f"Jules : {text}")
            reply = await chat(text)
            print(f"TARS : {reply}")
            speak(reply)

    except KeyboardInterrupt:
        speak("A plus Jules.")

if __name__ == "__main__":
    asyncio.run(main())