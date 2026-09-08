"""
Test minimal : est-ce qu'Ollama renvoie un tool_call structuré ?
On expose un seul outil bidon et on regarde ce que le modèle fait.
"""
import httpx
import json

OLLAMA_URL = "http://localhost:11434/api/chat"

# Un seul outil de test : obtenir la météo (bidon, on n'exécute rien)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Obtient la météo actuelle pour une ville donnée.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Le nom de la ville"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

def test_model(model):
    print(f"\n{'='*50}")
    print(f"TEST : {model}")
    print('='*50)

    # Question qui DEVRAIT déclencher l'outil
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Quel temps fait-il à Hanoi aujourd'hui ?"}
        ],
        "tools": TOOLS,
        "stream": False
    }

    try:
        response = httpx.post(OLLAMA_URL, json=payload, timeout=60)
        data = response.json()
        message = data.get("message", {})

        # Le modèle a-t-il demandé un appel d'outil ?
        tool_calls = message.get("tool_calls")
        if tool_calls:
            print("✓ TOOL CALL détecté :")
            print(json.dumps(tool_calls, indent=2, ensure_ascii=False))
        else:
            print("✗ Pas de tool call. Réponse texte à la place :")
            print(message.get("content", "(vide)")[:300])
    except Exception as e:
        print(f"ERREUR : {e}")


if __name__ == "__main__":
    for model in ["llama3.1:8b", "tars-ft-v2"]:
        test_model(model)