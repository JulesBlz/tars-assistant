"""
Couche d'abstraction sur les modèles.

call_model(messages, tools, model_config) renvoie TOUJOURS le même objet normalisé,
que le backend soit Ollama (local) ou l'API Anthropic (cloud). La boucle agentique
n'a jamais à connaître le backend : c'est ce qui permet de comparer deux modèles
sur le même jeu de test en changeant une variable d'environnement.

Objet normalisé renvoyé (ModelResponse) :
    - output_type : "tool_call" ou "text"
    - text : la réponse texte si output_type == "text", sinon None
    - tool_calls : liste de {id, name, arguments} si output_type == "tool_call", sinon []
    - tokens_prompt / tokens_completion : comptage pour le logging
    - raw : la réponse brute du backend (pour debug)
"""
import os
import json
import httpx
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/tars/.env"))

OLLAMA_URL = "http://localhost:11434/api/chat"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ModelResponse:
    output_type: str            # "tool_call" | "text"
    text: Optional[str] = None
    tool_calls: list = field(default_factory=list)
    tokens_prompt: int = 0
    tokens_completion: int = 0
    raw: dict = field(default_factory=dict)


def get_model_config():
    """
    Lit la configuration du modèle depuis les variables d'environnement.
    AGENT_BACKEND : "ollama" (défaut) ou "anthropic"
    AGENT_MODEL   : nom du modèle (ex: "tars-ft-v2", "llama3.1:8b", "claude-sonnet-4-5")
    """
    backend = os.getenv("AGENT_BACKEND", "ollama")
    model = os.getenv("AGENT_MODEL", "llama3.1:8b")
    return {"backend": backend, "model": model}


# ============================================================
# Backend Ollama (format type OpenAI)
# ============================================================

def _call_ollama(messages, tools, model):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools

    response = httpx.post(OLLAMA_URL, json=payload, timeout=120)
    data = response.json()
    message = data.get("message", {})

    # Comptage tokens (Ollama expose prompt_eval_count / eval_count)
    tokens_prompt = data.get("prompt_eval_count", 0)
    tokens_completion = data.get("eval_count", 0)

    tool_calls_raw = message.get("tool_calls")
    if tool_calls_raw:
        tool_calls = []
        for tc in tool_calls_raw:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            # Ollama renvoie déjà un dict pour arguments, mais on sécurise
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append(ToolCall(
                id=tc.get("id", ""),
                name=fn.get("name", ""),
                arguments=args,
            ))
        return ModelResponse(
            output_type="tool_call",
            tool_calls=tool_calls,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            raw=data,
        )
    else:
        return ModelResponse(
            output_type="text",
            text=message.get("content", ""),
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            raw=data,
        )


# ============================================================
# Backend Anthropic (format tool_use / content blocks)
# ============================================================

def _anthropic_tools_from_openai(tools):
    """Convertit les schémas format OpenAI vers le format Anthropic."""
    converted = []
    for t in tools:
        fn = t["function"]
        converted.append({
            "name": fn["name"],
            "description": fn["description"],
            "input_schema": fn["parameters"],
        })
    return converted


def _anthropic_messages_from_openai(messages):
    """
    Convertit l'historique format OpenAI (avec role 'tool') vers le format Anthropic
    (tool_result dans un message user, tool_use dans un message assistant).
    """
    system_prompt = None
    converted = []
    for m in messages:
        role = m["role"]
        if role == "system":
            system_prompt = m["content"]
        elif role == "user":
            converted.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            # Un assistant peut avoir demandé des outils
            if m.get("tool_calls"):
                blocks = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m["tool_calls"]:
                    fn = tc["function"]
                    args = fn["arguments"]
                    if isinstance(args, str):
                        args = json.loads(args)
                    blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": fn["name"],
                        "input": args,
                    })
                converted.append({"role": "assistant", "content": blocks})
            else:
                converted.append({"role": "assistant", "content": m["content"]})
        elif role == "tool":
            # Résultat d'outil → tool_result dans un message user
            converted.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id", ""),
                    "content": m["content"],
                }]
            })
    return system_prompt, converted


def _call_anthropic(messages, tools, model):
    from anthropic import Anthropic
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    system_prompt, converted_messages = _anthropic_messages_from_openai(messages)
    kwargs = {
        "model": model,
        "max_tokens": 2048,
        "messages": converted_messages,
    }
    if system_prompt:
        kwargs["system"] = system_prompt
    if tools:
        kwargs["tools"] = _anthropic_tools_from_openai(tools)

    response = client.messages.create(**kwargs)

    tokens_prompt = response.usage.input_tokens
    tokens_completion = response.usage.output_tokens

    # Anthropic renvoie une liste de content blocks
    tool_calls = []
    text_parts = []
    for block in response.content:
        if block.type == "tool_use":
            tool_calls.append(ToolCall(
                id=block.id,
                name=block.name,
                arguments=block.input,
            ))
        elif block.type == "text":
            text_parts.append(block.text)

    if tool_calls:
        return ModelResponse(
            output_type="tool_call",
            text=" ".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            raw={"stop_reason": response.stop_reason},
        )
    else:
        return ModelResponse(
            output_type="text",
            text=" ".join(text_parts),
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            raw={"stop_reason": response.stop_reason},
        )


# ============================================================
# Point d'entrée unifié
# ============================================================

def call_model(messages, tools, model_config):
    """
    Appelle le backend configuré et renvoie un ModelResponse normalisé.
    """
    backend = model_config["backend"]
    model = model_config["model"]

    if backend == "ollama":
        return _call_ollama(messages, tools, model)
    elif backend == "anthropic":
        return _call_anthropic(messages, tools, model)
    else:
        raise ValueError(f"Backend inconnu : {backend}")


# ============================================================
# Test manuel
# ============================================================
if __name__ == "__main__":
    from tools import TOOL_SCHEMAS

    config = get_model_config()
    print(f"Config : {config}\n")

    messages = [
        {"role": "user", "content": "Quel temps fait-il à Hanoi aujourd'hui ?"}
    ]
    resp = call_model(messages, TOOL_SCHEMAS, config)
    print(f"output_type : {resp.output_type}")
    if resp.output_type == "tool_call":
        for tc in resp.tool_calls:
            print(f"  outil : {tc.name}")
            print(f"  args  : {tc.arguments}")
    else:
        print(f"  texte : {resp.text[:200]}")
    print(f"tokens : {resp.tokens_prompt} prompt / {resp.tokens_completion} completion")
