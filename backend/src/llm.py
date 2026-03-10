import os
from ollama import Client as OllamaClient
from json_repair import loads as json_repair_loads

from .config import LLM_PROVIDER, OLLAMA_MODEL_NAME, OLLAMA_BASE_URL


ollama_client = None


def _repair_llm_json(response_content: str) -> dict:
    try:
        repaired_json = json_repair_loads(response_content)
        if not isinstance(repaired_json, dict) and not isinstance(repaired_json, list):
            raise ValueError("Repaired JSON is not a dictionary or list as expected.")
        return repaired_json
    except Exception as e:
        error_detail = f"JSON repair failed. Error: {e}. Raw response (partial): {response_content[:500]}..."
        raise ValueError(error_detail)


async def _call_llm(prompt: str, is_json_output: bool = True, temperature: float = 0.0) -> str:
    global ollama_client

    if LLM_PROVIDER != "ollama":
        raise ValueError(f"Invalid LLM_PROVIDER: {LLM_PROVIDER}. Only 'ollama' is supported.")

    if not ollama_client:
        ollama_client = OllamaClient(host=OLLAMA_BASE_URL)

    messages = [{"role": "user", "content": prompt}]
    options = {"temperature": temperature}

    response = ollama_client.chat(
        model=OLLAMA_MODEL_NAME,
        messages=messages,
        options=options
    )
    return response['message']['content']


def init_llm_client():
    global ollama_client
    ollama_client = OllamaClient(host=OLLAMA_BASE_URL)
    ollama_client.list()
    print(f"Backend: LLM Provider: Ollama ({OLLAMA_MODEL_NAME}) at {OLLAMA_BASE_URL}")
