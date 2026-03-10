import os
from ollama import Client as OllamaClient
from groq import AsyncGroq
from json_repair import loads as json_repair_loads

from .config import (
    MODE,
    OLLAMA_MODEL_NAME,
    OLLAMA_BASE_URL,
    GROQ_API_KEY,
    GROQ_MODEL_NAME,
)


ollama_client = None
groq_client = None


def _repair_llm_json(response_content: str) -> dict:
    try:
        repaired_json = json_repair_loads(response_content)
        if isinstance(repaired_json, dict):
            return repaired_json
        raise ValueError("Repaired JSON is not a dictionary as expected.")
    except Exception as e:
        error_detail = f"JSON repair failed. Error: {e}. Raw response (partial): {response_content[:500]}..."
        raise ValueError(error_detail)


async def _call_llm(
    prompt: str, is_json_output: bool = True, temperature: float = 0.0
) -> str:
    global ollama_client, groq_client

    if MODE == "development":
        if not ollama_client:
            ollama_client = OllamaClient(host=OLLAMA_BASE_URL)

        messages = [{"role": "user", "content": prompt}]
        options = {"temperature": temperature}

        response = ollama_client.chat(
            model=OLLAMA_MODEL_NAME, messages=messages, options=options
        )
        return response["message"]["content"]
    else:
        if not groq_client:
            groq_client = AsyncGroq(api_key=GROQ_API_KEY)

        messages = [{"role": "user", "content": prompt}]

        response = await groq_client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"}
            if is_json_output
            else {"type": "text"},
        )
        return response.choices[0].message.content


def init_llm_client():
    global ollama_client, groq_client

    if MODE == "development":
        ollama_client = OllamaClient(host=OLLAMA_BASE_URL)
        ollama_client.list()
        print(
            f"Backend: Mode: {MODE} | LLM Provider: Ollama ({OLLAMA_MODEL_NAME}) at {OLLAMA_BASE_URL}"
        )
    else:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required in production mode")
        print(f"Backend: Mode: {MODE} | LLM Provider: Groq ({GROQ_MODEL_NAME})")
