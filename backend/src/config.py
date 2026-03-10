import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

TECHNICAL_QUESTIONS_COUNT = 7
HR_QUESTIONS_COUNT = 5
MAX_QUESTIONS = TECHNICAL_QUESTIONS_COUNT + HR_QUESTIONS_COUNT
