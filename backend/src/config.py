import os
from dotenv import load_dotenv

load_dotenv()

MODE = os.getenv("MODE", "development").lower()

OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

TECHNICAL_QUESTIONS_COUNT = 7
HR_QUESTIONS_COUNT = 5
MAX_QUESTIONS = TECHNICAL_QUESTIONS_COUNT + HR_QUESTIONS_COUNT
