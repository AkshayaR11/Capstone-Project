"""Shared Google GenAI client and model configuration."""

import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

def create_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    return genai.Client(api_key=api_key) if api_key else None

def generate_gemini_content(client, prompt):
    return client.models.generate_content(model=GEMINI_MODEL_NAME, contents=prompt)
