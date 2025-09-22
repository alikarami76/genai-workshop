import os
import sys
from pathlib import Path
from langchain_groq import ChatGroq


DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


def langchain_groq_llm_connector(API_KEY: str, GROQ_MODEL_NAME: str = DEFAULT_GROQ_MODEL):
    """Connect to the LLM service using the provided API key."""
    if not API_KEY:
        raise RuntimeError(
            "API_KEY is not set. Put it write a correct API KEY extracted from Groq panel."
        )
    # Placeholder for actual connection logic
    print("Connecting to Groq LLM service…")
    llm = ChatGroq(
        groq_api_key=API_KEY, # type: ignore
        model_name=GROQ_MODEL_NAME, # type: ignore
        temperature=0, timeout=60, max_retries=2,
    ) # type: ignore
    print(f"Connected to Groq model '{GROQ_MODEL_NAME}'.")
    return llm
