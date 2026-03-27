#!/usr/bin/env python3
"""Check available models in Groq API."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src" / "models"))

from llm_groq_inference import _get_groq_api_key

try:
    from groq import Groq
except ImportError:
    print("ERROR: Groq SDK not installed. Install with: pip install groq")
    sys.exit(1)


def check_available_models():
    """List all available models in current Groq account."""
    api_key = _get_groq_api_key()
    
    if not api_key:
        print("ERROR: GROQ_API_KEY not found in environment")
        print("Set the API key in .env file: Groq_API_KEY=your_key")
        return
    
    try:
        client = Groq(api_key=api_key)
        models = client.models.list()
        
        print("\n" + "="*80)
        print("AVAILABLE GROQ MODELS")
        print("="*80 + "\n")
        
        for model in models.data:
            print(f"ID: {model.id}")
            if hasattr(model, 'owned_by'):
                print(f"  Owner: {model.owned_by}")
            print()
        
        print("="*80)
        print("\nTo use in .env, set:")
        print("LLM_MODEL_NAMES=model_id_1,model_id_2")
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        print("\nMake sure your Groq API key is valid and has access to the API.")


if __name__ == "__main__":
    check_available_models()
