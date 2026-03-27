#!/usr/bin/env python3
"""Debug script to examine raw responses from LLM models."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "models"))

from llm_groq_inference import (
    build_zero_shot_prompt,
    build_few_shot_prompt,
    select_few_shot_examples,
    _get_groq_api_key,
    _load_default_train_test_split,
)

try:
    from groq import Groq
except ImportError:
    print("ERROR: Groq SDK not installed. Install with: pip install groq")
    sys.exit(1)


def test_model_response(model_name: str, text_sample: str, mode: str = "zero_shot"):
    """Test a single model response."""
    api_key = _get_groq_api_key()
    if not api_key:
        print("ERROR: GROQ_API_KEY not found in environment")
        return None
    
    client = Groq(api_key=api_key)
    
    if mode == "zero_shot":
        prompt = build_zero_shot_prompt(text_sample)
    else:
        # For few-shot, we need examples
        try:
            train_texts, test_texts, train_labels, test_labels = _load_default_train_test_split()
            examples = select_few_shot_examples(train_texts, train_labels, max_per_class=3)
            prompt = build_few_shot_prompt(text_sample, examples)
        except Exception as e:
            print(f"Error loading examples: {e}")
            prompt = build_zero_shot_prompt(text_sample)
    
    print(f"\n{'='*80}")
    print(f"Model: {model_name}")
    print(f"Mode: {mode}")
    print(f"{'='*80}")
    
    print(f"\n[PROMPT]:\n{prompt}")
    print(f"\n{'-'*80}")
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            max_tokens=8,
            messages=[{"role": "user", "content": prompt}],
        )
        
        raw_response = response.choices[0].message.content
        print(f"\n[RAW RESPONSE]:\n{repr(raw_response)}")
        print(f"\n[PARSED]:\n{raw_response}")
        
        return raw_response
    except Exception as e:
        print(f"\n[ERROR]:\n{type(e).__name__}: {e}")
        return None


def main():
    """Test both models and modes."""
    models = [
        "openai/gpt-oss-20b",
        "llama-3.3-70b-versatile",
    ]
    
    # Sample Chinese text
    samples = [
        "符合方法命名標準",  # Meaningful
        "沒有完成作業",      # Meaningful
        "很好!讚",          # Not meaningful
        "是",               # Not meaningful
    ]
    
    for model in models:
        for mode in ["zero_shot", "few_shot"]:
            for sample in samples[:1]:  # Test with first sample
                test_model_response(model, sample, mode)
                print("\n\n")


if __name__ == "__main__":
    main()
