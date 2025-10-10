#!/usr/bin/env python3
"""
Test the pure SFT model by giving it diary-style prompts and seeing if it continues in your style.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse


def test_diary_continuation(model_path: str, prompts: list = None):
    """Test model's ability to continue diary entries."""

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Default test prompts (start of diary entries)
    if prompts is None:
        prompts = [
            "Today I've been thinking about",
            "I realized something important about myself:",
            "When I feel overwhelmed, I notice that",
            "The key to understanding this problem is",
            "I've been struggling with",
            "What really matters in this situation is"
        ]

    print("\n" + "="*80)
    print("PURE SFT MODEL - DIARY CONTINUATION TEST")
    print("="*80)

    for i, prompt in enumerate(prompts):
        print(f"\n{'─'*80}")
        print(f"PROMPT {i+1}: {prompt}")
        print(f"{'─'*80}\n")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        continuation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(continuation)
        print()


def main():
    parser = argparse.ArgumentParser(description="Test pure SFT model on diary-style continuations")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--prompts", type=str, nargs="+", help="Custom prompts to test")

    args = parser.parse_args()

    test_diary_continuation(args.model_path, args.prompts)


if __name__ == "__main__":
    main()
