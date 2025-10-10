#!/usr/bin/env python3
"""
Side-by-side comparison of base model vs fine-tuned model.
Tests on custom prompts to see qualitative differences.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from typing import List, Dict
import json


class ModelComparator:
    def __init__(self, base_model_path: str, finetuned_model_path: str):
        """Initialize both models for comparison."""
        print(f"Loading base model: {base_model_path}")
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        print(f"Loading fine-tuned model: {finetuned_model_path}")
        self.finetuned_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
        self.finetuned_model = AutoModelForCausalLM.from_pretrained(
            finetuned_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        print("✅ Both models loaded successfully\n")

    def generate_response(self, model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate response from a model."""
        # Format as instruction (adjust based on your training format)
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the response part (after "### Response:")
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()

        return response

    def compare_prompts(self, prompts: List[str], max_new_tokens: int = 512, save_to: str = None):
        """Compare both models on a list of prompts."""
        results = []

        for i, prompt in enumerate(prompts):
            print(f"\n{'='*80}")
            print(f"PROMPT {i+1}: {prompt}")
            print(f"{'='*80}\n")

            print("🔵 BASE MODEL:")
            base_response = self.generate_response(self.base_model, self.base_tokenizer, prompt, max_new_tokens)
            print(base_response)

            print("\n🟢 FINE-TUNED MODEL:")
            finetuned_response = self.generate_response(self.finetuned_model, self.finetuned_tokenizer, prompt, max_new_tokens)
            print(finetuned_response)

            results.append({
                "prompt": prompt,
                "base_response": base_response,
                "finetuned_response": finetuned_response
            })

        if save_to:
            with open(save_to, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✅ Results saved to {save_to}")

        return results


def get_test_prompts() -> List[str]:
    """Return test prompts - customize these based on your diary insights."""
    return [
        # Personal reasoning patterns
        "I'm feeling overwhelmed with multiple projects. How should I prioritize what to work on?",

        # Emotional intelligence
        "A close friend seems distant lately. How should I approach this situation?",

        # Meta-thinking
        "I keep procrastinating on important tasks. What's the underlying reason and how can I address it?",

        # Philosophical/spiritual
        "What makes a life well-lived?",

        # AI/Technical (if in your diary)
        "How should we think about the alignment problem in AI development?",

        # Creativity
        "I'm stuck on a creative project. What strategies can help me break through?",

        # General reasoning (benchmark-style)
        "If all roses are flowers and some flowers fade quickly, what can we conclude about roses?",

        # Ethical reasoning
        "Is it ethical to lie to protect someone's feelings? Explain your reasoning.",

        # Social situation
        "At a team meeting, someone takes credit for my idea. How should I handle this professionally?",

        # Self-reflection (very personal)
        "When I feel anxious about the future, what mental frameworks help me gain perspective?"
    ]


def main():
    parser = argparse.ArgumentParser(description="Compare base model vs fine-tuned model side-by-side")
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model (e.g., Qwen/Qwen2.5-3B)")
    parser.add_argument("--finetuned_model", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--prompts_file", type=str, help="JSON file with custom prompts (optional)")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--save_to", type=str, default="comparison_results.json", help="Save results to JSON")

    args = parser.parse_args()

    # Initialize comparator
    comparator = ModelComparator(args.base_model, args.finetuned_model)

    # Get prompts
    if args.prompts_file:
        with open(args.prompts_file, 'r') as f:
            prompts_data = json.load(f)
            prompts = prompts_data if isinstance(prompts_data, list) else prompts_data.get("prompts", [])
    else:
        prompts = get_test_prompts()

    # Run comparison
    comparator.compare_prompts(prompts, max_new_tokens=args.max_tokens, save_to=args.save_to)


if __name__ == "__main__":
    main()
