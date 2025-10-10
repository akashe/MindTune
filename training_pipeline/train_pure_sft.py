#!/usr/bin/env python3
"""
Pure SFT Training: Train model directly on raw diary text.
No instruction formatting - just continuation of your writing style.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_processing.retrieve_notes import retrieve_notes
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch
from datetime import datetime
import logging
from typing import List, Dict
import json

logging.basicConfig(level=logging.INFO)


class PureSFTTrainer:
    """Train model to continue diary entries in your authentic style."""

    def __init__(self, base_model: str = "Qwen/Qwen2.5-3B", output_dir: str = None):
        self.base_model = base_model
        self.output_dir = output_dir or f"./outputs/pure_sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logging.info(f"🚀 Initializing Pure SFT Trainer")
        logging.info(f"Base model: {base_model}")
        logging.info(f"Output dir: {self.output_dir}")

    def load_diary_entries(self) -> List[Dict]:
        """Load raw diary entries from PersonalNotes."""
        logging.info("📖 Loading diary entries...")

        docs = retrieve_notes()

        # Convert to list of dicts
        entries = []
        for doc in docs:
            entry_text = doc.page_content.strip()
            source = doc.metadata.get('source', 'unknown')

            # Filter out very short entries (< 100 chars - likely headers/metadata)
            if len(entry_text) < 100:
                continue

            entries.append({
                "text": entry_text,
                "source": source,
                "length": len(entry_text)
            })

        logging.info(f"✅ Loaded {len(entries)} diary entries")

        # Show statistics
        total_chars = sum(e['length'] for e in entries)
        avg_length = total_chars / len(entries) if entries else 0

        logging.info(f"📊 Statistics:")
        logging.info(f"   Total characters: {total_chars:,}")
        logging.info(f"   Average entry length: {avg_length:.0f} chars")
        logging.info(f"   Min length: {min(e['length'] for e in entries)}")
        logging.info(f"   Max length: {max(e['length'] for e in entries)}")

        return entries

    def prepare_training_data(self, entries: List[Dict], max_length: int = 2048,
                            train_split: float = 0.95) -> tuple:
        """Prepare data for pure causal language modeling.

        Format: Just the raw text - model learns to continue your writing.
        """
        logging.info(f"🔧 Preparing training data (max_length={max_length})...")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Format: Just raw text, no special tokens
        formatted_texts = []
        for entry in entries:
            # Option 1: Pure continuation (no special formatting)
            text = entry['text']

            # Option 2: Add minimal context (date as metadata)
            # text = f"[Diary Entry from {entry['source']}]\n\n{entry['text']}"

            formatted_texts.append(text)

        # Tokenize
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                max_length=max_length,
                padding=False  # Dynamic padding in collator
            )

        # Create dataset
        dataset = Dataset.from_dict({"text": formatted_texts})

        # Tokenize
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing diary entries"
        )

        # Train/val split
        split_dataset = tokenized_dataset.train_test_split(
            test_size=1 - train_split,
            seed=42
        )

        logging.info(f"✅ Prepared {len(split_dataset['train'])} training samples")
        logging.info(f"✅ Prepared {len(split_dataset['test'])} validation samples")

        return split_dataset, tokenizer

    def train(self, batch_size: int = 4, num_epochs: int = 3, learning_rate: float = 2e-5,
             gradient_accumulation_steps: int = 4, max_length: int = 2048):
        """Train model on pure diary text."""

        # Load data
        entries = self.load_diary_entries()

        if len(entries) == 0:
            raise ValueError("No diary entries found! Check your PersonalNotes directory.")

        # Prepare datasets
        datasets, tokenizer = self.prepare_training_data(entries, max_length=max_length)

        # Load model
        logging.info(f"🤖 Loading model: {self.base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,

            # Optimization
            bf16=True,
            gradient_checkpointing=True,

            # Logging
            logging_steps=10,
            logging_dir=f"{self.output_dir}/logs",

            # Evaluation
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,

            # Other
            warmup_steps=100,
            weight_decay=0.01,
            report_to="none",  # Change to "wandb" if you want W&B logging
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )

        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # Causal LM, not masked LM
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets['train'],
            eval_dataset=datasets['test'],
            data_collator=data_collator,
        )

        # Train
        logging.info("🏋️ Starting training...")
        trainer.train()

        # Save final model
        final_model_path = f"{self.output_dir}/final_model"
        logging.info(f"💾 Saving final model to {final_model_path}")
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)

        # Save training info
        with open(f"{self.output_dir}/training_info.json", 'w') as f:
            json.dump({
                "base_model": self.base_model,
                "num_entries": len(entries),
                "num_train_samples": len(datasets['train']),
                "num_val_samples": len(datasets['test']),
                "training_args": {
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "learning_rate": learning_rate,
                    "max_length": max_length,
                    "gradient_accumulation_steps": gradient_accumulation_steps
                },
                "training_type": "pure_sft",
                "description": "Direct continuation training on raw diary text"
            }, f, indent=2)

        logging.info(f"✅ Training complete! Model saved to {final_model_path}")
        return final_model_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train model on raw diary text (Pure SFT)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B", help="Base model to fine-tune")
    parser.add_argument("--output_dir", type=str, help="Output directory for trained model")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--gradient_accumulation", type=int, default=4, help="Gradient accumulation steps")

    args = parser.parse_args()

    trainer = PureSFTTrainer(
        base_model=args.base_model,
        output_dir=args.output_dir
    )

    trainer.train(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        gradient_accumulation_steps=args.gradient_accumulation
    )


if __name__ == "__main__":
    main()
