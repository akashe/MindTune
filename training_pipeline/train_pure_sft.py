#!/usr/bin/env python3
"""
Pure SFT Training: Train model directly on raw diary text using Unsloth + LoRA.
No instruction formatting - just continuation of your writing style.
Memory-efficient for 16GB GPUs.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_processing.retrieve_notes import retrieve_notes
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset
import torch
from datetime import datetime
import logging
from typing import List, Dict
import json

logging.basicConfig(level=logging.INFO)


class PureSFTTrainer:
    """Train model to continue diary entries in your authentic style using Unsloth + LoRA."""

    def __init__(self, base_model: str = "Qwen/Qwen2.5-3B",
                 output_dir: str = None,
                 max_seq_length: int = 2048):
        self.base_model = base_model
        self.max_seq_length = max_seq_length
        self.output_dir = output_dir or f"./outputs/pure_sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model = None
        self.tokenizer = None

        logging.info(f"🚀 Initializing Pure SFT Trainer (Unsloth + LoRA + FP16)")
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

    def setup_model(self, lora_r: int = 16, lora_alpha: int = 16):
        """Load model with Unsloth + LoRA (FP16, no quantization)."""
        logging.info(f"🤖 Loading model with Unsloth: {self.base_model}")

        # Load model in FP16 (no 4-bit quantization)
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model,
            max_seq_length=self.max_seq_length,
            dtype=None,  # Auto-detect (will use FP16/BF16)
            load_in_4bit=False,  # No quantization - full FP16
            device_map="auto",
        )

        # Add LoRA adapters
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=lora_alpha,
            lora_dropout=0.0,  # Optimized for Unsloth
            bias="none",
            use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
            random_state=3407,
        )

        logging.info("✅ Model loaded with LoRA adapters (FP16)")

    def prepare_training_data(self, entries: List[Dict], train_split: float = 0.95):
        """Prepare data for pure causal language modeling."""
        logging.info(f"🔧 Preparing training data...")

        # Format: Just raw text for continuation
        formatted_data = []
        for entry in entries:
            formatted_data.append({"text": entry['text']})

        # Create dataset
        dataset = Dataset.from_list(formatted_data)

        # Train/val split
        split_dataset = dataset.train_test_split(
            test_size=1 - train_split,
            seed=42
        )

        logging.info(f"✅ Prepared {len(split_dataset['train'])} training samples")
        logging.info(f"✅ Prepared {len(split_dataset['test'])} validation samples")

        return split_dataset

    def train(self,
              batch_size: int = 2,
              num_epochs: int = 1,  # Changed default to 1 to avoid overfitting
              learning_rate: float = 2e-4,
              gradient_accumulation_steps: int = 4,
              lora_r: int = 16,
              lora_alpha: int = 16):
        """Train model on pure diary text."""

        # Load data
        entries = self.load_diary_entries()

        if len(entries) == 0:
            raise ValueError("No diary entries found! Check your PersonalNotes directory.")

        # Setup model
        self.setup_model(lora_r=lora_r, lora_alpha=lora_alpha)

        # Prepare datasets
        datasets = self.prepare_training_data(entries)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,

            # Optimization - use FP16 only
            fp16=True,
            bf16=False,
            optim="adamw_8bit",  # Memory efficient optimizer
            weight_decay=0.01,
            warmup_steps=5,

            # Logging
            logging_steps=10,
            logging_dir=f"{self.output_dir}/logs",

            # Evaluation
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=2,  # Keep only 2 checkpoints
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",

            # Other
            report_to="none",
            seed=3407,
        )

        # Use SFTTrainer for pure text continuation
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=datasets['train'],
            eval_dataset=datasets['test'],
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            args=training_args,
            packing=False,  # Don't pack sequences for diary entries
        )

        # Train
        logging.info("🏋️ Starting training...")
        logging.info(f"⚠️  Training for {num_epochs} epoch(s) to avoid overfitting")
        trainer.train()

        # Save LoRA adapters (for archival)
        lora_model_path = f"{self.output_dir}/lora_adapters"
        logging.info(f"💾 Saving LoRA adapters to {lora_model_path}")
        self.model.save_pretrained(lora_model_path)
        self.tokenizer.save_pretrained(lora_model_path)

        # Save merged model (primary model for inference)
        final_model_path = f"{self.output_dir}/final_model"
        logging.info(f"🔄 Merging LoRA adapters and saving merged model...")

        # Merge LoRA adapters into base model
        merged_model = self.model.merge_and_unload()

        # Save merged model
        merged_model.save_pretrained(
            final_model_path,
            safe_serialization=True,
            max_shard_size="5GB"
        )
        self.tokenizer.save_pretrained(final_model_path)

        logging.info(f"✅ Merged model saved to {final_model_path}")

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
                    "max_length": self.max_seq_length,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "lora_r": lora_r,
                    "lora_alpha": lora_alpha
                },
                "training_type": "pure_sft_lora",
                "description": "Direct continuation training on raw diary text using Unsloth + LoRA"
            }, f, indent=2)

        logging.info(f"✅ Training complete!")
        logging.info(f"   LoRA adapters: {lora_model_path}")
        logging.info(f"   Final merged model: {final_model_path}")
        return final_model_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train model on raw diary text (Pure SFT with Unsloth + LoRA + FP16)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B",
                       help="Base model to fine-tune")
    parser.add_argument("--output_dir", type=str, help="Output directory for trained model")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs (1 recommended to avoid overfitting)")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--gradient_accumulation", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")

    args = parser.parse_args()

    trainer = PureSFTTrainer(
        base_model=args.base_model,
        output_dir=args.output_dir,
        max_seq_length=args.max_length
    )

    trainer.train(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )


if __name__ == "__main__":
    main()
