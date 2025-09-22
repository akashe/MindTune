# model_trainer.py
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
import os
import logging
from typing import Dict
import wandb

class ModelTrainer:
    def __init__(self, model_config, training_config, experiment_name: str):
        self.model_config = model_config
        self.training_config = training_config
        self.experiment_name = experiment_name
        self.model = None
        self.tokenizer = None
        
    def setup_model(self):
        """Initialize model and tokenizer with LoRA"""
        logging.info(f"🤖 Setting up model: {self.model_config.base_model}")
        
        # Load model
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_config.base_model,
            max_seq_length=self.model_config.max_seq_length,
            dtype=None,
            load_in_4bit=self.training_config.load_in_4bit,
        )
        
        # Add LoRA adapters
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.model_config.lora_r,
            target_modules=self.model_config.target_modules,
            lora_alpha=self.model_config.lora_alpha,
            lora_dropout=self.model_config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        
        logging.info("✅ Model setup complete")
    
    def train(self, dataset, run_name: str) -> str:
        """Train the model and return output directory"""
        
        if self.model is None:
            self.setup_model()
        
        # Setup output directory
        output_dir = os.path.join(
            self.training_config.output_dir,
            f"{self.experiment_name}_{run_name}"
        )
        
        # Setup wandb if available
        try:
            wandb.init(
                project="diary-finetuning-experiment",
                name=f"{self.experiment_name}_{run_name}",
                config={
                    "model": self.model_config.base_model,
                    "experiment": self.experiment_name,
                    "run": run_name
                }
            )
            report_to = "wandb"
        except:
            logging.warning("⚠️  Wandb not available, logging locally only")
            report_to = "none"
        
        # Training arguments
        training_args = TrainingArguments(
            per_device_train_batch_size=self.training_config.batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            warmup_steps=self.training_config.warmup_steps,
            max_steps=self.training_config.max_steps,
            learning_rate=self.training_config.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=self.training_config.logging_steps,
            save_steps=self.training_config.save_steps,
            eval_steps=self.training_config.eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            output_dir=output_dir,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            dataloader_num_workers=0,
            report_to=report_to,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        # Setup trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            dataset_text_field="text",
            max_seq_length=self.model_config.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=training_args,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        )
        
        # Train!
        logging.info(f"🚀 Starting training: {run_name}")
        trainer.train()
        
        # Save model
        final_model_path = os.path.join(output_dir, "final_model")
        trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        logging.info(f"✅ Training complete. Model saved to: {final_model_path}")
        
        # Cleanup wandb
        try:
            wandb.finish()
        except:
            pass
        
        return final_model_path
    