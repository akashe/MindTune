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

        # Determine dtype from config or auto-detect
        if self.model_config.dtype == "bfloat16":
            dtype = torch.bfloat16
        elif self.model_config.dtype == "float16":
            dtype = torch.float16
        else:
            # Auto-detect based on GPU capability
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        # Load model
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_config.base_model,
            max_seq_length=self.model_config.max_seq_length,
            dtype=None,
            load_in_4bit=self.training_config.load_in_4bit,
            use_cache=self.model_config.use_cache,
            device_map=self.model_config.device_map,
            trust_remote_code=self.model_config.trust_remote_code,
            attn_implementation=self.model_config.attn_implementation
        )
        
        # Add LoRA adapters
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.model_config.lora_r,
            target_modules=self.model_config.target_modules,
            lora_alpha=self.model_config.lora_alpha,
            lora_dropout=self.model_config.lora_dropout,
            bias="none",
            use_gradient_checkpointing=self.training_config.use_gradient_checkpointing,
            random_state=3407,
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

        # Auto-detect precision settings if not specified
        if self.training_config.fp16 is None and self.training_config.bf16 is None:
            use_fp16 = not torch.cuda.is_bf16_supported()
            use_bf16 = torch.cuda.is_bf16_supported()
        else:
            use_fp16 = self.training_config.fp16 or False
            use_bf16 = self.training_config.bf16 or False
        
        # Setup wandb if available
        report_to = self.training_config.report_to
        if report_to != "none":
            try:

                # Generate run name if not provided
                final_run_name = (self.training_config.run_name or 
                                f"{self.experiment_name}_{run_name}_{self.model_config.name}")

                wandb.init(
                project=self.training_config.wandb_project or "finetune-my-diary",
                entity=self.training_config.wandb_entity,  # Your username
                name=final_run_name,
                tags=self.training_config.wandb_tags or [],
                config={
                        # Log all important parameters
                        "model_name": self.model_config.base_model,
                        "experiment": self.experiment_name,
                        "run_type": run_name,
                        "lora_r": self.model_config.lora_r,
                        "lora_alpha": self.model_config.lora_alpha,
                        "batch_size": self.training_config.batch_size,
                        "learning_rate": self.training_config.learning_rate,
                        "max_steps": self.training_config.max_steps,
                        "optimizer": self.training_config.optim,
                        "max_seq_length": self.model_config.max_seq_length,
                        "attention_impl": self.model_config.attn_implementation
                    }
                )
                
                # Log additional info
                wandb.log({"gpu_info": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"})
                report_to = "wandb"
            except:
                logging.warning("⚠️  Wandb not available, logging locally only")
                report_to = "none"
        
        # Training arguments
        training_args = TrainingArguments(
            per_device_train_batch_size=self.training_config.batch_size,
            per_device_eval_batch_size=self.training_config.batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            warmup_steps=self.training_config.warmup_steps,
            max_steps=self.training_config.max_steps,
            learning_rate=self.training_config.learning_rate,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            fp16=use_fp16,
            bf16=use_bf16,
            logging_steps=self.training_config.logging_steps,
            save_steps=self.training_config.save_steps,
            eval_steps=self.training_config.eval_steps,
            eval_strategy=self.training_config.evaluation_strategy,
            save_strategy=self.training_config.save_strategy,
            save_only_model=self.training_config.save_only_model,
            save_total_limit=self.training_config.save_total_limit,
            metric_for_best_model=self.training_config.metric_for_best_model,
            greater_is_better=self.training_config.greater_is_better,
            load_best_model_at_end=self.training_config.load_best_model_at_end,
            output_dir=output_dir,
            optim=self.training_config.optim,
            weight_decay=self.training_config.weight_decay,
            max_grad_norm=self.training_config.max_grad_norm,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            seed=self.training_config.seed,
            data_seed=self.training_config.data_seed,
            dataloader_num_workers=self.training_config.dataloader_num_workers,
            dataloader_pin_memory=self.training_config.dataloader_pin_memory,
            group_by_length=self.training_config.group_by_length,
            remove_unused_columns=self.training_config.remove_unused_columns,
            dataloader_drop_last=self.training_config.dataloader_drop_last,
            report_to=report_to,
            run_name=final_run_name if report_to == "wandb" else None,
            prediction_loss_only=True
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
        
        # Save LoRA adapters (small, for archival)
        final_model_path = os.path.join(output_dir, "final_model")
        trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)

        # Save merged model for fast evaluation
        merged_model_path = os.path.join(output_dir, "merged_model")
        try:
            logging.info("🔄 Merging LoRA adapters for fast evaluation...")

            # Use Unsloth's merge_and_unload to get full merged model
            merged_model = self.model.merge_and_unload()

            # Save merged model in fp16 for fast evaluation
            merged_model.save_pretrained(
                merged_model_path,
                safe_serialization=True,
                max_shard_size="5GB"
            )
            self.tokenizer.save_pretrained(merged_model_path)

            logging.info(f"📦 Merged model saved to: {merged_model_path}")

            # Verify the merge worked by checking for model.safetensors
            model_file = os.path.join(merged_model_path, "model.safetensors")
            if os.path.exists(model_file):
                size_mb = os.path.getsize(model_file) / (1024*1024)
                logging.info(f"✅ Merged model file: {size_mb:.1f}MB")
            else:
                logging.warning("⚠️ No model.safetensors found - merge may have failed")

        except Exception as e:
            logging.warning(f"⚠️ Could not save merged model: {e}")
            logging.info("Will use LoRA adapters for evaluation instead")
            # Fallback: just copy adapter files with a note
            try:
                import shutil
                shutil.copytree(final_model_path, merged_model_path)
                logging.info("📋 Copied LoRA adapters as fallback")
            except:
                pass


        logging.info(f"✅ Training complete. LoRA adapters: {final_model_path}")

        # Cleanup GPU memory
        self._cleanup_gpu_memory(trainer)

        # Cleanup wandb
        if report_to != "none":
            try:
                wandb.finish()
            except:
                pass

        return merged_model_path

    def _cleanup_gpu_memory(self, trainer=None):
        """Clean up GPU memory after training"""
        import gc

        try:
            # Delete trainer and model references
            if trainer is not None:
                del trainer

            if self.model is not None:
                del self.model
                self.model = None

            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None

            # Force garbage collection
            gc.collect()

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            logging.info("🧹 GPU memory cleaned up")

        except Exception as e:
            logging.warning(f"⚠️ GPU cleanup warning: {e}")
    