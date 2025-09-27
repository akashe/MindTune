# config.py
import yaml
from dataclasses import dataclass
from typing import Dict, List, Optional
import os
import torch
from utils import get_optimal_attention_implementation

@dataclass
class ModelConfig:
    name: str
    base_model: str
    max_seq_length: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]
    use_cache: bool = False
    device_map: str = "auto"
    trust_remote_code: bool = True
    attn_implementation: str = "auto"  
    dtype: Optional[str] = None  # "bfloat16", "float16", or None for auto

@dataclass
class TrainingConfig:
    output_dir: str
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    warmup_steps: int
    load_in_4bit: bool
    auto_find_batch_size: bool

    # Training duration control
    use_epochs: bool = True
    num_train_epochs: Optional[int] = 1
    max_steps: Optional[int] = None
    eval_log_frequency: float = 0.2

    # These can be None and will be auto-calculated
    logging_steps: Optional[int] = None
    save_steps: Optional[int] = None
    eval_steps: Optional[int] = None
    # Optimizer settings
    optim: str = "adamw_torch"  # Better for fp16 training
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "linear"
    
    # Data loading optimization
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = False
    group_by_length: bool = False
    remove_unused_columns: bool = True
    dataloader_drop_last: bool = True
    
    # Memory and performance
    gradient_checkpointing: bool = True
    fp16: Optional[bool] = None  # None for auto-detection
    bf16: Optional[bool] = None  # None for auto-detection
    
    # LoRA specific training settings  
    use_gradient_checkpointing: str = "unsloth"  # or False
    
    # Evaluation settings
    evaluation_strategy: str = "steps"
    load_best_model_at_end: bool = True
    save_only_model: bool = True
    save_strategy: str = "no"
    save_total_limit: int = 1  
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Reporting
    report_to: str = "none"
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    run_name: Optional[str] = None

    # seeds
    seed: int = 42
    data_seed: int = 3407
    
@dataclass
class DataConfig:
    source: str  # "mongodb" or "json"
    mongodb_uri: str
    json_path: str
    test_size: int  # For pipeline testing only
    validation_split: float
    max_length: int

@dataclass
class ExperimentConfig:
    experiment_name: str
    models_to_train: List[str]  # ["non_reasoning", "reasoning"]
    benchmarks_to_run: List[str]
    save_results_to: str

class ConfigManager:
    def __init__(self, config_dir="configs"):
        self.config_dir = config_dir
        
    def load_experiment_config(self, config_path: str) -> Dict:
        """Load complete experiment configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_model_config(self, model_type: str) -> ModelConfig:
        """Get configuration for specific model type"""
        config_file = os.path.join(self.config_dir, f"{model_type}.yaml")
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        model_config = ModelConfig(**config['model'])
        
        # Auto-detect attention implementation if set to "auto"
        if model_config.attn_implementation == "auto":
            model_config.attn_implementation = get_optimal_attention_implementation()
        
        return model_config
    
    def get_training_config(self, model_type: str, test_mode: bool = False) -> TrainingConfig:
        """Get training configuration, with test mode overrides"""
        config_file = os.path.join(self.config_dir, f"{model_type}.yaml")
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        training_config = TrainingConfig(**config['training'])
        
        # Override for test mode
        if test_mode:
            training_config.max_steps = 10
            training_config.eval_steps = 5
            training_config.save_steps = 5
            training_config.logging_steps = 1
            
        return training_config