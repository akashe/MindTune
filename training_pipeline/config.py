# config.py
import yaml
from dataclasses import dataclass
from typing import Dict, List, Optional
import os

@dataclass
class ModelConfig:
    name: str
    base_model: str
    max_seq_length: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]

@dataclass
class TrainingConfig:
    output_dir: str
    max_steps: int
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    warmup_steps: int
    logging_steps: int
    save_steps: int
    eval_steps: int
    load_in_4bit: bool
    
@dataclass
class DataConfig:
    source: str  # "mongodb" or "json"
    mongodb_uri: str
    json_path: str
    test_size: int  # For pipeline testing
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
        return ModelConfig(**config['model'])
    
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