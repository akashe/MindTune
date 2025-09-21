# test_pipeline.py
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_manager import DataManager, DataConfig
from model_trainer import ModelTrainer
from evaluator import ModelEvaluator
from config import ConfigManager



def test_pipeline():
    """Test the complete pipeline with small dataset"""
    
    logging.basicConfig(level=logging.INFO)
    logging.info("🧪 Starting pipeline test...")
    
    # Setup data config for testing
    data_config = DataConfig(
        source="mongodb",  # or "json"
        mongodb_uri="mongodb://localhost:27017/",
        json_path="test_data.json",
        test_size=20,  # Small test set
        validation_split=0.2,
        max_length=1024
    )
    
    # Load configurations
    config_manager = ConfigManager()
    
    # Setup data manager
    data_manager = DataManager(data_config)
    datasets = data_manager.prepare_datasets(test_mode=True)
    
    logging.info(f"📊 Prepared datasets: {list(datasets.keys())}")
    
    # Test training for each model type
    experiment_name = "pipeline_test"
    
    for model_type in ["non_reasoning"]:  # Test one model first
        logging.info(f"🤖 Testing {model_type} model...")
        
        # Get configurations
        model_config = config_manager.get_model_config(model_type)
        training_config = config_manager.get_training_config(model_type, test_mode=True)
        
        # Setup trainer
        trainer = ModelTrainer(model_config, training_config, experiment_name)
        
        # Train model
        model_path = trainer.train(datasets[model_type], f"test_{model_type}")
        
        logging.info(f"✅ Test training complete: {model_path}")
        
        # Test evaluation (skip for now in quick test)
        # evaluator = ModelEvaluator({})
        # evaluator.evaluate_model(model_path, f"test_{model_type}", ["gsm8k"])
    
    logging.info("🎉 Pipeline test successful!")

if __name__ == "__main__":
    test_pipeline()