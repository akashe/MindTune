# pipeline_runner.py
import argparse
import logging
import os, sys
import yaml
from datetime import datetime
from typing import Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_manager import DataManager, DataConfig
from model_trainer import ModelTrainer
from evaluator import ModelEvaluator
from config import ConfigManager

class ExperimentPipeline:
    def __init__(self, config_path: str, test_mode: bool = False):
        self.config_path = config_path
        self.test_mode = test_mode
        self.experiment_name = f"diary_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Load experiment configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.config_manager = ConfigManager()
        self.results = {}
        
        # Setup logging
        log_file = f"experiment_{self.experiment_name}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        logging.info(f"🚀 Starting experiment: {self.experiment_name}")
    
    def run_full_experiment(self):
        """Run the complete experiment pipeline"""
        
        try:
            # 1. Prepare data
            datasets = self._prepare_data()
            
            # 2. Train models
            trained_models = self._train_models(datasets)
            
            # 3. Evaluate models
            evaluation_results = self._evaluate_models(trained_models)
            
            # 4. Generate report
            self._generate_final_report(evaluation_results)
            
            logging.info("🎉 Experiment completed successfully!")
            
        except Exception as e:
            logging.error(f"❌ Experiment failed: {e}")
            raise
    
    def _prepare_data(self) -> Dict:
        """Prepare training datasets"""
        
        logging.info("📁 Preparing datasets...")
        
        data_config = DataConfig(**self.config['data'])
        if self.test_mode:
            data_config.test_size = 600  # Small test set
        
        data_manager = DataManager(data_config)
        datasets = data_manager.prepare_datasets(test_mode=self.test_mode)
        
        # Log dataset statistics
        for model_type, dataset in datasets.items():
            train_size = len(dataset['train'])
            val_size = len(dataset['validation'])
            logging.info(f"📊 {model_type}: {train_size} train, {val_size} validation examples")
        
        return datasets
    
    def _train_models(self, datasets: Dict) -> Dict[str, str]:
        """Train all specified models"""
        
        trained_models = {}
        models_to_train = self.config['experiment']['models_to_train']
        
        for model_type in models_to_train:
            logging.info(f"🤖 Training {model_type} model...")
            
            # Get configurations
            model_config = self.config_manager.get_model_config(model_type)
            training_config = self.config_manager.get_training_config(model_type, self.test_mode)
            
            # Setup and train
            trainer = ModelTrainer(model_config, training_config, self.experiment_name)
            model_path = trainer.train(datasets[model_type], model_type)
            
            trained_models[model_type] = model_path
            logging.info(f"✅ {model_type} training complete: {model_path}")

            # Clean up GPU memory between models
            self._cleanup_gpu_memory()

        return trained_models
    
    def _evaluate_models(self, trained_models: Dict[str, str]) -> Dict:
        """Evaluate all trained models"""
        
        logging.info("📊 Starting model evaluation...")
        
        evaluator = ModelEvaluator(self.config['evaluation'])
        benchmarks = self.config['experiment']['benchmarks_to_run']
        
        # Add baseline evaluation (base model without fine-tuning)
        if not self.test_mode:  # Skip baseline in test mode for speed
            base_model = self.config_manager.get_model_config('non_reasoning').base_model
            evaluator.evaluate_model(base_model, "baseline", benchmarks)
        
        # Evaluate fine-tuned models
        for model_name, model_path in trained_models.items():
            evaluator.evaluate_model(model_path, model_name, benchmarks)
        
        return evaluator.results

    def _cleanup_gpu_memory(self):
        """Clean up GPU memory between operations"""
        import gc
        import torch

        try:
            # Force garbage collection
            gc.collect()

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            logging.info("🧹 Pipeline GPU memory cleaned")

        except Exception as e:
            logging.warning(f"⚠️ Pipeline GPU cleanup warning: {e}")

    def _generate_final_report(self, results: Dict):
        """Generate and save final experiment report"""
        
        evaluator = ModelEvaluator(self.config['evaluation'])
        evaluator.results = results
        
        # Generate comparison report
        report = evaluator.generate_comparison_report()
        
        # Save results
        results_file = f"results_{self.experiment_name}.json"
        report_file = f"report_{self.experiment_name}.md"
        
        evaluator.save_results(results_file)
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        logging.info(f"📊 Final report saved: {report_file}")
        print("\n" + "="*50)
        print("EXPERIMENT SUMMARY")
        print("="*50)
        print(report)

def main():
    parser = argparse.ArgumentParser(description="Run diary fine-tuning experiment")
    parser.add_argument("--config", required=True, help="Path to experiment config file")
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode with small dataset")
    
    args = parser.parse_args()
    
    pipeline = ExperimentPipeline(args.config, args.test_mode)
    pipeline.run_full_experiment()

if __name__ == "__main__":
    main()