# evaluator.py
import subprocess
import json
import os
import logging
from typing import Dict, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelEvaluator:
    def __init__(self, evaluation_config: Dict):
        self.config = evaluation_config
        self.results = {}
    
    def evaluate_model(self, model_path: str, model_name: str, benchmarks: List[str]) -> Dict:
        """Evaluate model on specified benchmarks"""
        
        logging.info(f"📊 Evaluating {model_name} on benchmarks: {benchmarks}")
        
        model_results = {}
        
        for benchmark in benchmarks:
            try:
                logging.info(f"🔄 Running {benchmark} evaluation...")
                
                if benchmark == "gsm8k":
                    score = self._evaluate_gsm8k(model_path)
                elif benchmark == "hellaswag":
                    score = self._evaluate_hellaswag(model_path)
                elif benchmark == "arc_easy":
                    score = self._evaluate_arc_easy(model_path)
                elif benchmark == "mmlu_subset":
                    score = self._evaluate_mmlu_subset(model_path)
                elif benchmark == "social_iqa":
                    score = self._evaluate_social_iqa(model_path)
                elif benchmark == "truthfulqa":
                    score = self._evaluate_truthfulqa(model_path)
                elif benchmark == "winogrande":
                    score = self._evaluate_winogrande(model_path)
                elif benchmark == "hhh_eval":
                    score = self._evaluate_hhh_eval(model_path)
                else:
                    logging.warning(f"⚠️  Unknown benchmark: {benchmark}")
                    continue
                
                model_results[benchmark] = score
                logging.info(f"✅ {benchmark}: {score:.3f}")
                
            except Exception as e:
                logging.error(f"❌ Failed to evaluate {benchmark}: {e}")
                model_results[benchmark] = 0.0
        
        self.results[model_name] = model_results
        return model_results
    
    def _evaluate_gsm8k(self, model_path: str) -> float:
        """Evaluate on GSM8K math reasoning"""
        
        cmd = [
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path}",
            "--tasks", "gsm8k",
            "--batch_size", "8",
            "--num_fewshot", "0",
            "--output_path", f"eval_results_{os.path.basename(model_path)}_gsm8k"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
            # Parse results from output
            output_file = f"eval_results_{os.path.basename(model_path)}_gsm8k/results.json"
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    results = json.load(f)
                return results.get('results', {}).get('gsm8k', {}).get('acc', 0.0)
            
        except subprocess.TimeoutExpired:
            logging.error("GSM8K evaluation timed out")
        except Exception as e:
            logging.error(f"GSM8K evaluation failed: {e}")
        
        return 0.0
    
    def _evaluate_hellaswag(self, model_path: str) -> float:
        """Evaluate on HellaSwag common sense reasoning"""
        
        cmd = [
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path}",
            "--tasks", "hellaswag",
            "--batch_size", "8",
            "--num_fewshot", "0"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
            # Parse accuracy from output
            # Implementation depends on lm-eval output format
            return self._parse_accuracy_from_output(result.stdout, "hellaswag")
        except:
            return 0.0
    
    def _evaluate_arc_easy(self, model_path: str) -> float:
        """Evaluate on ARC-Easy"""
        
        cmd = [
            "python", "-m", "lm_eval",
            "--model", "hf", 
            "--model_args", f"pretrained={model_path}",
            "--tasks", "arc_easy",
            "--batch_size", "8",
            "--num_fewshot", "0"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
            return self._parse_accuracy_from_output(result.stdout, "arc_easy")
        except:
            return 0.0
    
    def _evaluate_mmlu_subset(self, model_path: str) -> float:
        """Evaluate on MMLU subset (elementary math)"""
        
        cmd = [
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path}",
            "--tasks", "mmlu_elementary_mathematics",
            "--batch_size", "8",
            "--num_fewshot", "0"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
            return self._parse_accuracy_from_output(result.stdout, "mmlu_elementary_mathematics")
        except:
            return 0.0

    def _evaluate_social_iqa(self, model_path: str) -> float:
        """Evaluate on SocialIQA for social reasoning"""

        cmd = [
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path}",
            "--tasks", "social_iqa",
            "--batch_size", "4",
            "--num_fewshot", "0"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
            return self._parse_accuracy_from_output(result.stdout, "social_iqa")
        except:
            logging.error("SocialIQA evaluation failed")
            return 0.0

    def _evaluate_truthfulqa(self, model_path: str) -> float:
        """Evaluate on TruthfulQA for truthfulness"""

        cmd = [
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path}",
            "--tasks", "truthfulqa_mc2",
            "--batch_size", "2",
            "--num_fewshot", "0"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            return self._parse_accuracy_from_output(result.stdout, "truthfulqa_mc2")
        except:
            logging.error("TruthfulQA evaluation failed")
            return 0.0

    def _evaluate_winogrande(self, model_path: str) -> float:
        """Evaluate on Winogrande for commonsense reasoning"""

        cmd = [
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path}",
            "--tasks", "winogrande",
            "--batch_size", "8",
            "--num_fewshot", "0"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
            return self._parse_accuracy_from_output(result.stdout, "winogrande")
        except:
            logging.error("Winogrande evaluation failed")
            return 0.0

    def _evaluate_hhh_eval(self, model_path: str) -> float:
        """Evaluate on HHH (Helpful, Harmless, Honest) using Unitxt HH-RLHF"""

        # Use Unitxt's HH-RLHF evaluation through lm_eval
        cmd = [
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path}",
            "--tasks", "unitxt[card=cards.hh_rlhf,template=templates.classification.multi_class.relation.default,format=formats.chatapi]",
            "--batch_size", "2",
            "--num_fewshot", "0"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
            return self._parse_accuracy_from_output(result.stdout, "unitxt")
        except Exception as e:
            logging.error(f"HH-RLHF evaluation failed: {e}")
            logging.warning("Falling back to ethics proxy evaluation")

            # Fallback to ethics evaluation
            fallback_cmd = [
                "python", "-m", "lm_eval",
                "--model", "hf",
                "--model_args", f"pretrained={model_path}",
                "--tasks", "hendrycksTest-moral_scenarios",
                "--batch_size", "4",
                "--num_fewshot", "0"
            ]

            try:
                fallback_result = subprocess.run(fallback_cmd, capture_output=True, text=True, timeout=1200)
                return self._parse_accuracy_from_output(fallback_result.stdout, "hendrycksTest-moral_scenarios")
            except:
                return 0.0

    def _parse_accuracy_from_output(self, output: str, task: str) -> float:
        """Parse accuracy from lm-eval output"""
        
        lines = output.split('\n')
        for line in lines:
            if task in line and 'acc' in line:
                try:
                    # Extract number from line like "task_name: acc: 0.234"
                    parts = line.split(':')
                    for part in parts:
                        if 'acc' in part:
                            score = float(part.split()[-1])
                            return score
                except:
                    continue
        
        return 0.0
    
    def generate_comparison_report(self) -> str:
        """Generate comparison report between models"""
        
        if len(self.results) < 2:
            return "Need at least 2 models to compare"
        
        models = list(self.results.keys())
        benchmarks = list(self.results[models[0]].keys())
        
        report = f"""
# Diary Finetuning Experiment Results

## Model Comparison

| Benchmark | {' | '.join(models)} |
|-----------|{'-|' * len(models)}
"""
        
        for benchmark in benchmarks:
            row = f"| {benchmark} |"
            for model in models:
                score = self.results[model].get(benchmark, 0.0)
                row += f" {score:.3f} |"
            report += row + "\n"
        
        # Calculate improvements
        if len(models) == 2:
            base_model, personal_model = models
            report += "\n## Performance Changes\n\n"
            
            for benchmark in benchmarks:
                base_score = self.results[base_model][benchmark]
                personal_score = self.results[personal_model][benchmark]
                
                if base_score > 0:
                    change = ((personal_score - base_score) / base_score) * 100
                    direction = "↑" if change > 0 else "↓"
                    report += f"- **{benchmark}**: {direction} {abs(change):.1f}%\n"
        
        return report
    
    def save_results(self, output_path: str):
        """Save results to JSON file"""
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logging.info(f"📊 Results saved to: {output_path}")