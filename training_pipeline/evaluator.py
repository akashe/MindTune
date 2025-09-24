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
            "--batch_size", "auto:3",
            "--num_fewshot", "0",
            "--output_path", f"eval_results_{os.path.basename(model_path)}_gsm8k",
            "--verbosity", "DEBUG"
        ]

        try:
            # Stream output in real-time
            logging.info(f"Running command: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Stream output line by line
            output_lines = []
            for line in process.stdout:
                print(line.strip())  # Print to terminal
                output_lines.append(line)

            process.wait(timeout=1800)

            # Parse results from output file
            output_file = f"eval_results_{os.path.basename(model_path)}_gsm8k/results.json"
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    results = json.load(f)
                return results.get('results', {}).get('gsm8k', {}).get('acc', 0.0)

        except subprocess.TimeoutExpired:
            logging.error("GSM8K evaluation timed out")
            if process:
                process.kill()
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
            "--batch_size", "auto:3",
            "--num_fewshot", "0",
            "--verbosity", "DEBUG"
        ]

        try:
            logging.info(f"Running command: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            output_lines = []
            for line in process.stdout:
                print(line.strip())
                output_lines.append(line)

            process.wait(timeout=1200)

            return self._parse_accuracy_from_output(''.join(output_lines), "hellaswag")
        except Exception as e:
            logging.error(f"HellaSwag evaluation failed: {e}")
            return 0.0
    
    def _evaluate_arc_easy(self, model_path: str) -> float:
        """Evaluate on ARC-Easy"""

        cmd = [
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path}",
            "--tasks", "arc_easy",
            "--batch_size", "auto:3",
            "--num_fewshot", "0",
            "--verbosity", "DEBUG"
        ]

        try:
            logging.info(f"Running command: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            output_lines = []
            for line in process.stdout:
                print(line.strip())
                output_lines.append(line)

            process.wait(timeout=1200)

            return self._parse_accuracy_from_output(''.join(output_lines), "arc_easy")
        except Exception as e:
            logging.error(f"ARC-Easy evaluation failed: {e}")
            return 0.0
    
    def _evaluate_mmlu_subset(self, model_path: str) -> float:
        """Evaluate on MMLU subset (elementary math)"""

        cmd = [
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path}",
            "--tasks", "mmlu_elementary_mathematics",
            "--batch_size", "auto:3",
            "--num_fewshot", "0",
            "--verbosity", "DEBUG"
        ]

        try:
            logging.info(f"Running command: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            output_lines = []
            for line in process.stdout:
                print(line.strip())
                output_lines.append(line)

            process.wait(timeout=1200)

            return self._parse_accuracy_from_output(''.join(output_lines), "mmlu_elementary_mathematics")
        except Exception as e:
            logging.error(f"MMLU evaluation failed: {e}")
            return 0.0

    def _evaluate_social_iqa(self, model_path: str) -> float:
        """Evaluate on SocialIQA for social reasoning"""

        cmd = [
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path}",
            "--tasks", "social_iqa",
            "--batch_size", "auto:3",
            "--num_fewshot", "0",
            "--verbosity", "DEBUG"
        ]

        try:
            logging.info(f"Running command: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            output_lines = []
            for line in process.stdout:
                print(line.strip())
                output_lines.append(line)

            process.wait(timeout=1200)

            return self._parse_accuracy_from_output(''.join(output_lines), "social_iqa")
        except Exception as e:
            logging.error(f"SocialIQA evaluation failed: {e}")
            return 0.0

    def _evaluate_truthfulqa(self, model_path: str) -> float:
        """Evaluate on TruthfulQA for truthfulness"""

        cmd = [
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path}",
            "--tasks", "truthfulqa_mc2",
            "--batch_size", "auto:3",
            "--num_fewshot", "0",
            "--verbosity", "DEBUG"
        ]

        try:
            logging.info(f"Running command: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            output_lines = []
            for line in process.stdout:
                print(line.strip())
                output_lines.append(line)

            process.wait(timeout=1800)

            return self._parse_accuracy_from_output(''.join(output_lines), "truthfulqa_mc2")
        except Exception as e:
            logging.error(f"TruthfulQA evaluation failed: {e}")
            return 0.0

    def _evaluate_winogrande(self, model_path: str) -> float:
        """Evaluate on Winogrande for commonsense reasoning"""

        cmd = [
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path}",
            "--tasks", "winogrande",
            "--batch_size", "auto:3",
            "--num_fewshot", "0",
            "--verbosity", "DEBUG"
        ]

        try:
            logging.info(f"Running command: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            output_lines = []
            for line in process.stdout:
                print(line.strip())
                output_lines.append(line)

            process.wait(timeout=1200)

            return self._parse_accuracy_from_output(''.join(output_lines), "winogrande")
        except Exception as e:
            logging.error(f"Winogrande evaluation failed: {e}")
            return 0.0

    def _evaluate_hhh_eval(self, model_path: str) -> float:
        """Evaluate on HHH (Helpful, Harmless, Honest) using Unitxt HH-RLHF"""

        # Use Unitxt's HH-RLHF evaluation through lm_eval
        cmd = [
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path}",
            "--tasks", "unitxt[card=cards.hh_rlhf,template=templates.classification.multi_class.relation.default,format=formats.chatapi]",
            "--batch_size", "auto:3",
            "--num_fewshot", "0",
            "--verbosity", "DEBUG"
        ]

        try:
            logging.info(f"Running command: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            output_lines = []
            for line in process.stdout:
                print(line.strip())
                output_lines.append(line)

            process.wait(timeout=2400)

            return self._parse_accuracy_from_output(''.join(output_lines), "unitxt")
        except Exception as e:
            logging.error(f"HH-RLHF evaluation failed: {e}")
            logging.warning("Falling back to ethics proxy evaluation")

            # Fallback to ethics evaluation
            fallback_cmd = [
                "python", "-m", "lm_eval",
                "--model", "hf",
                "--model_args", f"pretrained={model_path}",
                "--tasks", "hendrycksTest-moral_scenarios",
                "--batch_size", "auto:3",
                "--num_fewshot", "0",
                "--verbosity", "DEBUG"
            ]

            try:
                logging.info(f"Running fallback command: {' '.join(fallback_cmd)}")

                fallback_process = subprocess.Popen(
                    fallback_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )

                fallback_output_lines = []
                for line in fallback_process.stdout:
                    print(line.strip())
                    fallback_output_lines.append(line)

                fallback_process.wait(timeout=1200)

                return self._parse_accuracy_from_output(''.join(fallback_output_lines), "hendrycksTest-moral_scenarios")
            except Exception as fallback_e:
                logging.error(f"Fallback evaluation failed: {fallback_e}")
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