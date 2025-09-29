# evaluator.py
import subprocess
import json
import os
import logging
from typing import Dict, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelEvaluator:
    def __init__(self, evaluation_config: Dict, output_dir: str = None):
        self.config = evaluation_config
        self.results = {}
        self.output_dir = output_dir

    def _get_model_args(self, model_path: str) -> str:
        """Get appropriate model arguments based on model type"""
        if "/outputs/" in model_path or "merged_model" in model_path:
            # Check if merged model exists (faster evaluation)
            if os.path.exists(model_path):
                logging.info(f"🚀 Using merged model for faster evaluation: {model_path}")
                return f"pretrained={model_path},device_map=auto,trust_remote_code=True"

            # Fallback to PEFT adapter loading
            logging.info(f"Using LoRA adapters (slower): {model_path}")
            return f"pretrained=unsloth/llama-2-7b-bnb-4bit,peft={model_path},device_map=auto,trust_remote_code=True"
        else:
            # Base model - may need quantization for memory efficiency
            return f"pretrained={model_path},device_map=auto"

    def _find_results_file(self, output_dir: str, expected_task: str) -> str:
        """Find the actual results file in lm_eval output directory"""
        if not os.path.exists(output_dir):
            return None

        import glob
        # Look for results files in any subdirectory, including hidden directories
        pattern = os.path.join(output_dir, "**/results_*.json")
        results_files = glob.glob(pattern, recursive=True, include_hidden=True)

        if results_files:
            # Return the most recent results file
            latest_file = max(results_files, key=os.path.getctime)
            logging.info(f"🔍 Found results file: {latest_file}")
            return latest_file

        # Fallback: look for results.json
        fallback = os.path.join(output_dir, "results.json")
        return fallback if os.path.exists(fallback) else None
    
    def evaluate_model(self, model_path: str, model_name: str, benchmarks: List[str], experiment_dir: str = None) -> Dict:
        """Evaluate model on specified benchmarks"""

        # Set up evaluation directory
        if experiment_dir:
            eval_dir = os.path.join(experiment_dir, "evals")
            os.makedirs(eval_dir, exist_ok=True)
            logging.info(f"📁 Creating eval results in: {eval_dir}")
        else:
            eval_dir = "."
        
        logging.info(f"📊 Evaluating {model_name} on benchmarks: {benchmarks}")
        
        model_results = {}
        
        for benchmark in benchmarks:
            try:
                logging.info(f"🔄 Running {benchmark} evaluation...")
                
                if benchmark == "gsm8k":
                    score = self._evaluate_gsm8k(model_path, eval_dir, model_name)
                elif benchmark == "hellaswag":
                    score = self._evaluate_hellaswag(model_path, eval_dir, model_name)
                elif benchmark == "arc_easy":
                    score = self._evaluate_arc_easy(model_path, eval_dir, model_name)
                elif benchmark == "mmlu_subset":
                    score = self._evaluate_mmlu_subset(model_path, eval_dir, model_name)
                elif benchmark == "social_iqa":
                    score = self._evaluate_social_iqa(model_path, eval_dir, model_name)
                elif benchmark == "truthfulqa":
                    score = self._evaluate_truthfulqa(model_path, eval_dir, model_name)
                elif benchmark == "winogrande":
                    score = self._evaluate_winogrande(model_path, eval_dir, model_name)
                elif benchmark == "hendrycks_ethics":
                    score = self._evaluate_hendrycks_ethics(model_path, eval_dir, model_name)
                elif benchmark == "hhh_eval":
                    logging.warning(f"⚠️  Skipping {benchmark}: task not available in lm_eval")
                    continue
                else:
                    logging.warning(f"⚠️  Unknown benchmark: {benchmark}")
                    continue
                
                model_results[benchmark] = score
                logging.info(f"✅ {benchmark}: {score:.3f}")
                
            except Exception as e:
                logging.error(f"❌ Failed to evaluate {benchmark}: {str(e)}")
                model_results[benchmark] = 0.0
        
        self.results[model_name] = model_results
        return model_results
    
    def _evaluate_gsm8k(self, model_path: str, eval_dir: str = ".", model_name: str = "model") -> float:
        """Evaluate on GSM8K math reasoning"""

        cmd = [
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", self._get_model_args(model_path),
            "--tasks", "gsm8k",
            "--batch_size", "auto",
            "--num_fewshot", "0",
            "--log_samples",
            "--output_path", os.path.join(eval_dir, f"eval_results_{model_name}_gsm8k"),
            "--verbosity", "DEBUG",
            "--limit", "5"
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

            # Parse results from output file using robust file finder
            output_dir = os.path.join(eval_dir, f"eval_results_{model_name}_gsm8k")
            results_file = self._find_results_file(output_dir, "gsm8k")
            if results_file:
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    return results.get('results', {}).get('gsm8k', {}).get('exact_match,strict-match', 0.0)
                except (json.JSONDecodeError, KeyError, TypeError) as parse_error:
                    logging.warning(f"Failed to parse GSM8K results: {str(parse_error)}")
                    return 0.0
            else:
                logging.error(f"GSM8K results file not found in: {output_dir}")
                return 0.0

        except subprocess.TimeoutExpired:
            logging.error("GSM8K evaluation timed out")
            if process:
                process.kill()
        except Exception as e:
            logging.error(f"GSM8K evaluation failed: {str(e)}")

        return 0.0
    
    def _evaluate_hellaswag(self, model_path: str, eval_dir: str = ".", model_name: str = "model") -> float:
        """Evaluate on HellaSwag common sense reasoning"""

        cmd = [
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", self._get_model_args(model_path),
            "--tasks", "hellaswag",
            "--batch_size", "auto",
            "--num_fewshot", "0",
            "--log_samples",
            "--output_path", os.path.join(eval_dir, f"eval_results_{model_name}_hellaswag"),
            "--verbosity", "DEBUG",
            "--limit", "5"
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

            # Parse results from output file using robust file finder
            output_dir = os.path.join(eval_dir, f"eval_results_{model_name}_hellaswag")
            results_file = self._find_results_file(output_dir, "hellaswag")
            if results_file:
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    return results.get('results', {}).get('hellaswag', {}).get('acc,none', 0.0)
                except (json.JSONDecodeError, KeyError, TypeError) as parse_error:
                    logging.warning(f"Failed to parse HellaSwag results: {str(parse_error)}")
                    return 0.0
            else:
                logging.error(f"HellaSwag results file not found in: {output_dir}")
                return 0.0
        except Exception as e:
            logging.error(f"HellaSwag evaluation failed: {str(e)}")
            return 0.0
    
    def _evaluate_arc_easy(self, model_path: str, eval_dir: str = ".", model_name: str = "model") -> float:
        """Evaluate on ARC-Easy"""

        cmd = [
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", self._get_model_args(model_path),
            "--tasks", "arc_easy",
            "--batch_size", "auto",
            "--num_fewshot", "0",
            "--log_samples",
            "--output_path", os.path.join(eval_dir, f"eval_results_{model_name}_arc_easy"),
            "--verbosity", "DEBUG",
            "--limit", "5"
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

            # Parse results from output file using robust file finder
            output_dir = os.path.join(eval_dir, f"eval_results_{model_name}_arc_easy")
            results_file = self._find_results_file(output_dir, "arc_easy")
            if results_file:
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    return results.get('results', {}).get('arc_easy', {}).get('acc,none', 0.0)
                except (json.JSONDecodeError, KeyError, TypeError) as parse_error:
                    logging.warning(f"Failed to parse ARC-Easy results: {str(parse_error)}")
                    return 0.0
            else:
                logging.error(f"ARC-Easy results file not found in: {output_dir}")
                return 0.0
        except Exception as e:
            logging.error(f"ARC-Easy evaluation failed: {str(e)}")
            return 0.0
    
    def _evaluate_mmlu_subset(self, model_path: str, eval_dir: str = ".", model_name: str = "model") -> float:
        """Evaluate on MMLU subset (elementary math)"""

        cmd = [
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", self._get_model_args(model_path),
            "--tasks", "mmlu_elementary_mathematics",
            "--batch_size", "auto",
            "--num_fewshot", "0",
            "--log_samples",
            "--output_path", os.path.join(eval_dir, f"eval_results_{model_name}_mmlu"),
            "--verbosity", "DEBUG",
            "--limit", "5"
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

            # Parse results from output file using robust file finder
            output_dir = os.path.join(eval_dir, f"eval_results_{model_name}_mmlu")
            results_file = self._find_results_file(output_dir, "mmlu_elementary_mathematics")
            if results_file:
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    return results.get('results', {}).get('mmlu_elementary_mathematics', {}).get('acc,none', 0.0)
                except (json.JSONDecodeError, KeyError, TypeError) as parse_error:
                    logging.warning(f"Failed to parse MMLU results: {str(parse_error)}")
                    return 0.0
            else:
                logging.error(f"MMLU results file not found in: {output_dir}")
                return 0.0
        except Exception as e:
            logging.error(f"MMLU evaluation failed: {str(e)}")
            return 0.0

    def _evaluate_social_iqa(self, model_path: str, eval_dir: str = ".", model_name: str = "model") -> float:
        """Evaluate on SocialIQA for social reasoning"""

        cmd = [
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", self._get_model_args(model_path),
            "--tasks", "social_iqa",
            "--batch_size", "auto",
            "--num_fewshot", "0",
            "--log_samples",
            "--output_path", os.path.join(eval_dir, f"eval_results_{model_name}_social_iqa"),
            "--verbosity", "DEBUG",
            "--trust_remote_code",
            "--limit", "5"
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

            # Parse results from output file using robust file finder
            output_dir = os.path.join(eval_dir, f"eval_results_{model_name}_social_iqa")
            results_file = self._find_results_file(output_dir, "social_iqa")
            if results_file:
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    return results.get('results', {}).get('social_iqa', {}).get('acc,none', 0.0)
                except (json.JSONDecodeError, KeyError, TypeError) as parse_error:
                    logging.warning(f"Failed to parse SocialIQA results: {str(parse_error)}")
                    return 0.0
            else:
                logging.error(f"SocialIQA results file not found in: {output_dir}")
                return 0.0
        except Exception as e:
            logging.error(f"SocialIQA evaluation failed: {str(e)}")
            return 0.0

    def _evaluate_truthfulqa(self, model_path: str, eval_dir: str = ".", model_name: str = "model") -> float:
        """Evaluate on TruthfulQA for truthfulness"""

        cmd = [
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", self._get_model_args(model_path),
            "--tasks", "truthfulqa_mc2",
            "--batch_size", "auto",
            "--num_fewshot", "0",
            "--log_samples",
            "--output_path", os.path.join(eval_dir, f"eval_results_{model_name}_truthfulqa"),
            "--verbosity", "DEBUG",
            "--limit", "5"
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

            # Parse results from output file using robust file finder
            output_dir = os.path.join(eval_dir, f"eval_results_{model_name}_truthfulqa")
            results_file = self._find_results_file(output_dir, "truthfulqa_mc2")
            if results_file:
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    return results.get('results', {}).get('truthfulqa_mc2', {}).get('acc,none', 0.0)
                except (json.JSONDecodeError, KeyError, TypeError) as parse_error:
                    logging.warning(f"Failed to parse TruthfulQA results: {str(parse_error)}")
                    return 0.0
            else:
                logging.error(f"TruthfulQA results file not found in: {output_dir}")
                return 0.0
        except Exception as e:
            logging.error(f"TruthfulQA evaluation failed: {str(e)}")
            return 0.0

    def _evaluate_winogrande(self, model_path: str, eval_dir: str = ".", model_name: str = "model") -> float:
        """Evaluate on Winogrande for commonsense reasoning"""

        cmd = [
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", self._get_model_args(model_path),
            "--tasks", "winogrande",
            "--batch_size", "auto",
            "--num_fewshot", "0",
            "--log_samples",
            "--output_path", os.path.join(eval_dir, f"eval_results_{model_name}_winogrande"),
            "--verbosity", "DEBUG",
            "--limit", "5"
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

            # Parse results from output file using robust file finder
            output_dir = os.path.join(eval_dir, f"eval_results_{model_name}_winogrande")
            results_file = self._find_results_file(output_dir, "winogrande")
            if results_file:
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    return results.get('results', {}).get('winogrande', {}).get('acc,none', 0.0)
                except (json.JSONDecodeError, KeyError, TypeError) as parse_error:
                    logging.warning(f"Failed to parse Winogrande results: {str(parse_error)}")
                    return 0.0
            else:
                logging.error(f"Winogrande results file not found in: {output_dir}")
                return 0.0
        except Exception as e:
            logging.error(f"Winogrande evaluation failed: {str(e)}")
            return 0.0

    def _evaluate_hendrycks_ethics(self, model_path: str, eval_dir: str = ".", model_name: str = "model") -> float:
        """Evaluate on Hendrycks Ethics for moral reasoning"""

        cmd = [
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", self._get_model_args(model_path),
            "--tasks", "hendrycksTest-moral_scenarios",
            "--batch_size", "auto",
            "--num_fewshot", "0",
            "--log_samples",
            "--output_path", os.path.join(eval_dir, f"eval_results_{model_name}_ethics"),
            "--verbosity", "DEBUG",
            "--limit", "5"
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

            # Parse results from output file using robust file finder
            output_dir = os.path.join(eval_dir, f"eval_results_{model_name}_ethics")
            results_file = self._find_results_file(output_dir, "hendrycksTest-moral_scenarios")
            if results_file:
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    return results.get('results', {}).get('hendrycksTest-moral_scenarios', {}).get('acc,none', 0.0)
                except (json.JSONDecodeError, KeyError, TypeError) as parse_error:
                    logging.warning(f"Failed to parse Ethics results: {str(parse_error)}")
                    return 0.0
            else:
                logging.error(f"Ethics results file not found in: {output_dir}")
                return 0.0
        except Exception as e:
            logging.error(f"hendrycks Ethics evaluation failed: {str(e)}")
            return 0.0

    def _evaluate_hhh_eval(self, model_path: str) -> float:
        """Evaluate on HHH (Helpful, Harmless, Honest) using Unitxt HH-RLHF"""

        # Use Unitxt's HH-RLHF evaluation through lm_eval
        cmd = [
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", self._get_model_args(model_path),
            "--tasks", "unitxt[card=cards.hh_rlhf,template=templates.classification.multi_class.relation.default,format=formats.chatapi]",
            "--batch_size", "auto",
            "--num_fewshot", "0",
            "--log_samples",
            "--output_path", f"eval_results_{os.path.basename(model_path) if model_path else 'model'}_hhh",
            "--verbosity", "DEBUG",
            "--limit", "5"
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

            # Parse results from output file
            output_file = f"eval_results_{os.path.basename(model_path) if model_path else 'model'}_hhh/results.json"
            if os.path.exists(output_file):
                try:
                    with open(output_file, 'r') as f:
                        results = json.load(f)
                    return results.get('results', {}).get('unitxt', {}).get('acc,none', 0.0)
                except (json.JSONDecodeError, KeyError, TypeError) as parse_error:
                    logging.warning(f"Failed to parse HHH results: {str(parse_error)}")
                    return 0.0
            else:
                logging.error(f"HHH results file not found: {output_file}")
                return 0.0
        except Exception as e:
            logging.error(f"HH-RLHF evaluation failed: {str(e)}")
            logging.warning("Falling back to ethics proxy evaluation")

            # # Fallback to ethics evaluation
            # fallback_cmd = [
            #     "python", "-m", "lm_eval",
            #     "--model", "hf",
            #     "--model_args", self._get_model_args(model_path),
            #     "--tasks", "hendrycksTest-moral_scenarios",
            #     "--batch_size", "auto",
            #     "--num_fewshot", "0",
            #     "--log_samples",
            #     "--output_path", f"eval_results_{os.path.basename(model_path) if model_path else 'model'}_ethics",
            #     "--verbosity", "DEBUG",
            #     "--limit", "10"
            # ]

            # try:
            #     logging.info(f"Running fallback command: {' '.join(fallback_cmd)}")

            #     fallback_process = subprocess.Popen(
            #         fallback_cmd,
            #         stdout=subprocess.PIPE,
            #         stderr=subprocess.STDOUT,
            #         text=True,
            #         bufsize=1,
            #         universal_newlines=True
            #     )

            #     fallback_output_lines = []
            #     for line in fallback_process.stdout:
            #         print(line.strip())
            #         fallback_output_lines.append(line)

            #     fallback_process.wait(timeout=1200)

            #     # Parse results from fallback output file
            #     fallback_output_file = f"eval_results_{os.path.basename(model_path) if model_path else 'model'}_ethics/results.json"
            #     if os.path.exists(fallback_output_file):
            #         try:
            #             with open(fallback_output_file, 'r') as f:
            #                 results = json.load(f)
            #             return results.get('results', {}).get('hendrycksTest-moral_scenarios', {}).get('acc,none', 0.0)
            #         except (json.JSONDecodeError, KeyError, TypeError) as parse_error:
            #             logging.warning(f"Failed to parse Ethics results: {str(parse_error)}")
            #             return 0.0
            # except Exception as fallback_e:
            #     logging.error(f"Fallback evaluation failed: {str(fallback_e)}")
            #     return 0.0

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