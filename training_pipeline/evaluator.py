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

        # Task-specific batch sizes (empirically determined for Qwen 2.5 3B on A40 48GB)
        # Modern benchmarks (2024-2025)
        self.batch_sizes = {
            # Modern challenging benchmarks
            'arc_challenge': 200,        # Multiple choice science reasoning
            'gpqa_diamond_zeroshot': 50, # PhD-level STEM (complex, long context)
            'musr': 100,                 # Multi-step soft reasoning
            'minerva_math': 32,          # Generative math (like GSM8K)

            # Legacy benchmarks
            'gsm8k': 300,
            'hellaswag': 250,
            'arc_easy': 250,
            'mmlu_elementary_mathematics': 250,
            'mmlu_philosophy': 250,
            'mmlu_moral_scenarios': 250,
            'mmlu_moral_disputes': 250,
            'mmlu_high_school_psychology': 250,
            'mmlu_formal_logic': 250,
            'mmlu_logical_fallacies': 250,
            'mmlu_machine_learning': 250,
            'social_iqa': 100,
            'truthfulqa_mc2': 150,
            'truthfulqa': 150,
            'winogrande': 250,
            'ethics_cm': 25,
            'ethics_deontology': 25,
            'ethics_justice': 25,
            'ethics_utilitarianism': 25,
            'ethics_virtue': 25,
            'hendrycks_ethics': 25,
            'bbh': 16,              # Generative: complex reasoning with long outputs
            'bbh_cot_fewshot': 16,  # Generative: few-shot examples + long reasoning
            'commonsense_qa': 200,
            'piqa': 200,
            'drop': 32,             # Generative: reading comp with calculations
            'strategyqa': 150,
            'bigbench_strategyqa_multiple_choice': 150,
            'eq_bench': 100,
            'moral_stories': 150,
        }

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

                # Get batch size (default to 100 if not found)
                batch_size = self.batch_sizes.get(benchmark, 100)
                logging.info(f"Got batch size: {batch_size} for {benchmark}")

                # Route to appropriate evaluation method
                if benchmark == "gsm8k":
                    # Add system instruction to enforce format
                    system_inst = "Always end your answer with '####' followed by the final numerical answer."
                    score = self._evaluate_task(model_path, eval_dir, model_name, "gsm8k", batch_size,
                                               metric="exact_match,strict-match", system_instruction=system_inst)
                elif benchmark == "hellaswag":
                    score = self._evaluate_task(model_path, eval_dir, model_name, "hellaswag", batch_size, metric="acc,none")
                elif benchmark == "arc_easy":
                    score = self._evaluate_task(model_path, eval_dir, model_name, "arc_easy", batch_size, metric="acc,none")
                elif benchmark == "social_iqa":
                    score = self._evaluate_task(model_path, eval_dir, model_name, "social_iqa", batch_size, metric="acc,none", trust_remote_code=True)
                elif benchmark == "truthfulqa" or benchmark == "truthfulqa_mc2":
                    score = self._evaluate_task(model_path, eval_dir, model_name, "truthfulqa_mc2", batch_size, metric="acc,none")
                elif benchmark == "winogrande":
                    score = self._evaluate_task(model_path, eval_dir, model_name, "winogrande", batch_size, metric="acc,none")

                # Ethics subtasks
                elif benchmark.startswith("ethics_"):
                    score = self._evaluate_task(model_path, eval_dir, model_name, benchmark, batch_size, metric="acc,none", trust_remote_code=True)
                elif benchmark == "hendrycks_ethics":
                    score = self._evaluate_ethics_all(model_path, eval_dir, model_name)

                # Modern challenging reasoning tasks (2024-2025)
                elif benchmark == "arc_challenge":
                    score = self._evaluate_task(model_path, eval_dir, model_name, "arc_challenge", batch_size, metric="acc_norm,none")
                elif benchmark == "gpqa_diamond_zeroshot":
                    score = self._evaluate_task(model_path, eval_dir, model_name, "gpqa_diamond_zeroshot", batch_size, metric="acc,none")
                elif benchmark == "musr":
                    score = self._evaluate_task(model_path, eval_dir, model_name, "musr", batch_size, metric="acc,none")
                elif benchmark == "minerva_math":
                    score = self._evaluate_task(model_path, eval_dir, model_name, "minerva_math", batch_size, metric="exact_match,none")

                # Traditional reasoning tasks
                elif benchmark == "commonsense_qa":
                    score = self._evaluate_task(model_path, eval_dir, model_name, "commonsense_qa", batch_size, metric="acc,none")
                elif benchmark == "piqa":
                    score = self._evaluate_task(model_path, eval_dir, model_name, "piqa", batch_size, metric="acc,none")
                elif benchmark == "strategyqa":
                    score = self._evaluate_task(model_path, eval_dir, model_name, "bigbench_strategyqa_multiple_choice", batch_size, metric="acc,none")

                # Legacy/deprecated reasoning tasks (saturated)
                elif benchmark == "bbh" or benchmark == "bbh_cot_fewshot":
                    score = self._evaluate_task(model_path, eval_dir, model_name, "bbh_cot_fewshot", batch_size, metric="acc,none", num_fewshot=3)
                elif benchmark == "drop":
                    score = self._evaluate_task(model_path, eval_dir, model_name, "drop", batch_size, metric="f1,none")

                # Emotional/Social tasks
                elif benchmark == "eq_bench":
                    score = self._evaluate_task(model_path, eval_dir, model_name, "eq_bench", batch_size, metric="acc,none")
                elif benchmark == "moral_stories":
                    score = self._evaluate_task(model_path, eval_dir, model_name, "moral_stories", batch_size, metric="acc,none")

                # MMLU subjects
                elif benchmark.startswith("mmlu_"):
                    score = self._evaluate_task(model_path, eval_dir, model_name, benchmark, batch_size, metric="acc,none")
                elif benchmark == "mmlu_subset":  # Legacy support
                    score = self._evaluate_task(model_path, eval_dir, model_name, "mmlu_elementary_mathematics", batch_size, metric="acc,none")

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

    def _evaluate_task(self, model_path: str, eval_dir: str, model_name: str, task_name: str,
                      batch_size: int, metric: str = "acc,none", num_fewshot: int = 0,
                      timeout: int = 1800, trust_remote_code: bool = False,
                      system_instruction: str = None) -> float:
        """Generic task evaluation using lm_eval

        Args:
            model_path: Path to model
            eval_dir: Directory for eval results
            model_name: Name of model
            task_name: lm_eval task name
            batch_size: Batch size for evaluation
            metric: Metric to extract from results
            num_fewshot: Number of few-shot examples
            timeout: Timeout in seconds
            trust_remote_code: Whether to trust remote code
            system_instruction: Optional system instruction to prepend to prompts

        Returns:
            Score for the specified metric
        """

        cmd = [
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", self._get_model_args(model_path),
            "--tasks", task_name,
            "--batch_size", str(batch_size),
            "--num_fewshot", str(num_fewshot),
            "--log_samples",
            "--output_path", os.path.join(eval_dir, f"eval_results_{model_name}_{task_name}"),
            "--verbosity", "DEBUG"
        ]

        # Only add gen_kwargs for generative tasks with task-specific limits
        generative_configs = {
            # Modern benchmarks
            "minerva_math": "null",    # Math reasoning: step-by-step solutions
            "gpqa": "null",            # STEM Q&A: moderate explanations

            # Legacy benchmarks
            "gsm8k": "null",           # Math: moderate length for step-by-step + answer
            "drop": 256,            # Reading comp: short answers with calculations
            "bbh_cot_fewshot": 512, # Complex reasoning: longer chains
            "bbh_cot_zeroshot": 512,
            "bigbench_strategyqa_multiple_choice": "null"  # Yes/No with brief reasoning
        }

        for task_pattern, max_tokens in generative_configs.items():
            if task_pattern in task_name:
                cmd.extend(["--gen_kwargs", f'{{"max_new_tokens":{max_tokens}}}'])
                break

        if trust_remote_code:
            cmd.append("--trust_remote_code")

        if system_instruction:
            cmd.extend(["--system_instruction", system_instruction])

        try:
            logging.info(f"Running: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Stream output
            for line in process.stdout:
                print(line.strip())

            process.wait(timeout=timeout)

            # Parse results
            output_dir = os.path.join(eval_dir, f"eval_results_{model_name}_{task_name}")
            results_file = self._find_results_file(output_dir, task_name)

            if results_file:
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)

                    # Extract metric
                    task_results = results.get('results', {}).get(task_name, {})
                    score = task_results.get(metric, 0.0)

                    return score

                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logging.warning(f"Failed to parse {task_name} results: {e}")
                    return 0.0
            else:
                logging.error(f"Results file not found for {task_name}")
                return 0.0

        except subprocess.TimeoutExpired:
            logging.error(f"{task_name} evaluation timed out")
            if process:
                process.kill()
        except Exception as e:
            logging.error(f"{task_name} evaluation failed: {e}")

        return 0.0

    def _evaluate_ethics_all(self, model_path: str, eval_dir: str, model_name: str) -> float:
        """Evaluate all 5 ethics subtasks and return average

        Returns average score across all ethics dimensions
        """

        ethics_tasks = [
            'ethics_cm',
            'ethics_deontology',
            'ethics_justice',
            'ethics_utilitarianism',
            'ethics_virtue'
        ]

        scores = []
        for task in ethics_tasks:
            logging.info(f"📊 Evaluating {task}...")
            batch_size = self.batch_sizes.get(task, 25)
            score = self._evaluate_task(model_path, eval_dir, model_name, task, batch_size,
                                       metric="acc,none", trust_remote_code=True)
            scores.append(score)
            logging.info(f"  ✅ {task}: {score:.3f}")

        avg_score = sum(scores) / len(scores) if scores else 0.0
        logging.info(f"📊 Ethics Average: {avg_score:.3f}")

        return avg_score
    
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
            "--gen_kwargs", '{"max_new_tokens":null}'
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
            "--gen_kwargs", '{"max_new_tokens":null}'
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
            "--gen_kwargs", '{"max_new_tokens":null}'
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
            "--gen_kwargs", '{"max_new_tokens":null}'
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
            "--gen_kwargs", '{"max_new_tokens":null}'
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
            "--gen_kwargs", '{"max_new_tokens":null}'
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
            "--gen_kwargs", '{"max_new_tokens":null}'
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
            "--tasks", "ethics_cm",
            "--batch_size", "auto",
            "--num_fewshot", "0",
            "--log_samples",
            "--output_path", os.path.join(eval_dir, f"eval_results_{model_name}_ethics"),
            "--verbosity", "DEBUG",
            "--trust_remote_code",
            "--gen_kwargs", '{"max_new_tokens":null}'
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
                    return results.get('results', {}).get('ethics_cm', {}).get('acc,none', 0.0)
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
            "--gen_kwargs", '{"max_new_tokens":null}'
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
            #     "--gen_kwargs", '{"max_new_tokens":null}'
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