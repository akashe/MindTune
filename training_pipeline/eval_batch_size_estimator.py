"""
Intelligent Batch Size Calculator for LM Evaluation
Automatically determines optimal batch sizes based on GPU memory and model size
"""

import torch
import logging
from typing import Dict, Optional
from transformers import AutoConfig


class BatchSizeCalculator:
    """
    Calculate optimal batch sizes for different evaluation tasks based on:
    1. Empirically discovered ratios (from Qwen 2.5 3B on A10 48GB)
    2. Current GPU memory availability
    3. Model size
    """

    # Reference configuration: Qwen 2.5 3B on A10 (48GB VRAM)
    REFERENCE_CONFIG = {
        'model_size_gb': 3.0,  # Qwen 2.5 3B in GB (params)
        'gpu_memory_gb': 48.0,  # A10 GPU VRAM
        'base_model': 'Qwen/Qwen2.5-3B'
    }

    # Empirically discovered batch sizes for reference config
    # These serve as the baseline ratios
    REFERENCE_BATCH_SIZES = {
        'gsm8k': 300,
        'hellaswag': 250,
        'arc_easy': 250,
        'mmlu_elementary_mathematics': 250,
        'mmlu': 250,  # Generic MMLU tasks
        'social_iqa': 100,
        'truthfulqa_mc2': 150,
        'truthfulqa': 150,
        'winogrande': 250,
        'ethics_cm': 25,
        'hendrycks_ethics': 25,
        'unitxt': 100,  # For HH-RLHF and similar
        'bbh': 100,  # Big Bench Hard (estimate based on complexity)
        'commonsense_qa': 200,  # Estimate based on similarity to arc_easy
    }

    # Task complexity factors (relative memory usage per sample)
    # Higher = more memory per sample (needs smaller batch)
    TASK_COMPLEXITY = {
        'ethics_cm': 12.0,          # Very long prompts (12x baseline)
        'hendrycks_ethics': 12.0,   # Very long prompts
        'unitxt': 3.0,              # Complex formatted prompts
        'bbh': 3.0,                 # Multi-step reasoning
        'social_iqa': 3.0,          # Long context
        'truthfulqa_mc2': 2.0,      # Multiple choice options
        'truthfulqa': 2.0,
        'commonsense_qa': 1.5,      # Medium length
        'gsm8k': 1.0,               # Baseline complexity
        'hellaswag': 1.2,
        'arc_easy': 1.2,
        'mmlu': 1.2,
        'mmlu_elementary_mathematics': 1.2,
        'winogrande': 1.2,
    }

    def __init__(self, model_path: str = None, gpu_id: int = 0):
        """
        Initialize batch size calculator

        Args:
            model_path: Path to model (for size estimation)
            gpu_id: GPU device ID to check memory
        """
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.available_memory_gb = self._get_available_gpu_memory()
        self.model_size_gb = self._estimate_model_size()

        logging.info(f"💾 GPU Memory: {self.available_memory_gb:.1f} GB available")
        logging.info(f"🤖 Model Size: {self.model_size_gb:.1f} GB (estimated)")

    def _get_available_gpu_memory(self) -> float:
        """Get available GPU memory in GB"""
        try:
            if torch.cuda.is_available():
                # Get total GPU memory
                total_memory = torch.cuda.get_device_properties(self.gpu_id).total_memory
                total_gb = total_memory / (1024**3)

                # Reserve 2GB for safety margin
                available_gb = total_gb - 2.0

                return max(available_gb, 1.0)  # At least 1GB
            else:
                logging.warning("⚠️  CUDA not available, using default")
                return 8.0  # Conservative default for CPU/MPS
        except Exception as e:
            logging.warning(f"⚠️  Could not detect GPU memory: {e}")
            return 8.0

    def _estimate_model_size(self) -> float:
        """Estimate model size in GB from config or path"""
        try:
            if self.model_path:
                # Try to load config
                config = AutoConfig.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )

                # Estimate size from hidden_size and num_layers
                hidden_size = getattr(config, 'hidden_size', 4096)
                num_layers = getattr(config, 'num_hidden_layers', 32)
                vocab_size = getattr(config, 'vocab_size', 50000)

                # Rough estimation: params ≈ hidden_size^2 * num_layers * 12 + vocab_size * hidden_size
                params_estimate = (hidden_size ** 2 * num_layers * 12 + vocab_size * hidden_size)

                # Convert to GB (assuming fp16, so 2 bytes per param)
                size_gb = (params_estimate * 2) / (1024**3)

                return size_gb

        except Exception as e:
            logging.warning(f"⚠️  Could not estimate model size: {e}")

        # Fallback: use reference model size
        return self.REFERENCE_CONFIG['model_size_gb']

    def _calculate_scaling_factor(self) -> float:
        """
        Calculate scaling factor based on available memory and model size

        Returns:
            Multiplier to apply to reference batch sizes
        """

        # Memory scaling factor
        # If we have 2x memory, we can use ~1.8x batch size (not exactly linear due to overhead)
        memory_ratio = self.available_memory_gb / self.REFERENCE_CONFIG['gpu_memory_gb']
        memory_factor = memory_ratio ** 0.9  # Sublinear scaling

        # Model size scaling factor
        # Larger models need smaller batches
        model_ratio = self.model_size_gb / self.REFERENCE_CONFIG['model_size_gb']
        model_factor = 1.0 / (model_ratio ** 0.85)  # Inverse relationship

        # Combined scaling
        scaling = memory_factor * model_factor

        logging.info(f"📊 Batch size scaling factor: {scaling:.2f}x")
        logging.info(f"   Memory factor: {memory_factor:.2f}x")
        logging.info(f"   Model factor: {model_factor:.2f}x")

        return scaling

    def get_batch_size(self, task: str, scaling_factor: Optional[float] = None) -> int:
        """
        Get optimal batch size for a specific task

        Args:
            task: Task name (e.g., 'gsm8k', 'social_iqa')
            scaling_factor: Optional override for scaling factor

        Returns:
            Recommended batch size
        """

        if scaling_factor is None:
            scaling_factor = self._calculate_scaling_factor()

        # Match task name to reference batch sizes
        # Handle task variants (e.g., mmlu_philosophy -> mmlu)
        base_batch_size = None

        # Direct match
        if task in self.REFERENCE_BATCH_SIZES:
            base_batch_size = self.REFERENCE_BATCH_SIZES[task]
        else:
            # Try prefix matching (e.g., mmlu_* -> mmlu)
            for key in self.REFERENCE_BATCH_SIZES:
                if task.startswith(key):
                    base_batch_size = self.REFERENCE_BATCH_SIZES[key]
                    break

        if base_batch_size is None:
            logging.warning(f"⚠️  Unknown task '{task}', using conservative default")
            base_batch_size = 100

        # Apply scaling factor
        scaled_batch_size = int(base_batch_size * scaling_factor)

        # Safety bounds
        batch_size = max(1, min(scaled_batch_size, 500))  # Between 1 and 500

        logging.debug(f"📦 {task}: base={base_batch_size}, scaled={scaled_batch_size}, final={batch_size}")

        return batch_size

    def get_all_batch_sizes(self, tasks: list) -> Dict[str, int]:
        """
        Get batch sizes for multiple tasks

        Args:
            tasks: List of task names

        Returns:
            Dictionary mapping task -> batch_size
        """

        scaling_factor = self._calculate_scaling_factor()

        batch_sizes = {}
        for task in tasks:
            batch_sizes[task] = self.get_batch_size(task, scaling_factor)

        return batch_sizes

    def print_recommendations(self, tasks: list):
        """Print batch size recommendations for given tasks"""

        print(f"\n{'='*60}")
        print("BATCH SIZE RECOMMENDATIONS")
        print(f"{'='*60}")
        print(f"GPU Memory: {self.available_memory_gb:.1f} GB")
        print(f"Model Size: {self.model_size_gb:.1f} GB")
        print(f"Scaling Factor: {self._calculate_scaling_factor():.2f}x")
        print(f"\n{'Task':<40} {'Batch Size':>10}")
        print(f"{'-'*40} {'-'*10}")

        batch_sizes = self.get_all_batch_sizes(tasks)

        for task in sorted(batch_sizes.keys()):
            batch_size = batch_sizes[task]
            print(f"{task:<40} {batch_size:>10}")

        print(f"{'='*60}\n")

        return batch_sizes


def main():
    """Test the batch size calculator"""

    # Example usage
    calculator = BatchSizeCalculator(model_path="Qwen/Qwen2.5-3B")

    test_tasks = [
        'gsm8k',
        'hellaswag',
        'arc_easy',
        'social_iqa',
        'hendrycks_ethics',
        'truthfulqa_mc2',
        'winogrande',
        'bbh',
        'commonsense_qa',
        'mmlu_philosophy',
        'mmlu_psychology',
    ]

    calculator.print_recommendations(test_tasks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
