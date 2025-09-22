# test_baseline_evaluation.py
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluator import ModelEvaluator

def test_baseline_evaluation():
    """Test baseline model evaluation on benchmarks"""

    logging.basicConfig(level=logging.INFO)
    logging.info("🧪 Starting baseline model evaluation...")

    # Use the original base model
    baseline_model = "unsloth/llama-2-7b-bnb-4bit"

    # Path to your trained model for comparison
    trained_model_path = "./outputs/pipeline_test_test_non_reasoning/final_model"

    # Setup evaluator
    evaluation_config = {}
    evaluator = ModelEvaluator(evaluation_config)

    # Test with a simple benchmark first
    benchmarks = ["gsm8k"]

    # Evaluate baseline model
    logging.info(f"📊 Evaluating baseline model: {baseline_model}")
    try:
        baseline_results = evaluator.evaluate_model(
            model_path=baseline_model,
            model_name="baseline_llama2_7b",
            benchmarks=benchmarks
        )
        logging.info(f"📊 Baseline results: {baseline_results}")

    except Exception as e:
        logging.error(f"❌ Baseline evaluation failed: {e}")
        return

    # Evaluate trained model
    if os.path.exists(trained_model_path):
        logging.info(f"📊 Evaluating trained model: {trained_model_path}")
        try:
            trained_results = evaluator.evaluate_model(
                model_path=trained_model_path,
                model_name="trained_non_reasoning",
                benchmarks=benchmarks
            )
            logging.info(f"📊 Trained results: {trained_results}")

        except Exception as e:
            logging.error(f"❌ Trained model evaluation failed: {e}")

    # Generate comparison report
    if len(evaluator.results) >= 2:
        report = evaluator.generate_comparison_report()
        print("\n" + "="*50)
        print(report)
        print("="*50)

        # Save comprehensive results
        evaluator.save_results("baseline_comparison_results.json")

        # Save the report to a file
        with open("comparison_report.md", "w") as f:
            f.write(report)

        logging.info("📊 Comparison report saved to: comparison_report.md")

if __name__ == "__main__":
    test_baseline_evaluation()