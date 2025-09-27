#!/usr/bin/env python3
"""
CLI tool to upload training artifacts to S3
Usage:
  python upload_to_s3.py --experiment-name my_experiment --output-dir ./outputs
  python upload_to_s3.py --list-experiments
  python upload_to_s3.py --upload-best-model ./outputs/experiment/merged_model --model-name qwen-3b-diary-v1 --score 0.45 --benchmark gsm8k
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from s3_uploader import S3Uploader
from configs.config import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_s3_config():
    """Load S3 configuration"""
    try:
        config = load_config("configs/s3_config.yaml")
        return config.s3
    except Exception as e:
        logging.error(f"Failed to load S3 config: {e}")
        # Default config
        return type('Config', (), {
            'bucket_name': 'diary-training-artifacts',
            'aws_profile': None
        })()


def upload_experiment(args):
    """Upload complete experiment artifacts"""
    s3_config = load_s3_config()
    uploader = S3Uploader(s3_config.bucket_name, s3_config.aws_profile)

    # Load experiment configs and results
    configs = {}
    results = {}

    # Try to load configs
    config_dir = Path(args.output_dir).parent / "configs"
    if config_dir.exists():
        for config_file in config_dir.glob("*.yaml"):
            try:
                configs[config_file.stem] = load_config(str(config_file))
            except Exception as e:
                logging.warning(f"Could not load config {config_file}: {e}")

    # Try to load results
    results_file = Path(args.output_dir) / "experiment_results.json"
    if results_file.exists():
        try:
            with open(results_file) as f:
                results = json.load(f)
        except Exception as e:
            logging.warning(f"Could not load results: {e}")

    # Upload experiment
    s3_url = uploader.upload_experiment_artifacts(
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        configs=configs,
        results=results,
        training_time=args.training_time or 0
    )

    if s3_url:
        print(f"\n🎉 Upload successful!")
        print(f"📍 S3 URL: {s3_url}")
        return True
    else:
        print("\n❌ Upload failed!")
        return False


def upload_best_model(args):
    """Upload a model to best_models directory"""
    s3_config = load_s3_config()
    uploader = S3Uploader(s3_config.bucket_name, s3_config.aws_profile)

    success = uploader.upload_best_model(
        model_path=args.model_path,
        model_name=args.model_name,
        performance_score=args.score,
        benchmark=args.benchmark
    )

    if success:
        print(f"\n🏆 Best model uploaded: {args.model_name}")
        print(f"📊 Score: {args.score} on {args.benchmark}")
        return True
    else:
        print(f"\n❌ Failed to upload best model!")
        return False


def list_experiments(args):
    """List all experiments in S3"""
    s3_config = load_s3_config()
    uploader = S3Uploader(s3_config.bucket_name, s3_config.aws_profile)

    experiments = uploader.list_experiments()

    if not experiments:
        print("No experiments found in S3")
        return

    print(f"\n📋 Found {len(experiments)} experiments:")
    print("-" * 80)

    for exp in experiments:
        exp_info = exp.get('experiment_info', {})
        name = exp_info.get('name', 'Unknown')
        timestamp = exp_info.get('timestamp', 'Unknown')
        training_time = exp_info.get('training_time_hours', 0)

        print(f"🔬 {name}")
        print(f"   📅 {timestamp}")
        print(f"   ⏱️  Training time: {training_time}h")

        # Show best scores if available
        best_scores = exp.get('best_scores', {})
        if best_scores:
            print(f"   📊 Best scores:")
            for model_type, score in best_scores.items():
                print(f"      {model_type}: {score:.3f}")

        print()


def main():
    parser = argparse.ArgumentParser(description="Upload training artifacts to S3")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Upload experiment command
    upload_parser = subparsers.add_parser('upload-experiment', help='Upload experiment artifacts')
    upload_parser.add_argument('--experiment-name', required=True, help='Name of the experiment')
    upload_parser.add_argument('--output-dir', required=True, help='Output directory with models')
    upload_parser.add_argument('--training-time', type=float, help='Training time in seconds')

    # Upload best model command
    best_parser = subparsers.add_parser('upload-best-model', help='Upload best model')
    best_parser.add_argument('--model-path', required=True, help='Path to model directory')
    best_parser.add_argument('--model-name', required=True, help='Name for the model')
    best_parser.add_argument('--score', type=float, required=True, help='Performance score')
    best_parser.add_argument('--benchmark', required=True, help='Benchmark name')

    # List experiments command
    list_parser = subparsers.add_parser('list-experiments', help='List experiments in S3')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == 'upload-experiment':
            upload_experiment(args)
        elif args.command == 'upload-best-model':
            upload_best_model(args)
        elif args.command == 'list-experiments':
            list_experiments(args)

    except KeyboardInterrupt:
        print("\n⏹️  Upload cancelled by user")
    except Exception as e:
        logging.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()