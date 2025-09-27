import os
import json
import boto3
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import hashlib
from botocore.exceptions import ClientError, NoCredentialsError


class S3Uploader:
    """Upload training artifacts to S3 with organized structure"""

    def __init__(self, bucket_name: str, aws_profile: Optional[str] = None):
        self.bucket_name = bucket_name
        self.aws_profile = aws_profile

        # Initialize S3 client
        try:
            if aws_profile:
                session = boto3.Session(profile_name=aws_profile)
                self.s3_client = session.client('s3')
            else:
                self.s3_client = boto3.client('s3')

            # Test connection
            self.s3_client.head_bucket(Bucket=bucket_name)
            logging.info(f"✅ Connected to S3 bucket: {bucket_name}")

        except NoCredentialsError:
            logging.error("❌ AWS credentials not found. Configure with 'aws configure'")
            raise
        except ClientError as e:
            logging.error(f"❌ Failed to connect to S3 bucket {bucket_name}: {e}")
            raise

    def create_experiment_metadata(self, experiment_name: str, configs: Dict,
                                 results: Dict, training_time: float) -> Dict:
        """Create metadata summary for the experiment"""
        metadata = {
            "experiment_info": {
                "name": experiment_name,
                "timestamp": datetime.now().isoformat(),
                "training_time_hours": round(training_time / 3600, 2)
            },
            "model_configs": configs,
            "evaluation_results": results,
            "best_scores": {
                "non_reasoning": max(results.get("non_reasoning", {}).values()) if results.get("non_reasoning") else 0,
                "reasoning": max(results.get("reasoning", {}).values()) if results.get("reasoning") else 0
            },
            "artifacts": {
                "merged_models": True,
                "lora_adapters": True,
                "evaluation_logs": True,
                "training_logs": True
            }
        }
        return metadata

    def upload_file(self, local_path: str, s3_key: str,
                   check_exists: bool = True) -> bool:
        """Upload a single file to S3"""
        try:
            # Check if file already exists
            if check_exists:
                try:
                    self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
                    logging.info(f"⏭️  Skipping {s3_key} (already exists)")
                    return True
                except ClientError:
                    pass  # File doesn't exist, proceed with upload

            # Get file size for progress
            file_size = os.path.getsize(local_path)
            size_mb = file_size / (1024 * 1024)

            logging.info(f"📤 Uploading {os.path.basename(local_path)} ({size_mb:.1f}MB) to {s3_key}")

            # Upload with progress callback for large files
            if size_mb > 100:
                def progress_callback(bytes_transferred):
                    percentage = (bytes_transferred / file_size) * 100
                    print(f"\r   Progress: {percentage:.1f}%", end="", flush=True)

                self.s3_client.upload_file(
                    local_path, self.bucket_name, s3_key,
                    Callback=progress_callback
                )
                print()  # New line after progress
            else:
                self.s3_client.upload_file(local_path, self.bucket_name, s3_key)

            logging.info(f"✅ Uploaded: {s3_key}")
            return True

        except Exception as e:
            logging.error(f"❌ Failed to upload {local_path}: {e}")
            return False

    def upload_directory(self, local_dir: str, s3_prefix: str,
                        exclude_patterns: List[str] = None) -> bool:
        """Upload entire directory to S3"""
        exclude_patterns = exclude_patterns or []
        local_path = Path(local_dir)

        if not local_path.exists():
            logging.warning(f"⚠️  Directory doesn't exist: {local_dir}")
            return False

        success_count = 0
        total_count = 0

        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                # Check exclude patterns
                relative_path = file_path.relative_to(local_path)
                if any(pattern in str(relative_path) for pattern in exclude_patterns):
                    continue

                total_count += 1
                s3_key = f"{s3_prefix}/{relative_path}".replace("\\", "/")

                if self.upload_file(str(file_path), s3_key):
                    success_count += 1

        logging.info(f"📁 Directory upload: {success_count}/{total_count} files successful")
        return success_count == total_count

    def upload_experiment_artifacts(self, experiment_name: str,
                                  output_dir: str, configs: Dict,
                                  results: Dict, training_time: float) -> bool:
        """Upload complete experiment artifacts to S3"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_key = f"experiments/{experiment_name}_{timestamp}"

        logging.info(f"🚀 Starting upload of experiment: {experiment_name}")
        logging.info(f"📍 S3 location: s3://{self.bucket_name}/{experiment_key}")

        success = True

        # 1. Upload metadata
        metadata = self.create_experiment_metadata(experiment_name, configs, results, training_time)
        metadata_path = f"/tmp/metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        success &= self.upload_file(metadata_path, f"{experiment_key}/metadata.json")
        os.remove(metadata_path)  # Cleanup temp file

        # 2. Upload configs
        config_dir = os.path.dirname(output_dir) + "/configs"
        if os.path.exists(config_dir):
            success &= self.upload_directory(config_dir, f"{experiment_key}/configs")

        # 3. Upload model artifacts
        output_path = Path(output_dir)

        # Find all experiment directories
        for exp_dir in output_path.glob(f"{experiment_name}_*"):
            if exp_dir.is_dir():
                model_type = "non_reasoning" if "non_reasoning" in exp_dir.name else "reasoning"

                # Upload merged model (priority)
                merged_path = exp_dir / "merged_model"
                if merged_path.exists():
                    success &= self.upload_directory(
                        str(merged_path),
                        f"{experiment_key}/models/{model_type}/merged_model"
                    )

                # Upload LoRA adapters (backup)
                final_path = exp_dir / "final_model"
                if final_path.exists():
                    success &= self.upload_directory(
                        str(final_path),
                        f"{experiment_key}/models/{model_type}/final_model",
                        exclude_patterns=["training_args.bin"]  # Skip large training args
                    )

        # 4. Upload evaluation results from experiment directories
        output_path = Path(output_dir)
        for exp_dir in output_path.glob(f"{experiment_name}_*"):
            if exp_dir.is_dir():
                eval_dir = exp_dir / "evals"
                if eval_dir.exists():
                    model_type = "non_reasoning" if "non_reasoning" in exp_dir.name else "reasoning"
                    success &= self.upload_directory(
                        str(eval_dir),
                        f"{experiment_key}/evaluation_results/{model_type}"
                    )

        # 5. Upload training logs (if wandb)
        wandb_dir = os.path.expanduser("~/wandb")
        if os.path.exists(wandb_dir):
            # Find recent wandb runs
            recent_runs = sorted([d for d in Path(wandb_dir).glob("*") if d.is_dir()],
                               key=lambda x: x.stat().st_mtime, reverse=True)[:2]

            for run_dir in recent_runs:
                run_name = run_dir.name
                success &= self.upload_directory(
                    str(run_dir),
                    f"{experiment_key}/training_logs/wandb/{run_name}",
                    exclude_patterns=[".tmp", "tmp", "__pycache__"]
                )

        if success:
            s3_url = f"s3://{self.bucket_name}/{experiment_key}"
            logging.info(f"🎉 Experiment upload completed successfully!")
            logging.info(f"📍 Available at: {s3_url}")
            return s3_url
        else:
            logging.error(f"❌ Some uploads failed for experiment: {experiment_name}")
            return None

    def upload_best_model(self, model_path: str, model_name: str,
                         performance_score: float, benchmark: str) -> bool:
        """Upload a model to the best_models directory"""

        best_model_key = f"best_models/{model_name}"

        # Upload model
        success = self.upload_directory(model_path, f"{best_model_key}/model")

        # Create performance metadata
        perf_metadata = {
            "model_name": model_name,
            "performance": {
                "score": performance_score,
                "benchmark": benchmark,
                "timestamp": datetime.now().isoformat()
            },
            "source_path": model_path
        }

        # Upload performance metadata
        perf_path = f"/tmp/performance_{model_name}.json"
        with open(perf_path, 'w') as f:
            json.dump(perf_metadata, f, indent=2)

        success &= self.upload_file(perf_path, f"{best_model_key}/performance.json")
        os.remove(perf_path)

        if success:
            logging.info(f"🏆 Best model uploaded: {model_name} (score: {performance_score})")

        return success

    def list_experiments(self) -> List[Dict]:
        """List all experiments in S3"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix="experiments/",
                Delimiter="/"
            )

            experiments = []
            for prefix in response.get('CommonPrefixes', []):
                exp_name = prefix['Prefix'].replace('experiments/', '').rstrip('/')

                # Try to get metadata
                try:
                    metadata_key = f"experiments/{exp_name}/metadata.json"
                    metadata_obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=metadata_key)
                    metadata = json.loads(metadata_obj['Body'].read())
                    experiments.append(metadata)
                except:
                    experiments.append({"experiment_info": {"name": exp_name}})

            return experiments

        except Exception as e:
            logging.error(f"Failed to list experiments: {e}")
            return []