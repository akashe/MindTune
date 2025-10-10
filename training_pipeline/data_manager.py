# data_manager.py
import json
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from mongodb_connection import DiaryDataManager
from config import DataConfig
import logging
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

class DataManager:
    def __init__(self, config: DataConfig):
        self.config = config
        self.raw_data = None
        self.processed_datasets = {}
        
    def load_data(self) -> List[Dict]:
        """Load data from MongoDB or JSON file"""
        
        if self.config.source == "mongodb":
            return self._load_from_mongodb()
        elif self.config.source == "json":
            return self._load_from_json()
        else:
            raise ValueError(f"Unknown data source: {self.config.source}")
    
    def _load_from_mongodb(self) -> List[Dict]:
        """Load data from MongoDB"""
        logging.info("📁 Loading data from MongoDB...")
        
        mongo_manager = DiaryDataManager(
            connection_string=self.config.mongodb_uri,
            db_name="diary_training"
        )
        
        try:
            collection = mongo_manager.collection
            cursor = collection.find({}, {"_id": 0, "created_at": 0, "processed_at": 0})
            data = list(cursor)
            logging.info(f"✅ Loaded {len(data)} examples from MongoDB")
            return data
        finally:
            mongo_manager.close()
    
    def _load_from_json(self) -> List[Dict]:
        """Load data from JSON file"""
        logging.info(f"📁 Loading data from {self.config.json_path}...")
        
        with open(self.config.json_path, 'r') as f:
            data = json.load(f)
        
        logging.info(f"✅ Loaded {len(data)} examples from JSON")
        return data
    
    def prepare_datasets(self, test_mode: bool = False) -> Dict[str, DatasetDict]:
        """Prepare training dataset with clean, simple formatting"""

        # Load raw data
        self.raw_data = self.load_data()

        # Use subset for testing
        if test_mode:
            self.raw_data = self.raw_data[:self.config.test_size]
            logging.info(f"🧪 Test mode: Using {len(self.raw_data)} examples")

        logging.info(f"📊 Total examples: {len(self.raw_data)}")

        # Prepare single unified dataset
        dataset = self._prepare_dataset(self.raw_data)

        # Return in dict format for compatibility (but only one model type now)
        datasets = {
            'unified': dataset
        }

        return datasets
    
    def _prepare_dataset(self, data: List[Dict]) -> DatasetDict:
        """Prepare dataset with clean, simple formatting - no verbose system prompts"""

        formatted_data = []
        for example in data:
            # Simple, clean format - just instruction, input (if present), and response
            if example.get('input', '').strip():
                text = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
            else:
                text = f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}"""

            # Preserve data_type for stratified splitting
            data_type = example.get('metadata', {}).get('data_type', 'unknown')

            formatted_data.append({
                "text": text,
                "data_type": data_type
            })

        return self._create_train_val_split(formatted_data)
    
    def _create_train_val_split(self, data: List[Dict]) -> DatasetDict:
        """Create stratified train/validation split by data_type"""

        if len(data) < 2:
            # Not enough data for split
            dataset = Dataset.from_list(data)
            return DatasetDict({
                "train": dataset,
                "validation": dataset
            })

        # Extract data_type labels for stratification
        data_types = [item.get('data_type', 'unknown') for item in data]

        # Check if we have valid data_types
        unique_types = set(data_types)
        if len(unique_types) <= 1 or 'unknown' in unique_types:
            # Fallback to random split if no data_type info
            logging.warning("⚠️  No data_type found, using random split")
            train_data, val_data = train_test_split(
                data,
                test_size=self.config.validation_split,
                random_state=42
            )
        else:
            # Stratified split by data_type
            try:
                train_data, val_data = train_test_split(
                    data,
                    test_size=self.config.validation_split,
                    random_state=42,
                    stratify=data_types
                )
                logging.info(f"✅ Created stratified split by data_type")

                # Log distribution
                from collections import Counter
                train_types = Counter([item.get('data_type') for item in train_data])
                val_types = Counter([item.get('data_type') for item in val_data])

                logging.info(f"Train distribution: {dict(train_types)}")
                logging.info(f"Val distribution: {dict(val_types)}")

            except ValueError as e:
                # Stratification failed (e.g., too few samples in some classes)
                logging.warning(f"⚠️  Stratification failed: {e}, using random split")
                train_data, val_data = train_test_split(
                    data,
                    test_size=self.config.validation_split,
                    random_state=42
                )

        return DatasetDict({
            "train": Dataset.from_list(train_data),
            "validation": Dataset.from_list(val_data)
        })