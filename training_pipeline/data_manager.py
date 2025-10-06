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
        """Prepare training datasets for both model types"""
        
        # Load raw data
        self.raw_data = self.load_data()
        
        # Use subset for testing
        if test_mode:
            self.raw_data = self.raw_data[:self.config.test_size]
            logging.info(f"🧪 Test mode: Using {len(self.raw_data)} examples")
        
        # # Split by template type
        # reasoning_data = [ex for ex in self.raw_data if ex.get('template') == 'reasoning']
        # general_data = [ex for ex in self.raw_data if ex.get('template') in ['general_knowledge', 'planning']]
        
        # Both reasoning and non-reasoning models have access to same data
        reasoning_data = deepcopy(self.raw_data)
        general_data = deepcopy(self.raw_data)

        logging.info(f"📊 Data split: {len(reasoning_data)} reasoning, {len(general_data)} general")
        
        # Prepare datasets for each model type
        datasets = {
            'non_reasoning': self._prepare_non_reasoning_dataset(general_data),
            'reasoning': self._prepare_reasoning_dataset(reasoning_data)
        }
        
        return datasets
    
    def _prepare_non_reasoning_dataset(self, data: List[Dict]) -> DatasetDict:
        """Prepare dataset for non-reasoning model (standard instruction tuning)"""

        formatted_data = []
        for example in data:
            # Standard Alpaca format
            if example.get('input', '').strip():
                text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
            else:
                text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{example['instruction']}

### Response:
{example['output']}"""

            # Preserve data_type for stratified splitting
            data_type = example.get('metadata', {}).get('data_type', 'unknown')

            formatted_data.append({
                "text": text,
                "data_type": data_type,
                "template": example.get('template', 'unknown')
            })

        return self._create_train_val_split(formatted_data)
    
    def _prepare_reasoning_dataset(self, data: List[Dict]) -> DatasetDict:
        """Prepare dataset for reasoning model (enhanced reasoning format)"""

        formatted_data = []
        for example in data:
            # Enhanced reasoning format
            enhanced_output = self._enhance_reasoning_output(example['output'])

            if example.get('input', '').strip():
                text = f"""Below is an instruction that describes a task, paired with an input. Write a step-by-step response that shows your reasoning process.

### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{enhanced_output}"""
            else:
                text = f"""Below is an instruction that describes a task. Write a step-by-step response that shows your reasoning process.

### Instruction:
{example['instruction']}

### Response:
{enhanced_output}"""

            # Preserve data_type for stratified splitting
            data_type = example.get('metadata', {}).get('data_type', 'unknown')

            formatted_data.append({
                "text": text,
                "data_type": data_type,
                "template": "reasoning"
            })

        return self._create_train_val_split(formatted_data)
    
    def _enhance_reasoning_output(self, output: str) -> str:
        """Enhance reasoning output format"""
        
        # Convert mechanical steps to more natural reasoning
        enhanced = output.replace("Step 1:", "\n**Analysis:**")
        enhanced = enhanced.replace("Step 2:", "\n**Approach:**")
        enhanced = enhanced.replace("Step 3:", "\n**Implementation:**")
        enhanced = enhanced.replace("Conclusion:", "\n**Conclusion:**")
        
        # Add reasoning preamble if not present
        if not enhanced.startswith("Let me think"):
            enhanced = "Let me think through this systematically:\n" + enhanced
        
        return enhanced
    
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