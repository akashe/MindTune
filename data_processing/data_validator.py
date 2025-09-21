# data_processing/data_validator.py

class DataValidator:
    def __init__(self, training_data):
        self.data = training_data
    
    def validate_format(self):
        """Ensure all examples have correct format"""
        
        valid_data = []
        
        for example in self.data:
            if all(key in example for key in ['instruction', 'input', 'output']):
                if len(example['output']) > 50:  # Minimum quality check
                    valid_data.append(example)
        
        print(f"Kept {len(valid_data)} out of {len(self.data)} examples")
        return valid_data
    
    def remove_duplicates(self):
        """Remove near-duplicate examples"""
        # Implementation for deduplication
        pass
    
    def check_contamination(self, benchmark_data):
        """Check if any examples are too similar to benchmark questions"""
        # Implementation to detect benchmark contamination
        pass