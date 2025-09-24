# mongodb_connection.py
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import logging
from datetime import datetime

class DiaryDataManager:
    def __init__(self, connection_string="mongodb://localhost:27017/", db_name="diary_training"):
        self.connection_string = connection_string
        self.db_name = db_name
        self.client = None
        self.db = None
        self.collection = None
        self.connect()
        
    def connect(self):
        """Establish MongoDB connection"""
        try:
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ismaster')
            self.db = self.client[self.db_name]
            self.collection = self.db.training_examples
            
            # Create indexes for better performance
            self.collection.create_index("entry_hash")
            self.collection.create_index("template")
            self.collection.create_index("created_at")
            
            logging.info(f"✅ Connected to MongoDB: {self.db_name}")
            return True
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logging.error(f"❌ Failed to connect to MongoDB: {e}")
            return False
    
    def save_training_example(self, example_data, entry_hash):
        """Save a single training example"""
        try:
            # Add metadata
            document = {
                **example_data,
                "entry_hash": entry_hash,
                "created_at": datetime.utcnow(),
                "processed_at": datetime.utcnow()
            }
            
            # Insert document
            result = self.collection.insert_one(document)
            logging.info(f"💾 Saved example: {result.inserted_id}")
            return result.inserted_id
            
        except Exception as e:
            logging.error(f"❌ Failed to save example: {e}")
            return None
    
    def get_processed_entries(self):
        """Get list of already processed entry hashes"""
        try:
            processed = self.collection.distinct("entry_hash")
            return set(processed)
        except Exception as e:
            logging.error(f"❌ Failed to get processed entries: {e}")
            return set()
    
    def get_stats(self):
        """Get processing statistics"""
        try:
            total = self.collection.count_documents({})
            by_template = list(self.collection.aggregate([
                {"$group": {"_id": "$template", "count": {"$sum": 1}}}
            ]))
            
            return {
                "total_examples": total,
                "by_template": {item["_id"]: item["count"] for item in by_template}
            }
        except Exception as e:
            logging.error(f"❌ Failed to get stats: {e}")
            return {"total_examples": 0, "by_template": {}}
    
    def export_to_json(self, output_file, data_size=None):
        """Export all data to JSON for training"""
        try:
            cursor = self.collection.find({}, {"_id": 0, "entry_hash": 0, "created_at": 0, "processed_at": 0})
            data = list(cursor)
            if data_size:
                data = data[:data_size]
            
            import json
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logging.info(f"📁 Exported {len(data)} examples to {output_file}")
            return len(data)
            
        except Exception as e:
            logging.error(f"❌ Failed to export data: {e}")
            return 0
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logging.info("🔒 MongoDB connection closed")