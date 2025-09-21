# export_to_drive.py - Run this locally
from mongodb_connection import DiaryDataManager
import json
import os

def export_for_colab():
    """Export data to JSON for Google Drive upload"""
    
    # Connect to local MongoDB
    mongo_manager = DiaryDataManager()
    
    # Export to JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"diary_training_data_{timestamp}.json"
    
    count = mongo_manager.export_to_json(filename)
    print(f"✅ Exported {count} examples to {filename}")
    
    # Create metadata file
    stats = mongo_manager.get_stats()
    metadata = {
        "export_date": timestamp,
        "total_examples": count,
        "breakdown": stats['by_template'],
        "description": "Personal diary training data for fine-tuning experiment"
    }
    
    with open(f"metadata_{timestamp}.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📋 Upload {filename} and metadata_{timestamp}.json to Google Drive")
    print(f"🔗 Then use the Google Drive integration in Colab")

if __name__ == "__main__":
    export_for_colab()