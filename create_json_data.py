import json
from mongodb_connection import DiaryDataManager
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mongo_manager = DiaryDataManager()
    
    if not mongo_manager.connect():
        logging.error("❌ Failed to connect to MongoDB. Exiting.")
    else:
        output_file = "data/exported_diary_data_1000.json"
        mongo_manager.export_to_json(output_file, data_size=1000)  # Export only 100 entries for testing
        mongo_manager.close()
        logging.info(f"✅ Data exported to {output_file}")