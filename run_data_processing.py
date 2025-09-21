from data_processing.data_validator import DataValidator
from data_processing.create_dataset_from_diary import PrivateDiaryProcessor
from data_processing.retrieve_notes import retrieve_notes
import pdb
from pprint import pprint
from mongodb_connection import DiaryDataManager
import signal
import sys
import logging

def signal_handler(sig, frame):
    """Graceful shutdown handler"""
    logging.info("🛑 Shutdown signal received. Saving progress...")
    if 'mongo_manager' in globals():
        mongo_manager.close()
    sys.exit(0)

def main():

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logging.info("🔒 Starting privacy-preserving dataset creation...")

    all_notes = retrieve_notes()
    logging.info(f"📝 Retrieved {len(all_notes)} notes from local storage.")

    # Initialize MongoDB connection
    mongo_manager = DiaryDataManager()
    
    if not mongo_manager.connect():
        logging.error("❌ Failed to connect to MongoDB. Exiting.")
        return
    
    # Process diary data
    diary_processor = PrivateDiaryProcessor(all_notes, mongo_manager=mongo_manager)
    # diary_processor.create_reasoning_dataset()

    # processed_diary_data = diary_processor.processed_data
    # for i,data in enumerate(processed_diary_data):
    #     print(f'--- Example {i+1} ---')
    #     pprint(data)

    try:
        # Start processing
        diary_processor.create_reasoning_dataset()
        
        # Export final dataset
        output_file = f"data/diary_training_data_{diary_processor.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        mongo_manager.export_to_json(output_file)
        
    except KeyboardInterrupt:
        logging.info("🛑 Processing interrupted by user")
    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}")
    finally:
        # Cleanup
        mongo_manager.close()
        logging.info("🔚 Processing finished")
    
    # Process OpenAI data (user queries only)
    # openai_processor = OpenAIDataProcessor('openai_export.json')
    # user_patterns = openai_processor.extract_user_patterns()
    
    # Combine and validate
    # all_training_data = diary_processor.processed_data + user_patterns
    
    # validator = DataValidator(all_training_data)
    # clean_data = validator.validate_format()
    
    # # Create splits for different experiments
    # datasets = {
    #     'diary_only': [ex for ex in clean_data if 'diary' in ex.get('source', '')],
    #     'queries_only': [ex for ex in clean_data if 'openai' in ex.get('source', '')],
    #     'combined': clean_data
    # }
    
    # # Save datasets
    # for name, data in datasets.items():
    #     with open(f'{name}_training_data.json', 'w') as f:
    #         json.dump(data, f, indent=2)
    #     print(f"✅ Saved {len(data)} examples to {name}_training_data.json")
    
    print("🎉 Dataset creation complete! Your privacy was protected.")

if __name__ == "__main__":
    main()