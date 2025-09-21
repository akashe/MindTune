# data_processing/local_diary_processor.py

import requests
import json
import os
from pathlib import Path
import pdb
import logging
import time
import hashlib
from datetime import datetime

# Setup logging for overnight run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/diary_processing.log'),
        logging.StreamHandler()
    ]
)

class PrivateDiaryProcessor:
    def __init__(self, notes, mongo_manager, ollama_model="llama3.1:8b"):
        self.notes = notes
        self.mongo_manager = mongo_manager
        self.model = ollama_model
        self.processed_count = 0
        self.failed_count = 0
        self.model_name = ollama_model

        self.start_time = datetime.now()

        self.processed_entries = self.mongo_manager.get_processed_entries()
        logging.info(f"🔄 Resuming processing. Already processed: {len(self.processed_entries)} entries")
        
    def ollama_inference(self, prompt, temperature=0.7, max_retries=3):
        """Local inference using Ollama"""
        
        for attempt in range(max_retries):
            try:
                url = "http://localhost:11434/api/generate"
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": 1000
                    }
                }
                
                response = requests.post(url, json=payload)

                response.raise_for_status()

                return response.json()['response']
            except Exception as e:
                logging.warning(f"⚠️  Ollama inference attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    time.sleep(5)  # Exponential backoff
                else:
                      logging.error(f"❌ All Ollama inference attempts failed for prompt")
                      return None
              
    def create_entry_hash(self, entry):
        """Create unique hash for diary entry"""
        return hashlib.md5(entry.encode()).hexdigest()
    
    def create_reasoning_dataset(self):
        """Convert diary entries to reasoning training examples"""
        
        diary_entries = self.notes
        logging.info(f"📚 Loaded {len(diary_entries)} diary entries")

        for i, diary_entry in enumerate(diary_entries):
            logging.info(f"Processing entry {i+1}/{len(diary_entries)}")

            entry = diary_entry.page_content
            entry_hash = self.create_entry_hash(entry)

            # Skip if already processed
            if entry_hash in self.processed_entries:
                logging.info(f"⏭️  Skipping entry {i+1}/{len(diary_entries)} (already processed)")
                continue

            if len(entry)<50:
                logging.info("Entry too short, skipping.")
                continue
            
            try:
                # Create multiple training examples per entry
                reasoning_examples = self.generate_reasoning_examples(entry)
                knowledge_examples = self.generate_knowledge_examples(entry)
                planning_examples = self.generate_planning_examples(entry)
                
                all_examples = reasoning_examples + knowledge_examples + planning_examples
                
                for example in all_examples:
                    saved_id = self.mongo_manager.save_training_example(example, entry_hash)
                    
                    if saved_id:
                        self.processed_count += 1
                    else:
                        self.failed_count += 1
                    
                # Mark this entry as processed
                self.processed_entries.add(entry_hash)
                
                # Log progress every 10 entries
                if (i + 1) % 10 == 0:
                    self.log_progress()
                
                # Small delay to prevent overwhelming the system
                time.sleep(60)
            except Exception as e:
                logging.error(f"❌ Failed to process entry {i+1}: {e}")
                self.failed_count += 1
                continue
        
        # Final statistics
        self.log_final_stats()
    
    def generate_reasoning_examples(self, entry):
        """Generate step-by-step reasoning examples"""
        
        reasoning_prompt = f"""
Based on this diary entry, create 2 training examples that capture the person's reasoning style:

Diary Entry: {entry}

For each example, create:
1. A similar problem/situation that requires step-by-step thinking
2. How this person would approach it systematically
3. Their reasoning process and conclusion

Format as JSON:
[
  {{
    "instruction": "How would you approach this situation?",
    "input": "situation description", 
    "output": "Let me think through this step by step:\\n\\n**First**: I would...\\n**Next**: I would...\\n**Finally**: ...\\n\\n**Why this works**: ..."
  }},
  {{
    "instruction": "What's your reasoning for this decision?",
    "input": "decision scenario",
    "output": "Let me analyze this systematically:\\n1. ...\\n2. ...\\nTherefore: ..."
  }}
]

Only return the JSON and no explainer text.
"""
        
        response = self.ollama_inference(reasoning_prompt, temperature=0.8)
        
        try:
            examples = json.loads(response.replace('\n',''))
            if examples and isinstance(examples, list):
              for example in examples:
                example['entry'] = entry  # Add original entry for context
                example['template'] = "reasoning"
                example['model'] = self.model_name
              return examples
            else:
              logging.info("No reasoning examples generated from this entry.")
              return []
        except Exception as e:
            logging.warning("Failed to parse reasoning examples, {e}")
            return []
    
    def generate_knowledge_examples(self, entry):
        """Generate factual knowledge examples"""
        
        knowledge_prompt = f"""
From this diary entry, create 2 training examples about the person's experiences, preferences, or knowledge:

Diary Entry: {entry}

Create factual Q&A pairs about:
- Personal experiences mentioned
- Preferences revealed  
- Knowledge demonstrated
- Opinions expressed

Format as JSON:
[
  {{
    "instruction": "Tell me about your experience with...",
    "input": "",
    "output": "Based on my experience..."
  }},
  {{
    "instruction": "What do you think about...?", 
    "input": "",
    "output": "In my opinion..."
  }}
]

Only return the JSON and no explainer text.
"""
        
        response = self.ollama_inference(knowledge_prompt)
        
        try:
            examples = json.loads(response.replace('\n',''))
            if examples and isinstance(examples, list):
              for example in examples:
                example['entry'] = entry  # Add original entry for context
                example['template'] = "general_knowledge"
                example['model'] = self.model_name
              return examples
            else:
              logging.info("No General Knowledge examples generated from this entry.")
              return []
        except Exception as e:
            logging.warning("Failed to parse knowledge examples, {e}")
            return []
    
    def generate_planning_examples(self, entry):
        """Generate planning and decision-making examples"""
        
        planning_prompt = f"""
From this diary entry, create 1 training example about planning or decision-making:

Diary Entry: {entry}

If the entry shows planning, decision-making, or problem-solving, create:
[
{{
  "instruction": "How would you plan for this situation?",
  "input": "planning scenario",
  "output": "Here's how I would approach this:\\n1. First, I'd...\\n2. Next, I'd...\\n3. Finally, I'd...\\n\\nThis approach works because...",
}}
]

If no planning is evident, return: []


Only return the JSON and no explainer text.
"""
        
        response = self.ollama_inference(planning_prompt)
        
        try:
            example = json.loads(response.replace('\n',''))
            if example and isinstance(example, list) and len(example)>0:
                for example_ in example:
                    example_['entry'] = entry  # Add original entry for context
                    example_['template'] = "planning"
                    example_['model'] = self.model_name
                return example
            else:
                logging.info("No planning examples generated from this entry.")
                return []
        except Exception as e:
            logging.warning("Failed to parse planning examples, {e}")
            return []
    
    # def save_progress(self):
    #     """Save processed data"""
        
    #     with open('diary_training_data.json', 'w') as f:
    #         json.dump(self.processed_data, f, indent=2)
        
    #     print(f"Saved {len(self.processed_data)} training examples")

    def log_progress(self):
        """Log current progress"""
        elapsed = datetime.now() - self.start_time
        stats = self.mongo_manager.get_stats()
        
        logging.info(f"""
                    📊 Progress Update:
                      ⏱️  Time elapsed: {elapsed}
                      ✅ Examples saved: {self.processed_count}
                      ❌ Failed: {self.failed_count}
                      📁 Total in DB: {stats['total_examples']}
                      📈 By template: {stats['by_template']}
                            """)
    
    def log_final_stats(self):
        """Log final processing statistics"""
        elapsed = datetime.now() - self.start_time
        stats = self.mongo_manager.get_stats()
        
        logging.info(f"""
                    🎉 Processing Complete!
                      ⏱️  Total time: {elapsed}
                      ✅ Examples processed: {self.processed_count}
                      ❌ Failed: {self.failed_count}
                      📁 Total examples in DB: {stats['total_examples']}
                      📊 Breakdown: {stats['by_template']}
                            """)


if __name__ == "__main__":
  
    # Usage
    processor = PrivateDiaryProcessor('path/to/diary.txt')
    processor.create_reasoning_dataset()