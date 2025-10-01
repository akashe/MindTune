"""
Topic Extraction Pipeline for Diary Entries
Extracts topics from diary entries using Llama via Ollama for categorized data generation
"""

import requests
import json
from tqdm import tqdm
import logging
from datetime import datetime
from typing import List, Dict, Set

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/topic_extraction.log'),
        logging.StreamHandler()
    ]
)


class TopicExtractor:
    """Extract and categorize topics from diary entries"""

    def __init__(self, ollama_model="llama3.1:8b"):
        self.model = ollama_model
        self.start_time = datetime.now()

        # Expected topic categories based on discussion
        self.expected_categories = [
            "meta_thinking",
            "spirituality",
            "personal_growth",
            "ai_technical",
            "emotional_intelligence",
            "philosophy",
            "productivity",
            "relationships",
            "learning",
            "creativity"
        ]

    def ollama_inference(self, prompt: str, temperature: float = 0.3, max_retries: int = 3) -> str:
        """Local inference using Ollama with retry logic"""

        for attempt in range(max_retries):
            try:
                url = "http://localhost:11434/api/generate"
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": 500  # Topics don't need long responses
                    }
                }

                response = requests.post(url, json=payload, timeout=60)
                response.raise_for_status()

                return response.json()['response']

            except Exception as e:
                logging.warning(f"⚠️  Ollama inference attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(5)
                else:
                    logging.error(f"❌ All Ollama inference attempts failed")
                    return None

    def extract_topics(self, entry: str) -> Dict[str, any]:
        """
        Extract topics from a single diary entry
        Returns: {
            'primary_topics': List[str],
            'secondary_topics': List[str],
            'reasoning_type': str,  # 'pure_reasoning' or 'emotional_understanding'
            'key_insights': List[str]
        }
        """

        topic_extraction_prompt = f"""Analyze this diary entry and extract topics and insights.

Diary Entry:
{entry}

Your task:
1. Identify PRIMARY topics (2-3 most important themes)
2. Identify SECONDARY topics
3. Classify the reasoning type:
   - "pure_reasoning" if it's about logical thinking, problem-solving, planning
   - "emotional_understanding" if it's about emotions, relationships, human behavior
   - "mixed" if it contains both
4. Extract 1-2 key insights or realizations from this entry

Expected topic categories (use these if they fit, or create new ones):
- meta_thinking: How to think, thinking strategies, metacognition
- spirituality: Spiritual insights, consciousness, existence
- personal_growth: What works/doesn't work, lessons learned
- ai_technical: AI concepts, technical ideas about AI
- emotional_intelligence: Understanding emotions, human psychology
- philosophy: Deep philosophical questions and insights
- productivity: Work methods, efficiency, focus
- relationships: Interactions with others, social dynamics
- learning: How to learn, learning strategies
- creativity: Creative process, ideas generation

You will be given a diary entry and the thing discussed in it might vary. Sometimes the user might be talking something eclectic and sometimes it might be more focused on a specific topic.
so be ready to be surprised because sometimes the user might be using some very nuanced jargon or referring to esoteric concepts and ideas.
So please do the best job you can do to extract the topics. These topics will be used to categorize the diary entries for future reference and analysis.


Return ONLY a valid JSON object in this exact format:
{{
  "primary_topics": ["topic1", "topic2"],
  "secondary_topics": ["topic3"],
  "reasoning_type": "pure_reasoning" or "emotional_understanding" or "mixed",
  "key_insights": ["insight 1", "insight 2"]
}}

Return ONLY the JSON, no explanations or additional text."""

        response = self.ollama_inference(topic_extraction_prompt, temperature=0.3)

        if not response:
            logging.error("Failed to extract topics")
            return None

        try:
            # Clean response to extract JSON
            response = response.strip()

            # Try to find JSON in the response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            # Remove newlines within JSON structure for parsing
            topics_data = json.loads(response)

            # Validate structure
            required_keys = ['primary_topics', 'secondary_topics', 'reasoning_type', 'key_insights']
            if not all(key in topics_data for key in required_keys):
                logging.error(f"Missing required keys in response: {topics_data.keys()}")
                return None

            return topics_data

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {e}")
            logging.error(f"Response was: {response}")
            return None

    def process_all_entries(self, entries: List) -> List[Dict]:
        """
        Process all diary entries and extract topics
        Returns list of dicts with entry + topic metadata
        """

        logging.info(f"📚 Processing {len(entries)} diary entries for topic extraction")

        results = []
        failed_count = 0

        for i, entry_doc in tqdm(enumerate(entries)):
            logging.info(f"Processing entry {i+1}/{len(entries)}")

            entry_text = entry_doc["page_content"]
            source = entry_doc.get('source', 'unknown')

            # Skip very short entries
            if len(entry_text) < 50:
                logging.info(f"⏭️  Skipping entry {i+1} (too short: {len(entry_text)} chars)")
                continue

            try:
                topics_data = self.extract_topics(entry_text)

                if topics_data:
                    result = {
                        'entry': entry_text,
                        'source': source,
                        'topics': topics_data,
                        'entry_length': len(entry_text),
                        'processed_at': datetime.now().isoformat()
                    }
                    results.append(result)

                    logging.info(f"✅ Entry {i+1}: {topics_data['primary_topics']} ({topics_data['reasoning_type']})")
                else:
                    failed_count += 1
                    logging.warning(f"❌ Failed to extract topics from entry {i+1}")

                # Progress logging every 10 entries
                if (i + 1) % 10 == 0:
                    self.log_progress(len(results), failed_count)

                # Small delay to prevent overwhelming Ollama
                import time
                time.sleep(2)

            except Exception as e:
                logging.error(f"❌ Error processing entry {i+1}: {e}")
                failed_count += 1
                continue

        self.log_final_stats(results, failed_count)
        return results

    def log_progress(self, success_count: int, failed_count: int):
        """Log current progress"""
        elapsed = datetime.now() - self.start_time
        logging.info(f"""
📊 Progress Update:
  ⏱️  Time elapsed: {elapsed}
  ✅ Successfully processed: {success_count}
  ❌ Failed: {failed_count}
        """)

    def log_final_stats(self, results: List[Dict], failed_count: int):
        """Log final statistics with topic distribution"""
        elapsed = datetime.now() - self.start_time

        # Count topic distribution
        primary_topic_counts = {}
        reasoning_type_counts = {}

        for result in results:
            for topic in result['topics']['primary_topics']:
                primary_topic_counts[topic] = primary_topic_counts.get(topic, 0) + 1

            reasoning_type = result['topics']['reasoning_type']
            reasoning_type_counts[reasoning_type] = reasoning_type_counts.get(reasoning_type, 0) + 1

        logging.info(f"""
🎉 Topic Extraction Complete!
  ⏱️  Total time: {elapsed}
  ✅ Successfully processed: {len(results)}
  ❌ Failed: {failed_count}

📊 Topic Distribution:
  {json.dumps(primary_topic_counts, indent=2)}

🧠 Reasoning Type Distribution:
  {json.dumps(reasoning_type_counts, indent=2)}
        """)

    def save_results(self, results: List[Dict], output_file: str = 'data/diary_topics.json'):
        """Save topic extraction results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logging.info(f"💾 Saved {len(results)} entries with topics to {output_file}")


def main():
    """Main execution function"""

    # Load diary entries
    logging.info("🔍 Loading diary entries...")
    with open("data/all_notes.json", "r") as f:
        entries = json.load(f)
    logging.info(f"📚 Loaded {len(entries)} diary entries")

    # Extract topics
    model_name = "gemma3:12b"
    extractor = TopicExtractor(ollama_model=model_name)
    results = extractor.process_all_entries(entries)

    # Save results
    output_file = f"data/diary_topics_{model_name.replace(':','_').replace('.','_')}.json"
    extractor.save_results(results, output_file)

    logging.info("✅ Topic extraction pipeline complete!")


if __name__ == "__main__":
    main()
