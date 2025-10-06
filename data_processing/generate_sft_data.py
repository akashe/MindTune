"""
SFT Data Generation Pipeline
Generates supervised fine-tuning data from diary entries with topics
"""

import json
import requests
import logging
import time
import hashlib
import re
from datetime import datetime
from typing import List, Dict
from collections import Counter

from topic_taxonomy import TopicTaxonomy
from prompt_templates_v2 import get_prompt_template_v2
import sys
sys.path.append('..')
from mongodb_connection import DiaryDataManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../data/sft_generation.log'),
        logging.StreamHandler()
    ]
)


class SFTDataGenerator:
    """Generate SFT training data from diary entries"""

    def __init__(self, ollama_model="llama3.1:8b", mongo_manager=None, version = "1.0"):
        self.model = ollama_model
        self.mongo = mongo_manager or DiaryDataManager()
        self.taxonomy = TopicTaxonomy()
        self.version = version

        self.start_time = datetime.now()
        self.stats = {
            'entries_processed': 0,
            'entries_skipped': 0,
            'examples_generated': 0,
            'examples_rejected': 0,
            'by_category': Counter(),
            'by_data_type': Counter()
        }

    def ollama_inference(self, prompt: str, temperature: float = 0.7, max_retries: int = 3) -> str:
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
                        "num_predict": 4000  # Allow longer responses for multiple examples
                    }
                }

                response = requests.post(url, json=payload, timeout=120)
                response.raise_for_status()

                return response.json()['response']

            except Exception as e:
                logging.warning(f"⚠️  Ollama inference attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    logging.error(f"❌ All Ollama inference attempts failed")
                    return None

    def assess_entry_quality(self, entry_text: str, topics: Dict) -> float:
        """Pre-assess entry quality (0.0 to 1.0)"""

        quality_score = 0.0

        # Check 1: Has clear topics
        if len(topics.get('primary_topics', [])) >= 1:
            quality_score += 0.3

        # Check 2: Has insights
        if len(topics.get('key_insights', [])) >= 1:
            quality_score += 0.2

        # Check 3: Sufficient length
        word_count = len(entry_text.split())
        if word_count > 200:
            quality_score += 0.2
        if word_count > 400:
            quality_score += 0.2

        # Check 4: Has reasoning markers
        reasoning_markers = ['because', 'therefore', 'realized', 'understand', 'understood', 'hit', 'why',
                            'noticed', 'thought', 'idea', 'insight', 'learned', 'learnt', 'lesson', 'pondered',
                            'discovered', 'figured', 'concluded']
        marker_count = sum(1 for marker in reasoning_markers if marker in entry_text.lower())
        if marker_count >= 2:
            quality_score += 0.2

        return min(quality_score, 1.0)

    def calculate_example_count(self, entry_text: str, quality_score: float) -> int:
        """Determine how many examples to generate"""

        entry_length = len(entry_text.split())

        if quality_score < 0.4:
            return 0  # Skip low quality

        elif quality_score < 0.7:  # Medium quality
            if entry_length < 100:
                return 1
            elif entry_length < 300:
                return 2
            else:
                return 3

        else:  # High quality (>= 0.7)
            if entry_length < 100:
                return 1
            elif entry_length < 200:
                return 2
            elif entry_length < 300:
                return 3
            elif entry_length < 500:
                return 4
            elif entry_length < 700:
                return 5
            elif entry_length < 1000:
                return 7
            elif entry_length < 1500:
                return 10
            else:
                return 15

    def get_data_types_for_category(self, category: str, reasoning_type: str) -> List[str]:
        """Determine which data types to generate for this category"""

        TYPE_MAPPING = {
            'meta_thinking': {
                'pure_reasoning': ['chain_of_thought', 'conceptual_reasoning', 'qa'],
                'mixed': ['chain_of_thought', 'qa'],
                'emotional_understanding': ['qa']
            },
            'ai_technical': {
                'pure_reasoning': ['conceptual_reasoning', 'chain_of_thought', 'qa'],
                'mixed': ['conceptual_reasoning', 'qa'],
                'emotional_understanding': ['qa']
            },
            'emotional_intelligence': {
                'emotional_understanding': ['multiple_choice', 'qa'],
                'mixed': ['qa'],
                'pure_reasoning': ['qa']
            },
            'personal_growth': {
                'pure_reasoning': ['chain_of_thought', 'qa'],
                'emotional_understanding': ['qa'],
                'mixed': ['chain_of_thought', 'qa']
            },
            'spirituality': {
                'pure_reasoning': ['deep_reasoning', 'conceptual_reasoning', 'qa'],
                'mixed': ['deep_reasoning', 'qa'],
                'emotional_understanding': ['qa']
            },
            'creativity': {
                'pure_reasoning': ['qa', 'conceptual_reasoning'],
                'mixed': ['qa'],
                'emotional_understanding': ['qa']
            }
        }

        # Handle specific categories (rare topics)
        if category.startswith('specific_'):
            return ['qa']  # Just use Q&A for specific topics

        return TYPE_MAPPING.get(category, {}).get(reasoning_type, ['qa'])

    def get_prompt_template_old(self, category: str, data_type: str) -> str:
        """Get prompt template for data generation"""

        # Base instruction
        base_rules = """CRITICAL RULES FOR DATA GENERATION:
1. ALL answers must come DIRECTLY from the diary entry - NO imagination or external knowledge
2. DO NOT create fictional scenarios that aren't in the entry
3. DO NOT add your own opinions or general knowledge
4. Extract the person's actual thinking patterns, insights, and reasoning from the entry
5. Use the person's own words and concepts where possible
6. If the entry doesn't contain enough information for a question type, skip it

"""

        if data_type == 'chain_of_thought':
            return f"""You are creating training data from a personal diary entry.

Diary Entry:
{{entry}}

{base_rules}

Your task: Create {{num_examples}} chain-of-thought reasoning examples.

Create questions about:
- Reasoning processes shown IN THE ENTRY
- Thinking patterns demonstrated IN THE ENTRY
- Lessons or realizations from THE ENTRY
- Problem-solving approaches from THE ENTRY

Each answer should show step-by-step thinking that comes from the entry.

Format as JSON array:
[
  {{
    "instruction": "Question about reasoning/thinking from the entry",
    "input": "",
    "output": "Let me think through this step by step:\\n\\n**Step 1**: [First insight from entry]...\\n\\n**Step 2**: [Next insight from entry]...\\n\\n**Therefore**: [Conclusion from entry]..."
  }}
]

Return ONLY a valid JSON array, no other text.
"""

        elif data_type == 'conceptual_reasoning':
            return f"""You are creating training data from a personal diary entry.

Diary Entry:
{{entry}}

{base_rules}

Your task: Create {{num_examples}} conceptual reasoning examples.

Create questions about:
- Concepts or ideas explained IN THE ENTRY
- Connections made IN THE ENTRY
- Understanding demonstrated IN THE ENTRY
- Frameworks or mental models from THE ENTRY

Each answer should explain concepts using the person's perspective from the entry.

Format as JSON array:
[
  {{
    "instruction": "Question about a concept from the entry",
    "input": "",
    "output": "Based on the entry: [explanation using entry's concepts and language]..."
  }}
]

Return ONLY a valid JSON array, no other text.
"""

        elif data_type == 'multiple_choice':
            return f"""You are creating training data from a personal diary entry.

Diary Entry:
{{entry}}

{base_rules}

Your task: Create {{num_examples}} multiple choice scenarios.

Base scenarios on:
- Situations described IN THE ENTRY
- Social/emotional dynamics from THE ENTRY
- Decisions or dilemmas from THE ENTRY

The correct answer must be grounded in insights from the entry.

Format as JSON array:
[
  {{
    "instruction": "Scenario based on entry",
    "input": "",
    "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "output": "B",
    "reasoning": "Explanation using entry's insights"
  }}
]

Return ONLY a valid JSON array, no other text.
"""

        elif data_type == 'deep_reasoning':
            return f"""You are creating training data from a personal diary entry about philosophy/spirituality.

Diary Entry:
{{entry}}

{base_rules}

Your task: Create {{num_examples}} deep reasoning examples.

Create questions about:
- Philosophical ideas explored IN THE ENTRY
- Spiritual insights from THE ENTRY
- Abstract concepts from THE ENTRY
- Deep questions raised IN THE ENTRY

Answers should preserve the person's unique philosophical perspective.

Format as JSON array:
[
  {{
    "instruction": "Deep question from the entry",
    "input": "",
    "output": "Philosophical exploration based on entry: [reasoning from entry]..."
  }}
]

Return ONLY a valid JSON array, no other text.
"""

        else:  # Default to 'qa'
            return f"""You are creating training data from a personal diary entry.

Diary Entry:
{{entry}}

{base_rules}

Your task: Create {{num_examples}} question-answer pairs.

Create diverse questions about:
- Insights or realizations from THE ENTRY
- Experiences described IN THE ENTRY
- Lessons learned IN THE ENTRY
- Ideas or thoughts from THE ENTRY

Each answer should directly address content from the entry.

Format as JSON array:
[
  {{
    "instruction": "Question about entry content",
    "input": "",
    "output": "Answer from the entry: [content from entry]..."
  }}
]

Return ONLY a valid JSON array, no other text.
"""

    def check_grounding(self, output_text: str, entry_text: str) -> bool:
        """Verify output is grounded in entry"""

        # Extract key words (simple approach)
        def extract_keywords(text):
            # Remove common words
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                        'to', 'for', 'of', 'with', 'by', 'from', 'is', 'was',
                        'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                        'do', 'does', 'did', 'will', 'would', 'should', 'could',
                        'may', 'might', 'can', 'this', 'that', 'these', 'those',
                        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
                        'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
                        'both', 'few', 'more', 'most', 'other', 'some', 'such'}

            words = re.findall(r'\b[a-z]+\b', text.lower())
            return [w for w in words if w not in stopwords and len(w) > 3]

        output_keywords = extract_keywords(output_text)
        entry_keywords = extract_keywords(entry_text)

        if len(output_keywords) == 0:
            return False

        # Check overlap
        overlap = len(set(output_keywords) & set(entry_keywords))
        overlap_ratio = overlap / len(output_keywords)

        # Need at least 30% keyword overlap
        if overlap_ratio < 0.2:
            return False

        # Check for generic AI phrases (hallucination indicators)
        generic_phrases = ['as an ai', 'in general', 'typically', 'usually',
                          'in conclusion', 'to summarize', 'in summary',
                          'it is important', 'one should', 'you should always']
        if any(phrase in output_text.lower() for phrase in generic_phrases):
            return False

        return True

    def parse_and_validate_json(self, response: str) -> List[Dict]:
        """Parse JSON from LLM response and validate"""

        if not response:
            return []

        try:
            # Clean response
            response = response.strip()

            # Handle thinking tokens (DeepSeek R1, etc.)
            if "<think>" in response and "</think>" in response:
                response = response.split("</think>")[1].strip()

            # Extract JSON
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            # Find JSON array
            if "[" in response and "]" in response:
                start_idx = response.find("[")
                end_idx = response.rfind("]") + 1
                response = response[start_idx:end_idx]

            # Parse
            examples = json.loads(response)

            if not isinstance(examples, list):
                logging.warning("Response is not a JSON array")
                return []

            return examples

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON: {e}")
            logging.error(f"Response was: {response[:500]}...")
            return []

    def generate_examples_for_entry(self, entry_data: Dict) -> List[Dict]:
        """Generate training examples for a single entry"""

        entry_text = entry_data['entry']
        topics_data = entry_data['topics']
        source = entry_data.get('source', 'unknown')

        # Step 1: Assess quality
        quality_score = self.assess_entry_quality(entry_text, topics_data)

        # Step 2: Calculate example count
        num_examples = self.calculate_example_count(entry_text, quality_score)

        if num_examples == 0:
            logging.info(f"⏭️  Skipping low quality entry: {source} (quality: {quality_score:.2f})")
            self.stats['entries_skipped'] += 1
            return []

        # Step 3: Get category
        primary_topics = topics_data.get('primary_topics', [])
        category = self.taxonomy.get_category_for_topics(primary_topics)
        reasoning_type = topics_data.get('reasoning_type', 'mixed')

        # Skip specific_ topics (rare/outlier topics)
        if category.startswith('specific_'):
            logging.info(f"⏭️  Skipping entry with specific topic: {category}")
            self.stats['entries_skipped'] += 1
            return []

        # Step 4: Get data types
        data_types = self.get_data_types_for_category(category, reasoning_type)

        logging.info(f"📝 Processing {source}: {category} ({reasoning_type}) - {num_examples} examples across {len(data_types)} types")

        all_examples = []
        examples_per_type = max(1, num_examples // len(data_types))

        # Step 5: Generate for each data type with 50/50 input/no-input split
        for data_type in data_types:

            # Split examples between with_input and no_input variants
            # Priority to no_input when can't split evenly
            examples_no_input = (examples_per_type + 1) // 2  # Ceiling division
            examples_with_input = examples_per_type // 2       # Floor division

            # Generate no_input variant
            if examples_no_input > 0:
                try:
                    prompt_template = get_prompt_template_v2(data_type, 'no_input')
                    # Replace placeholders manually to avoid KeyError from braces in entry_text
                    prompt = prompt_template.replace('{entry}', entry_text)
                    prompt = prompt.replace('{num_examples}', str(examples_no_input))
                except Exception as e:
                    logging.error(f"Template error for {data_type}: {e}")
                    continue

                # Retry logic for inference + parsing
                examples = []
                for attempt in range(3):
                    response = self.ollama_inference(prompt, temperature=0.7)
                    if response:
                        examples = self.parse_and_validate_json(response)
                        if examples:  # Successfully parsed
                            break
                        else:
                            logging.warning(f"Retry {attempt + 1}/3 for {data_type} no_input parsing")

                if examples:
                    self._process_examples(examples, entry_text, source, category,
                                          data_type, reasoning_type, quality_score,
                                          'no_input', all_examples)

            # Generate with_input variant
            if examples_with_input > 0:
                try:
                    prompt_template = get_prompt_template_v2(data_type, 'with_input')
                    # Replace placeholders manually to avoid KeyError from braces in entry_text
                    prompt = prompt_template.replace('{entry}', entry_text)
                    prompt = prompt.replace('{num_examples}', str(examples_with_input))
                except Exception as e:
                    logging.error(f"Template error for {data_type} with_input: {e}")
                    continue

                # Retry logic for inference + parsing
                examples = []
                for attempt in range(3):
                    response = self.ollama_inference(prompt, temperature=0.7)
                    if response:
                        examples = self.parse_and_validate_json(response)
                        if examples:  # Successfully parsed
                            break
                        else:
                            logging.warning(f"Retry {attempt + 1}/3 for {data_type} with_input parsing")

                if examples:
                    self._process_examples(examples, entry_text, source, category,
                                          data_type, reasoning_type, quality_score,
                                          'with_input', all_examples)
        time.sleep(18)  

        logging.info(f"✅ Generated {len(all_examples)} examples for {source}")
        return all_examples

    def _process_examples(self, examples: List[Dict], entry_text: str, source: str,
                         category: str, data_type: str, reasoning_type: str,
                         quality_score: float, variant: str, all_examples: List[Dict]):
        """Helper to process and validate generated examples"""

        # Forbidden phrases that indicate the example references the entry
        forbidden_phrases = [
            'based on entry', 'based on the entry',
            'the person', 'the diary', 'the entry',
            'according to the entry', 'from the entry',
            'the diary mentions', 'as mentioned in'
        ]

        for example in examples:
            # Check required keys
            if 'instruction' not in example or 'output' not in example:
                logging.warning("Example missing required keys")
                self.stats['examples_rejected'] += 1
                continue

            # Check for source_insight (required in V2)
            if 'source_insight' not in example:
                logging.warning("Example missing source_insight")
                self.stats['examples_rejected'] += 1
                continue

            # Check for forbidden phrases in instruction and output
            combined_text = (example['instruction'] + ' ' + example['output']).lower()
            if any(phrase in combined_text for phrase in forbidden_phrases):
                logging.warning(f"Example contains forbidden reference phrase")
                self.stats['examples_rejected'] += 1
                continue

            # Note: Grounding check disabled for V2.5
            # We're generating generalized insights, not direct quotes
            # The source_insight field provides the connection to the entry
            # if not self.check_grounding(example['output'], entry_text):
            #     logging.warning("Example not grounded in entry")
            #     self.stats['examples_rejected'] += 1
            #     continue

            # Ensure input field exists (even if empty)
            if 'input' not in example:
                example['input'] = ""

            # Store source_insight separately in metadata, remove from main example
            source_insight = example.pop('source_insight')
            print(source_insight)
            print("--- ------------- ")

            # Add metadata
            example['metadata'] = {
                'source_entry': source,
                'source_text': entry_text,
                'category': category,
                'data_type': data_type,
                'reasoning_type': reasoning_type,
                'variant': variant,
                'quality_score': quality_score,
                'source_insight': source_insight,  # For verification
                'generated_at': datetime.now().isoformat(),
                'model': self.model,
                'version': self.version
            }

            all_examples.append(example)
            self.stats['examples_generated'] += 1
            self.stats['by_category'][category] += 1
            self.stats['by_data_type'][data_type] += 1

    def create_entry_hash(self, entry: str) -> str:
        """
        Create unique hash for entry + version
        This allows same entry to be processed by different versions
        """
        hash_input = entry + self.version
        return hashlib.md5(hash_input.encode()).hexdigest()

    def process_all_entries(self, diary_entries: List[Dict]):
        """Main processing loop"""

        logging.info(f"📚 Starting SFT data generation for {len(diary_entries)} entries")

        # Get already processed entries
        processed_hashes = self.mongo.get_processed_entries() if self.mongo else set()
        logging.info(f"🔄 Resuming: {len(processed_hashes)} entries already processed")

        for i, entry_data in enumerate(diary_entries):
            logging.info(f"\n{'='*60}")
            logging.info(f"Entry {i+1}/{len(diary_entries)}")

            entry_text = entry_data['entry']
            entry_hash = self.create_entry_hash(entry_text)

            # Skip if already processed
            if entry_hash in processed_hashes:
                logging.info(f"⏭️  Skipping (already processed)")
                continue

            try:
                # Generate examples
                examples = self.generate_examples_for_entry(entry_data)

                # Save to MongoDB
                if self.mongo and len(examples) > 0:
                    for example in examples:
                        self.mongo.save_training_example(example, entry_hash)

                self.stats['entries_processed'] += 1

                # Progress logging
                if (i + 1) % 10 == 0:
                    self.log_progress()

                # Rate limiting
                time.sleep(3)

            except Exception as e:
                logging.error(f"❌ Error processing entry {i+1}: {e}")
                continue

        self.log_final_stats()

    def log_progress(self):
        """Log current progress"""
        elapsed = datetime.now() - self.start_time
        logging.info(f"""
📊 Progress Update:
  ⏱️  Time elapsed: {elapsed}
  ✅ Entries processed: {self.stats['entries_processed']}
  ⏭️  Entries skipped: {self.stats['entries_skipped']}
  📝 Examples generated: {self.stats['examples_generated']}
  ❌ Examples rejected: {self.stats['examples_rejected']}
        """)

    def log_final_stats(self):
        """Log final statistics"""
        elapsed = datetime.now() - self.start_time

        logging.info(f"""
🎉 SFT Data Generation Complete!
  ⏱️  Total time: {elapsed}
  ✅ Entries processed: {self.stats['entries_processed']}
  ⏭️  Entries skipped: {self.stats['entries_skipped']}
  📝 Examples generated: {self.stats['examples_generated']}
  ❌ Examples rejected: {self.stats['examples_rejected']}

📊 By Category:
  {json.dumps(dict(self.stats['by_category']), indent=2)}

📊 By Data Type:
  {json.dumps(dict(self.stats['by_data_type']), indent=2)}
        """)


def main():
    """Main execution"""

    # Load diary entries with topics
    logging.info("🔍 Loading diary entries with topics...")
    with open('../data/diary_topics.json', 'r') as f:
        diary_data = json.load(f)
    logging.info(f"📚 Loaded {len(diary_data)} entries")

    # Initialize MongoDB
    mongo = DiaryDataManager()

    # Initialize generator
    generator = SFTDataGenerator(
        ollama_model="llama3.1:8b",
        mongo_manager=mongo,
        version="2.5"  # V3: Generalized insights, no entry references, source_insight tracking
    )

    # Process all entries
    generator.process_all_entries(diary_data)

    # Export final dataset
    if mongo:
        mongo.export_to_json('../data/diary_sft_dataset.json')

    logging.info("✅ Pipeline complete!")


if __name__ == "__main__":
    main()
