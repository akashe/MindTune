"""
Topic Taxonomy and Normalization
Clusters and normalizes extracted topics into clean categories
"""

import json
from collections import Counter
from typing import Dict, List, Set


class TopicTaxonomy:
    """Normalize and cluster topics into categories"""

    def __init__(self, category_mappings_path='../data/category_mappings.json'):
        # Load main categories from external file
        try:
            with open(category_mappings_path, 'r') as f:
                config = json.load(f)
                self.MAIN_CATEGORIES = config['main_categories']
        except FileNotFoundError:
            print(f"⚠️  Warning: {category_mappings_path} not found. Using empty categories.")
            self.MAIN_CATEGORIES = {}

        # Store normalized mapping
        self.topic_to_category = {}
        self.normalized_topics = {}

    def normalize_topic(self, topic: str) -> str:
        """Normalize topic by lowercasing and removing special chars"""
        return topic.lower().strip().replace('_', ' ')

    def map_topic_to_category(self, topic: str) -> str:
        """Map a topic to its main category"""

        normalized = self.normalize_topic(topic)

        # Check if already mapped
        if normalized in self.topic_to_category:
            return self.topic_to_category[normalized]

        # Try to match against main categories
        for category, data in self.MAIN_CATEGORIES.items():
            keywords = [self.normalize_topic(kw) for kw in data['keywords']]

            # Exact match
            if normalized in keywords:
                self.topic_to_category[normalized] = category
                return category

            # Fuzzy match (contains)
            for keyword in keywords:
                if keyword in normalized or normalized in keyword:
                    self.topic_to_category[normalized] = category
                    return category

        # If no match, create a specific category for it
        # This ensures we don't lose data from unique topics
        self.topic_to_category[normalized] = f"specific_{normalized.replace(' ', '_')}"
        return self.topic_to_category[normalized]

    def build_taxonomy_from_counter(self, topic_counter: Counter) -> Dict:
        """
        Build taxonomy from topic counter
        Returns mapping and statistics
        """

        category_counts = Counter()
        category_to_topics = {}

        for topic, count in topic_counter.items():
            category = self.map_topic_to_category(topic)

            category_counts[category] += count

            if category not in category_to_topics:
                category_to_topics[category] = []
            category_to_topics[category].append({
                'original': topic,
                'normalized': self.normalize_topic(topic),
                'count': count
            })

        # Build final taxonomy
        taxonomy = {
            'main_categories': {},
            'specific_categories': {},
            'statistics': {
                'total_topics': len(topic_counter),
                'total_occurrences': sum(topic_counter.values()),
                'main_category_count': 0,
                'specific_category_count': 0
            }
        }

        for category, topics in category_to_topics.items():
            category_data = {
                'topics': topics,
                'total_count': category_counts[category],
                'description': self.MAIN_CATEGORIES.get(category, {}).get('description', 'Specific topic')
            }

            if category.startswith('specific_'):
                taxonomy['specific_categories'][category] = category_data
                taxonomy['statistics']['specific_category_count'] += 1
            else:
                taxonomy['main_categories'][category] = category_data
                taxonomy['statistics']['main_category_count'] += 1

        return taxonomy

    def get_category_for_topics(self, topics: List[str]) -> str:
        """
        Given a list of topics from an entry, determine the primary category
        """

        # Map each topic to category and count
        category_votes = Counter()

        for topic in topics:
            category = self.map_topic_to_category(topic)

            # Main categories get more weight
            if not category.startswith('specific_'):
                category_votes[category] += 2
            else:
                category_votes[category] += 1

        if len(category_votes) == 0:
            return 'miscellaneous'

        # Return most common category
        return category_votes.most_common(1)[0][0]

    def save_taxonomy(self, taxonomy: Dict, filepath: str = '../data/topic_taxonomy.json'):
        """Save taxonomy to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(taxonomy, f, indent=2)
        print(f"💾 Saved taxonomy to {filepath}")

    def print_taxonomy_summary(self, taxonomy: Dict):
        """Print human-readable summary"""

        print("\n" + "="*60)
        print("TOPIC TAXONOMY SUMMARY")
        print("="*60)

        print(f"\n📊 Statistics:")
        print(f"  Total unique topics: {taxonomy['statistics']['total_topics']}")
        print(f"  Total occurrences: {taxonomy['statistics']['total_occurrences']}")
        print(f"  Main categories: {taxonomy['statistics']['main_category_count']}")
        print(f"  Specific categories: {taxonomy['statistics']['specific_category_count']}")

        print(f"\n🏷️  Main Categories:")
        for category, data in sorted(taxonomy['main_categories'].items(),
                                     key=lambda x: x[1]['total_count'],
                                     reverse=True):
            print(f"\n  {category.upper()}: {data['total_count']} occurrences")
            print(f"    {data['description']}")
            print(f"    Top topics: {', '.join([t['original'] for t in data['topics'][:5]])}")

        print(f"\n📌 Specific Categories (top 10):")
        specific_sorted = sorted(taxonomy['specific_categories'].items(),
                                key=lambda x: x[1]['total_count'],
                                reverse=True)[:10]
        for category, data in specific_sorted:
            topic_name = category.replace('specific_', '').replace('_', ' ')
            print(f"  {topic_name}: {data['total_count']} occurrences")


def main():
    """Test taxonomy builder by loading topic statistics from file"""

    # Load topic counts from generated statistics file
    try:
        with open('../data/topic_statistics.json', 'r') as f:
            stats = json.load(f)
            topic_counts = Counter(stats['all_topics_counter'])
            print(f"📊 Loaded {len(topic_counts)} topics from topic_statistics.json")
    except FileNotFoundError:
        print("⚠️  Run topic_and_insight_analysis.py first to generate topic_statistics.json")
        return

    # Build taxonomy
    taxonomy_builder = TopicTaxonomy()
    taxonomy = taxonomy_builder.build_taxonomy_from_counter(topic_counts)

    # Print summary
    taxonomy_builder.print_taxonomy_summary(taxonomy)

    # Save to file
    taxonomy_builder.save_taxonomy(taxonomy)

    # Test category mapping
    print("\n" + "="*60)
    print("TEST: Category mapping for sample topics")
    print("="*60)
    test_topics = [
        ['meta_thinking', 'learning'],
        ['ai_technical', 'AGI'],
        ['relationships', 'emotional intelligence'],
        ['job search', 'EPFO tasks']
    ]
    for topics in test_topics:
        category = taxonomy_builder.get_category_for_topics(topics)
        print(f"  {topics} → {category}")


if __name__ == "__main__":
    main()
