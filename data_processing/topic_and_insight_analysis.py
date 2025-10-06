import json
from pprint import pprint
from collections import Counter
import sys
sys.path.append('..')

def analyze_topics_and_save():
    """Analyze topics and save statistics for taxonomy building"""
    topics_files = "../data/diary_topics.json"

    with open(topics_files, "r") as f:
        topics_data = json.load(f)

    primary_topics = []
    all_topics = []
    all_insights = set()

    for entry in topics_data:
        coarse_topics = entry.get("topics", [])

        if coarse_topics:
            for topic_entry, topic_list in coarse_topics.items():
                if "primary" in topic_entry.lower():
                    primary_topics.extend(topic_list)
                if "primary" in topic_entry.lower() or "secondary" in topic_entry.lower():
                    all_topics.extend(topic_list)
                if "insights" in topic_entry.lower():
                    all_insights.update(topic_list)

    print(f"Extracted {len(set(primary_topics))} unique primary topics")
    print(f"Extracted {len(set(all_topics))} unique all topics")
    print(f"Extracted {len(all_insights)} unique key insights")

    # Create counters
    primary_counter = Counter(primary_topics)
    all_counter = Counter(all_topics)

    # Save to files (for taxonomy to load)
    output = {
        'primary_topics_counter': dict(primary_counter),
        'all_topics_counter': dict(all_counter),
        'unique_insights': list(all_insights)
    }

    with open('../data/topic_statistics.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\n💾 Saved topic statistics to data/topic_statistics.json")
    print(f"\nTop 10 Primary Topics:")
    pprint(primary_counter.most_common(10))

    return primary_counter, all_counter

def main():
    analyze_topics_and_save()

if __name__ == "__main__":
    main()