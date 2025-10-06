"""
Post-generation quality analysis for SFT training data
Uses LLM to grade examples and analyze dataset coverage
"""

import sys
sys.path.append('..')
from mongodb_connection import DiaryDataManager
import json
import requests
from collections import Counter, defaultdict
import random
from typing import List, Dict

class QualityAnalyzer:
    """Analyze quality and coverage of generated training data"""

    def __init__(self, version="2.5", ollama_model="llama3.1:8b"):
        self.mongo = DiaryDataManager()
        self.version = version
        self.model = ollama_model

    def ollama_inference(self, prompt: str, temperature: float = 0.3) -> str:
        """Get LLM judgment on example quality"""
        try:
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": 500
                }
            }
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()['response']
        except Exception as e:
            print(f"Error in LLM inference: {e}")
            return None

    def grade_example(self, example: Dict) -> Dict:
        """
        Grade a single example using LLM
        Returns: {
            'grade': 'GOOD' | 'MEDIOCRE' | 'BAD',
            'issues': List[str],
            'score': float (0-1)
        }
        """

        grading_prompt = f"""Grade this training data example on a scale of GOOD/MEDIOCRE/BAD.

Example:
Instruction: {example['instruction']}
Input: {example.get('input', '')}
Output: {example['output'][:500]}...
Source Insight: {example['metadata'].get('source_insight', 'N/A')}

Grading Criteria:

BAD (reject):
- References specific people, books, podcasts, or interviews
- Contains "the person", "the entry", "based on entry", "according to X"
- Hallucinated facts or appeals to external authority
- Instruction and output don't match

MEDIOCRE (acceptable but not ideal):
- Generic advice that could come from anywhere
- Weak insight extraction
- Missing depth or reasoning structure
- Forced/artificial scenarios in input field

GOOD (high quality):
- Generalizes insight into universal principle
- No references to sources or specific entities
- Clear, structured reasoning
- Input field adds meaningful context (if used)

Respond in this EXACT format:
GRADE: [GOOD/MEDIOCRE/BAD]
ISSUES: [list issues if any, or "none"]
SCORE: [0.0-1.0]
"""

        response = self.ollama_inference(grading_prompt, temperature=0.1)

        if not response:
            return {'grade': 'UNKNOWN', 'issues': ['LLM grading failed'], 'score': 0.5}

        # Parse response
        grade = 'UNKNOWN'
        issues = []
        score = 0.5

        lines = response.strip().split('\n')
        for line in lines:
            if line.startswith('GRADE:'):
                grade = line.split('GRADE:')[1].strip()
            elif line.startswith('ISSUES:'):
                issues_text = line.split('ISSUES:')[1].strip()
                if issues_text.lower() != 'none':
                    issues = [issues_text]
            elif line.startswith('SCORE:'):
                try:
                    score = float(line.split('SCORE:')[1].strip())
                except:
                    score = 0.5

        return {
            'grade': grade,
            'issues': issues,
            'score': score
        }

    def analyze_coverage(self):
        """Analyze topic and data type coverage"""

        examples = list(self.mongo.collection.find({
            "metadata.version": self.version
        }))

        print(f"\n{'='*60}")
        print(f"COVERAGE ANALYSIS FOR VERSION {self.version}")
        print(f"{'='*60}")

        print(f"\n📊 Total examples: {len(examples)}")

        # Category distribution
        category_dist = Counter([ex['metadata']['category'] for ex in examples])
        print(f"\n📂 By Category:")
        for cat, count in category_dist.most_common():
            pct = (count / len(examples)) * 100
            print(f"  {cat}: {count} ({pct:.1f}%)")

        # Data type distribution
        type_dist = Counter([ex['metadata']['data_type'] for ex in examples])
        print(f"\n📝 By Data Type:")
        for dtype, count in type_dist.most_common():
            pct = (count / len(examples)) * 100
            print(f"  {dtype}: {count} ({pct:.1f}%)")

        # Variant distribution
        variant_dist = Counter([ex['metadata']['variant'] for ex in examples])
        print(f"\n🔀 By Variant:")
        for var, count in variant_dist.most_common():
            pct = (count / len(examples)) * 100
            print(f"  {var}: {count} ({pct:.1f}%)")

        # Reasoning type distribution
        reasoning_dist = Counter([ex['metadata']['reasoning_type'] for ex in examples])
        print(f"\n🧠 By Reasoning Type:")
        for rtype, count in reasoning_dist.most_common():
            pct = (count / len(examples)) * 100
            print(f"  {rtype}: {count} ({pct:.1f}%)")

        return examples

    def sample_quality_check(self, sample_size=50):
        """
        Grade a random sample of examples using LLM
        """

        examples = list(self.mongo.collection.find({
            "metadata.version": self.version
        }))

        if len(examples) == 0:
            print("No examples found for this version")
            return

        # Random sample
        sample = random.sample(examples, min(sample_size, len(examples)))

        print(f"\n{'='*60}")
        print(f"QUALITY SAMPLE CHECK ({len(sample)} examples)")
        print(f"{'='*60}")

        grades = []
        issues_by_category = defaultdict(list)

        for i, example in enumerate(sample):
            print(f"\nGrading {i+1}/{len(sample)}...", end=' ')

            grade_result = self.grade_example(example)
            grades.append(grade_result)

            print(f"{grade_result['grade']} (score: {grade_result['score']:.2f})")

            if grade_result['issues']:
                category = example['metadata']['category']
                issues_by_category[category].extend(grade_result['issues'])

        # Summary
        grade_dist = Counter([g['grade'] for g in grades])
        avg_score = sum(g['score'] for g in grades) / len(grades)

        print(f"\n{'='*60}")
        print(f"QUALITY SUMMARY")
        print(f"{'='*60}")

        print(f"\n📊 Grade Distribution:")
        for grade, count in grade_dist.most_common():
            pct = (count / len(sample)) * 100
            print(f"  {grade}: {count} ({pct:.1f}%)")

        print(f"\n⭐ Average Score: {avg_score:.2f}/1.0")

        if issues_by_category:
            print(f"\n⚠️  Common Issues by Category:")
            for cat, issues in issues_by_category.items():
                print(f"\n  {cat}:")
                issue_counts = Counter(issues)
                for issue, count in issue_counts.most_common(3):
                    print(f"    - {issue}: {count} times")

        # Estimate total bad examples
        bad_pct = (grade_dist.get('BAD', 0) / len(sample)) * 100
        total_bad_estimate = int((bad_pct / 100) * len(examples))

        print(f"\n🚨 Estimated BAD examples in full dataset: ~{total_bad_estimate} ({bad_pct:.1f}%)")

        return grades, issues_by_category

    def find_problematic_patterns(self):
        """Find examples with known problematic patterns"""

        examples = list(self.mongo.collection.find({
            "metadata.version": self.version
        }))

        problematic_patterns = {
            'references_people': [],
            'references_entry': [],
            'too_short_output': [],
            'generic_advice': [],
        }

        for ex in examples:
            text = (ex['instruction'] + ' ' + ex['output']).lower()

            # Check for people references (common names patterns)
            people_indicators = [' said', ' mentioned', ' interview', ' podcast', ' book by', ' author', "'s idea"]
            if any(indicator in text for indicator in people_indicators):
                problematic_patterns['references_people'].append(ex)

            # Check for entry references
            entry_refs = ['the person', 'the entry', 'based on entry', 'the diary']
            if any(ref in text for ref in entry_refs):
                problematic_patterns['references_entry'].append(ex)

            # Check for too short outputs
            if len(ex['output'].split()) < 30:
                problematic_patterns['too_short_output'].append(ex)

            # Check for generic phrases
            generic = ['in general', 'typically', 'usually', 'in most cases', 'generally speaking']
            if sum(1 for phrase in generic if phrase in text) >= 2:
                problematic_patterns['generic_advice'].append(ex)

        print(f"\n{'='*60}")
        print(f"PROBLEMATIC PATTERNS DETECTED")
        print(f"{'='*60}")

        total_flagged = 0
        for pattern, examples_list in problematic_patterns.items():
            if examples_list:
                count = len(examples_list)
                total_flagged += count
                pct = (count / len(examples)) * 100
                print(f"\n⚠️  {pattern.replace('_', ' ').title()}: {count} ({pct:.1f}%)")

                # Show sample
                if count > 0:
                    sample = examples_list[0]
                    print(f"    Sample: {sample['instruction'][:80]}...")

        print(f"\n🔢 Total flagged (may overlap): {total_flagged}")
        print(f"⚠️  Estimated unique problematic examples: ~{len(set(ex['_id'] for pattern_list in problematic_patterns.values() for ex in pattern_list))}")

        return problematic_patterns

    def full_report(self):
        """Run complete quality analysis"""

        print(f"\n{'#'*60}")
        print(f"# FULL QUALITY ANALYSIS REPORT - VERSION {self.version}")
        print(f"{'#'*60}")

        # Coverage analysis
        examples = self.analyze_coverage()

        # Pattern detection
        problematic = self.find_problematic_patterns()

        # Sample quality check
        grades, issues = self.sample_quality_check(sample_size=50)

        # Recommendations
        print(f"\n{'='*60}")
        print(f"RECOMMENDATIONS")
        print(f"{'='*60}")

        bad_count = sum(1 for g in grades if g['grade'] == 'BAD')
        bad_pct = (bad_count / len(grades)) * 100

        if bad_pct > 15:
            print(f"\n🚨 HIGH: {bad_pct:.1f}% BAD examples - consider regenerating with stricter prompts")
        elif bad_pct > 5:
            print(f"\n⚠️  MODERATE: {bad_pct:.1f}% BAD examples - acceptable but could improve")
        else:
            print(f"\n✅ LOW: {bad_pct:.1f}% BAD examples - quality is good")

        mediocre_count = sum(1 for g in grades if g['grade'] == 'MEDIOCRE')
        mediocre_pct = (mediocre_count / len(grades)) * 100

        if mediocre_pct > 40:
            print(f"📊 {mediocre_pct:.1f}% MEDIOCRE - insights are getting diluted, may need better extraction")

        print(f"\n💡 Consider:")
        print(f"  - Filtering out examples with pattern flags")
        print(f"  - Focus on categories with lower quality scores")
        print(f"  - Review source_insight quality in problematic examples")


def main():
    analyzer = QualityAnalyzer(version="2.5")
    analyzer.full_report()


if __name__ == "__main__":
    main()
