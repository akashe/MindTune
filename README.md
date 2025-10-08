# Personal Reasoning Model: Fine-tuning on Diary Insights

## Overview

This project explores whether fine-tuning language models on personal diary entries can improve their reasoning capabilities and human understanding. Rather than training on generic knowledge, we extract insights from personal experiences and convert them into diverse training data formats.

## Research Question

**Can models learn to reason better and understand humans more deeply by training on real personal insights rather than synthetic/generic data?**

## Hypothesis

Training on personal diary insights will improve:
1. **Multi-step reasoning** (chain-of-thought, problem decomposition)
2. **Social & emotional intelligence** (understanding relationships, emotions)
3. **Moral reasoning** (ethical judgment across multiple frameworks)
4. **Domain knowledge** (philosophy, psychology, meta-cognition)

While maintaining general capabilities and accepting trade-offs in areas not covered by the training data (e.g., math, factual trivia).

## Approach

### 1. Data Pipeline (`data_processing/`)

**Topic Extraction → Analysis → Taxonomy → Training Data Generation**

```
┌─────────────────┐
│  Diary Entries  │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ Topic Extraction    │  Extract topics and insights
│ (topic_extraction)  │  from diary entries
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Topic Analysis      │  Analyze topic frequency
│ (topic_and_insight) │  and co-occurrence
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Taxonomy Building   │  Organize into main
│ (topic_taxonomy)    │  categories
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Training Data Gen   │  Generate diverse formats:
│ (generate_sft_data) │  • QA (62.6%)
│                     │  • Chain-of-Thought (29.1%)
│                     │  • Conceptual (7.0%)
│                     │  • Deep Reasoning (1.0%)
│                     │  • Multiple Choice (0.3%)
└─────────────────────┘
```

### 2. Data Categories (from `topic_taxonomy.json`)

| Category | Samples | Description |
|----------|---------|-------------|
| **emotional_intelligence** | ~25% | Relationships, emotions, mental health |
| **personal_growth** | ~20% | Productivity, learning, self-improvement |
| **meta_thinking** | ~17% | Thinking about thinking, learning strategies |
| **spirituality** | ~15% | Philosophy, consciousness, existential questions |
| **ai_technical** | ~13% | AI/ML concepts, AGI implications |
| **creativity** | ~10% | Creative processes, artistic thinking |

### 3. Training Data Diversity

We deliberately use **multiple formats** to teach different reasoning modes:

| Format | % | Purpose | Explicit Steps? |
|--------|---|---------|-----------------|
| **Question-Answer** | 62.6% | Direct knowledge transfer | ❌ Natural prose |
| **Chain-of-Thought** | 29.1% | Step-by-step reasoning | ✅ "Step 1, Step 2, Therefore" |
| **Conceptual Reasoning** | 7.0% | Structured concept explanation | ⚠️ Structured but natural |
| **Deep Reasoning** | 1.0% | Philosophical exploration | ❌ Narrative |
| **Multiple Choice** | 0.3% | Scenario-based understanding | ❌ MCQ format |

**Key Design Decision:** Only 29.1% uses explicit "Step 1, Step 2" formatting to avoid overfitting to procedural reasoning while still teaching systematic thinking.





