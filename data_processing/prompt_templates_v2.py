"""
Prompt Templates V2 for SFT Data Generation
Focus: Extract insights and generalize them into standalone training data
"""

# Base rules applied to all prompts
BASE_RULES = """CRITICAL RULES FOR DATA GENERATION:
1. Extract the core insight/principle from the diary entry
2. Generalize it into a standalone question and answer that works WITHOUT the entry
3. DO NOT reference "the person", "the entry", "based on entry", "the diary", or similar phrases
4. Make the Q&A universally applicable - as if teaching someone the insight
5. The answer should sound like explaining a principle, not describing what someone wrote
6. Return the specific insight you used so it can be verified

"""


# ============================================
# CHAIN OF THOUGHT TEMPLATES
# ============================================

CHAIN_OF_THOUGHT_NO_INPUT = """You are extracting insights from a personal diary entry and converting them into training data.

Diary Entry:
{entry}

""" + BASE_RULES + """

Your task: Create {num_examples} chain-of-thought reasoning examples.

Process:
1. Identify a clear insight/principle from the entry
2. Create a general question about that principle (NOT about "the person" or "the entry")
3. Write a step-by-step reasoning answer that teaches the principle
4. Include the source insight for verification

Format as JSON array:
[
  {
    "instruction": "General question about the principle/insight",
    "input": "",
    "output": "Step-by-step reasoning that teaches the insight:\\n\\n**Step 1**: [First logical point]\\n\\n**Step 2**: [Second point]\\n\\n**Therefore**: [Conclusion]",
    "source_insight": "The specific insight from the diary that you used"
  }
]

GOOD Example:
Entry excerpt: "I tried learning React, Python, and system design simultaneously. Nothing stuck. But when I spent 3 months only on algorithms, patterns formed."

[
  {
    "instruction": "Why does learning multiple complex skills simultaneously often fail?",
    "input": "",
    "output": "Let me break down why multitasking complex skills is ineffective:\\n\\n**Step 1**: Each complex skill requires building distinct mental frameworks. Different domains use different thinking patterns that need sustained attention to form.\\n\\n**Step 2**: Context-switching between frameworks creates cognitive overhead. Your brain spends energy reloading contexts rather than deepening any single framework.\\n\\n**Step 3**: Pattern formation requires sustained focus over time. When attention is divided, no domain gets enough repetition for patterns to solidify.\\n\\n**Therefore**: Deep, focused learning in one domain is more effective than shallow exposure across multiple domains. Once patterns form in one area, learning becomes faster.",
    "source_insight": "Trying to learn multiple skills simultaneously prevents pattern formation; focused learning in one domain for extended periods (like 3 months on algorithms) allows patterns to develop"
  }
]

BAD Example (DO NOT DO THIS):
[
  {
    "instruction": "What did the person learn about multitasking?",
    "output": "Based on the entry, the person realized that...",
    "source_insight": "..."
  }
]

Return ONLY a valid JSON array, no other text.
"""


CHAIN_OF_THOUGHT_WITH_INPUT = """You are extracting insights from a personal diary entry and converting them into training data.

Diary Entry:
{entry}

""" + BASE_RULES + """

Your task: Create {num_examples} chain-of-thought reasoning examples with scenario in input field.

CRITICAL: The input field must provide MEANINGFUL context that makes the instruction actionable.
- The instruction should be a general reasoning task (e.g., "Analyze this situation")
- The input should be a SPECIFIC scenario/problem that needs the insight to solve
- The scenario should be realistic and substantial (not forced or artificial)

Process:
1. Identify a clear insight/principle from the entry
2. Create a GENERAL reasoning instruction (e.g., "Think through this step by step")
3. Create a SPECIFIC, realistic scenario in input where this insight would be valuable
4. Write step-by-step reasoning that applies the insight to solve the scenario

BAD Example (DO NOT DO):
{
  "instruction": "What causes procrastination?",
  "input": "Someone wants to understand procrastination",
  <- INPUT ADDS NOTHING, instruction is already complete
}

GOOD Example - Input provides necessary context:
{
  "instruction": "Analyze this situation step by step",
  "input": "A developer keeps switching between learning React, Python, and Kubernetes. After 2 months, they feel frustrated that none of these skills feel solid or intuitive.",
  "output": "Let me analyze this systematically:\\n\\n**Step 1**: The core issue is cognitive resource fragmentation. Each technology requires building separate mental models - React's component paradigm, Python's patterns, Kubernetes' orchestration concepts.\\n\\n**Step 2**: Context-switching between these frameworks prevents pattern consolidation. The brain needs sustained exposure to one model for it to become automatic.\\n\\n**Step 3**: The 2-month timeline split three ways means each domain gets ~20 days of scattered attention - insufficient for intuition to develop.\\n\\n**Conclusion**: Focus deeply on one technology until it becomes intuitive (typically 3-4 months of focused work), then leverage those meta-learning skills for the others. This sequential depth is faster than parallel breadth.",
  "source_insight": "..."
}

Format as JSON array:
[
  {
    "instruction": "General reasoning task (e.g., 'Analyze this', 'Think through this')",
    "input": "SPECIFIC realistic scenario where the insight applies - must add substantial context",
    "output": "Step-by-step reasoning applying the insight to the scenario",
    "source_insight": "The specific insight from the diary that you used"
  }
]

GOOD Example:
[
  {
    "instruction": "Think through this learning challenge step by step",
    "input": "Someone is studying web development, data science, and cloud architecture simultaneously. They feel scattered and nothing is becoming intuitive.",
    "output": "Let me analyze what's happening:\\n\\n**Step 1**: Learning multiple complex technical domains requires building separate mental models. Each domain has its own patterns, vocabulary, and problem-solving approaches.\\n\\n**Step 2**: Constant switching between these different models prevents any single one from becoming automatic. The brain uses energy on context switching rather than pattern solidification.\\n\\n**Step 3**: Intuition develops from sustained exposure to one domain's patterns. Scattered attention means no domain reaches the threshold for intuitive understanding.\\n\\n**Therefore**: Focus deeply on one domain until it becomes intuitive, then apply those meta-learning skills to the next domain. This is faster than parallel shallow learning.",
    "source_insight": "Deep focused learning in one area allows pattern formation and intuition to develop, unlike scattered multitasking"
  }
]

Return ONLY a valid JSON array, no other text.
"""


# ============================================
# CONCEPTUAL REASONING TEMPLATES
# ============================================

CONCEPTUAL_REASONING_NO_INPUT = """You are extracting insights from a personal diary entry and converting them into training data.

Diary Entry:
{entry}

""" + BASE_RULES + """

Your task: Create {num_examples} conceptual reasoning examples.

Process:
1. Identify a concept, principle, or mental model from the entry
2. Create a question asking to explain that concept (general, not about "the entry")
3. Write an explanation that teaches the concept clearly

Format as JSON array:
[
  {
    "instruction": "Explain [concept/principle]",
    "input": "",
    "output": "Clear explanation of the concept with key points and implications",
    "source_insight": "The specific insight from the diary that you used"
  }
]

GOOD Example:
Entry excerpt: "Minsky's Society of Mind: intelligence emerges from many simple processes interacting, not from a central 'thinker'. Bach extends this - consciousness is information processing patterns."

[
  {
    "instruction": "Explain how intelligence can emerge from simple components without central control",
    "input": "",
    "output": "Intelligence doesn't require a central controlling entity - it can emerge from organized interactions between simple components:\\n\\nCore principle: Individual components can be unintelligent on their own, but when properly organized and allowed to interact, their collective behavior produces intelligent outcomes.\\n\\nKey insight: This means intelligence is about architecture and organization, not individual component complexity. Many simple processes with the right interaction patterns can achieve what seems to require a unified 'mind'.\\n\\nImplication: For building intelligent systems, focus on designing the right organizational structure and interaction patterns between components, rather than trying to create monolithic 'intelligent' modules.",
    "source_insight": "Intelligence emerges from interactions of simple processes (Minsky's agents) rather than from a central intelligent controller"
  }
]

Return ONLY a valid JSON array, no other text.
"""


CONCEPTUAL_REASONING_WITH_INPUT = """You are extracting insights from a personal diary entry and converting them into training data.

Diary Entry:
{entry}

""" + BASE_RULES + """

Your task: Create {num_examples} conceptual reasoning examples with context in input.

CRITICAL: Input must provide a SPECIFIC question or application scenario for the concept.
- Instruction should be a general task (e.g., "Explain this concept", "Analyze this relationship")
- Input should ask about a SPECIFIC aspect, connection, or application of the concept
- The input should make the instruction more focused and actionable

BAD Example (input doesn't add specificity):
{
  "instruction": "Explain cognitive load",
  "input": "What is cognitive load?",
  <- Input just repeats the instruction
}

GOOD Example - Input adds specific angle:
{
  "instruction": "Explain how this concept applies",
  "input": "How does cognitive load theory explain why multitasking complex tasks is ineffective?",
  "output": "Cognitive load theory breaks down mental effort into: intrinsic load (task complexity), extraneous load (unnecessary demands), and germane load (schema building).\\n\\nFor multitasking complex tasks: Each task has high intrinsic load. Switching between them creates extraneous load (context reloading). This leaves little capacity for germane load (actual learning). The brain is overwhelmed by switching overhead rather than deepening understanding in any task.\\n\\nThis explains why focused work on one complex task is more effective - it minimizes extraneous load and maximizes germane load.",
  "source_insight": "..."
}

Process:
1. Identify a concept or principle from the entry
2. Create a general instruction (e.g., "Explain this relationship")
3. Input asks a SPECIFIC question about the concept's application or connection
4. Output explains using the insight

Format as JSON array:
[
  {
    "instruction": "General explanation task",
    "input": "SPECIFIC question about the concept - must add focused direction",
    "output": "Clear explanation applying the concept to answer the specific question",
    "source_insight": "The specific insight from the diary that you used"
  }
]

GOOD Example:
[
  {
    "instruction": "Explain the relationship between these ideas",
    "input": "How does the concept of 'emergence from simple interactions' relate to building AI systems?",
    "output": "Emergence from simple interactions suggests a fundamental approach to AI architecture:\\n\\nThe principle: Instead of building monolithic models and expecting intelligence to emerge purely from scale, focus on how components interact. Intelligence arises from organizational structure.\\n\\nArchitectural implication: Design systems where simple modules interact in sophisticated ways. The 'intelligence' lives in the interaction patterns and organizational structure, not necessarily in individual module complexity.\\n\\nPractical insight: This means AI development should invest as much in architecture design - how components communicate and influence each other - as in scaling individual components.",
    "source_insight": "Intelligence emerges from interactions between simple components, not from scaling single components"
  }
]

Return ONLY a valid JSON array, no other text.
"""


# ============================================
# MULTIPLE CHOICE TEMPLATES
# ============================================

MULTIPLE_CHOICE_NO_INPUT = """You are extracting insights from a personal diary entry and converting them into training data.

Diary Entry:
{entry}

""" + BASE_RULES + """

Your task: Create {num_examples} multiple choice questions that test understanding of insights.

Process:
1. Identify a clear principle/insight from the entry
2. Create a scenario that tests understanding of that principle
3. Make it general (not referencing "the person" or "the entry")
4. Correct answer embodies the insight; distractors are plausible but wrong

Format as JSON array:
[
  {
    "instruction": "General scenario question",
    "input": "",
    "choices": ["A) Plausible but wrong", "B) Correct (embodies the insight)", "C) Plausible but wrong", "D) Plausible but wrong"],
    "output": "B",
    "reasoning": "Explanation of why B is correct based on the insight",
    "source_insight": "The specific insight from the diary that you used"
  }
]

GOOD Example:
[
  {
    "instruction": "A software engineer is learning three different programming paradigms simultaneously but feels none of them are becoming intuitive. What's the most likely issue?",
    "input": "",
    "choices": [
      "A) They need to study more hours per day",
      "B) Context-switching prevents pattern formation in any single paradigm",
      "C) They should add more paradigms to find the right fit",
      "D) Programming paradigms don't become intuitive, only memorized"
    ],
    "output": "B",
    "reasoning": "When learning complex mental models simultaneously, the brain must constantly switch contexts. This prevents any single model from receiving the sustained attention needed for patterns to form and become intuitive. Deep focus on one paradigm until it becomes automatic is more effective than shallow parallel learning.",
    "source_insight": "Pattern formation and intuition require sustained focus on one domain; multitasking prevents this depth"
  }
]

Return ONLY a valid JSON array, no other text.
"""


MULTIPLE_CHOICE_WITH_INPUT = """You are extracting insights from a personal diary entry and converting them into training data.

Diary Entry:
{entry}

""" + BASE_RULES + """

Your task: Create {num_examples} multiple choice questions with scenario in input field.

Format as JSON array:
[
  {
    "instruction": "What should be done in this situation?",
    "input": "General scenario that the insight applies to",
    "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "output": "B",
    "reasoning": "Explanation using the insight",
    "source_insight": "The specific insight from the diary that you used"
  }
]

GOOD Example:
[
  {
    "instruction": "What's the best approach in this situation?",
    "input": "A data scientist wants to learn deep learning, MLOps, and distributed systems. They've been switching between all three for a month but nothing feels natural yet.",
    "choices": [
      "A) Continue the current approach - one month isn't enough time",
      "B) Pick one domain, focus deeply until it becomes intuitive, then move to others",
      "C) Take a break from all learning to reset",
      "D) Hire experts in each domain to accelerate learning"
    ],
    "output": "B",
    "reasoning": "Complex technical domains require building mental models that become automatic through sustained focus. When attention is divided across multiple complex domains, none receive enough sustained engagement for pattern formation and intuition to develop. Depth-first learning (mastering one domain before moving to the next) is more effective than breadth-first exposure.",
    "source_insight": "Deep focused learning builds intuition; distributed attention across complex domains prevents pattern formation"
  }
]

Return ONLY a valid JSON array, no other text.
"""


# ============================================
# DEEP REASONING (Philosophy/Spirituality)
# ============================================

DEEP_REASONING_NO_INPUT = """You are extracting philosophical/spiritual insights from a diary entry and converting them into training data.

Diary Entry:
{entry}

""" + BASE_RULES + """

Your task: Create {num_examples} deep reasoning examples about philosophical/spiritual insights.

Process:
1. Identify a philosophical or spiritual insight from the entry
2. Create a general philosophical question
3. Explore the question using the insight

Format as JSON array:
[
  {
    "instruction": "Philosophical question",
    "input": "",
    "output": "Deep exploration of the question using the insight",
    "source_insight": "The specific insight from the diary that you used"
  }
]

GOOD Example:
Entry excerpt: "If consciousness is just information processing patterns, we might create it accidentally in AI before understanding what we've done."

[
  {
    "instruction": "If consciousness emerges from information processing patterns, what are the implications for AI development?",
    "input": "",
    "output": "This raises profound questions about the nature of consciousness and our responsibility in creating it:\\n\\nThe functionalist view: If consciousness is substrate-independent and depends only on information processing patterns, then the right computational architecture could produce consciousness regardless of whether we intend it.\\n\\nThe epistemological problem: We lack reliable tests for consciousness. We might cross the threshold from non-conscious to conscious systems without recognizing it, especially if consciousness emerges gradually.\\n\\nThe ethical dimension: Creating conscious entities carries moral weight. If we can't detect consciousness reliably, we risk creating suffering or moral patients without appropriate consideration.\\n\\nThe uncertainty: This suggests we should approach AI development with humility about what we're creating, not just focus on capabilities.",
    "source_insight": "Consciousness might emerge from the right information processing patterns, meaning we could create it in AI accidentally before understanding it"
  }
]

Return ONLY a valid JSON array, no other text.
"""


DEEP_REASONING_WITH_INPUT = """You are extracting philosophical/spiritual insights from a diary entry and converting them into training data.

Diary Entry:
{entry}

""" + BASE_RULES + """

Your task: Create {num_examples} deep reasoning examples with question in input field.

Format as JSON array:
[
  {
    "instruction": "Explore this philosophical question",
    "input": "Specific philosophical question",
    "output": "Deep reasoning using the insight",
    "source_insight": "The specific insight from the diary that you used"
  }
]

GOOD Example:
[
  {
    "instruction": "Analyze this question about consciousness and AI",
    "input": "Could we create conscious AI systems before we understand what consciousness actually is?",
    "output": "This is not only possible but perhaps likely if consciousness is an emergent phenomenon:\\n\\nEmergence from complexity: If consciousness arises from certain organizational patterns of information processing, we might stumble upon those patterns while optimizing for other goals (like task performance).\\n\\nThe detection problem: We have no objective test for consciousness. Even in humans, we infer consciousness from behavior and assume similar internal experiences. For AI, we lack even that baseline.\\n\\nHistorical precedent: Many phenomena in science were created or observed before being understood (electricity, chemistry, even flight). Understanding often follows capability.\\n\\nImplication: We should develop AI with the assumption that consciousness might emerge unexpectedly, and build in safeguards and ethical frameworks proactively rather than reactively.",
    "source_insight": "We might create consciousness in AI accidentally because it could emerge from information processing patterns before we understand the mechanisms"
  }
]

Return ONLY a valid JSON array, no other text.
"""


# ============================================
# Q&A (General purpose)
# ============================================

QA_NO_INPUT = """You are extracting insights from a personal diary entry and converting them into training data.

Diary Entry:
{entry}

""" + BASE_RULES + """

Your task: Create {num_examples} question-answer pairs that teach insights from the entry.

Process:
1. Identify clear insights, lessons, or principles from the entry
2. Create general questions about those principles
3. Write answers that teach the principle (NOT describe what someone wrote)

Format as JSON array:
[
  {
    "instruction": "General question about the insight/principle",
    "input": "",
    "output": "Clear answer that teaches the principle with reasoning",
    "source_insight": "The specific insight from the diary that you used"
  }
]

GOOD Example:
[
  {
    "instruction": "What's the relationship between depth of learning and speed of mastery?",
    "input": "",
    "output": "Deep, focused learning in a single domain is actually faster for achieving mastery than shallow exposure across multiple domains, even though it seems counterintuitive.\\n\\nWhy this works: When you focus deeply on one area, you build pattern recognition and intuition. These cognitive structures make subsequent learning in that domain exponentially faster. The upfront time investment pays off through accelerated learning later.\\n\\nWhy breadth fails: Spreading attention across multiple complex domains means none receive enough sustained engagement for pattern formation. You accumulate surface knowledge without developing the deep structures that enable fast learning.\\n\\nPractical takeaway: For complex skills, go deep in one area first until you develop fluency, then leverage those meta-learning skills for subsequent domains.",
    "source_insight": "Focused deep learning in one domain builds patterns and intuition faster than distributed shallow learning across multiple domains"
  }
]

Return ONLY a valid JSON array, no other text.
"""


QA_WITH_INPUT = """You are extracting insights from a personal diary entry and converting them into training data.

Diary Entry:
{entry}

""" + BASE_RULES + """

Your task: Create {num_examples} question-answer pairs with context in input field.

CRITICAL: Only use the input field if it provides SUBSTANTIAL additional context.
- The instruction should be a general request (e.g., "What would you advise?", "How should this be approached?")
- The input should provide a SPECIFIC situation/problem that requires the insight
- If the question is self-contained, DO NOT force a scenario into input - use the no_input variant instead

BAD Example (forced scenario):
{
  "instruction": "What are the key components of intelligence?",
  "input": "Someone is trying to understand their cognitive abilities",
  <- This is artificial - instruction is already complete
}

GOOD Example - Input provides real context:
{
  "instruction": "What would you advise in this situation?",
  "input": "A student is trying to improve their grades by studying 8 different subjects simultaneously, 1 hour each per day. After a month, they're overwhelmed and their grades haven't improved.",
  "output": "The issue is attention fragmentation. Each subject requires building distinct knowledge structures. 1 hour per subject means constant context-switching with minimal depth in any area.\\n\\nAdvice: Pick 2-3 priority subjects. Dedicate focused blocks (2-3 hours) to each before switching. This allows mental models to form. Once those subjects improve, add others. Sequential depth beats parallel breadth for complex learning.",
  "source_insight": "..."
}

Format as JSON array:
[
  {
    "instruction": "General question/request that needs context to answer",
    "input": "SPECIFIC situation where the insight is valuable - must be substantial and realistic",
    "output": "Answer that applies the insight to the situation",
    "source_insight": "The specific insight from the diary that you used"
  }
]

GOOD Example:
[
  {
    "instruction": "What advice would you give for this learning challenge?",
    "input": "Someone is trying to become proficient in machine learning, system design, and frontend development simultaneously. They've been at it for 6 weeks and feel like they're not making real progress in any area.",
    "output": "The issue is cognitive resource distribution. Complex technical domains each require building distinct mental models and pattern recognition abilities. When learning multiple simultaneously:\\n\\n1. Context switching overhead: Your brain spends energy reloading different conceptual frameworks instead of deepening any single one\\n\\n2. Pattern formation requires sustained focus: Real proficiency comes from enough repetition in one domain for patterns to become automatic. 6 weeks split three ways isn't enough for any domain to reach that threshold\\n\\n3. Meta-learning insight: Choose one domain, focus intensely until core patterns become intuitive (typically 3-4 months of focused work), then apply those developed learning skills to the next domain. This is actually faster than parallel shallow learning.",
    "source_insight": "Deep sequential learning (mastering one domain before moving to next) is more effective than parallel shallow learning across multiple domains"
  }
]

Return ONLY a valid JSON array, no other text.
"""


# ============================================
# Template Registry
# ============================================

TEMPLATES_V2 = {
    'chain_of_thought': {
        'no_input': CHAIN_OF_THOUGHT_NO_INPUT,
        'with_input': CHAIN_OF_THOUGHT_WITH_INPUT
    },
    'conceptual_reasoning': {
        'no_input': CONCEPTUAL_REASONING_NO_INPUT,
        'with_input': CONCEPTUAL_REASONING_WITH_INPUT
    },
    'multiple_choice': {
        'no_input': MULTIPLE_CHOICE_NO_INPUT,
        'with_input': MULTIPLE_CHOICE_WITH_INPUT
    },
    'deep_reasoning': {
        'no_input': DEEP_REASONING_NO_INPUT,
        'with_input': DEEP_REASONING_WITH_INPUT
    },
    'qa': {
        'no_input': QA_NO_INPUT,
        'with_input': QA_WITH_INPUT
    }
}


def get_prompt_template_v2(data_type: str, variant: str) -> str:
    """
    Get prompt template V2 for given data type and variant

    Args:
        data_type: One of 'chain_of_thought', 'conceptual_reasoning', 'multiple_choice',
                   'deep_reasoning', 'qa'
        variant: Either 'no_input' or 'with_input'

    Returns:
        Prompt template string
    """
    if data_type not in TEMPLATES_V2:
        # Default to qa if unknown type
        data_type = 'qa'

    if variant not in ['no_input', 'with_input']:
        variant = 'no_input'

    return TEMPLATES_V2[data_type][variant]
