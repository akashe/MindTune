"""
Prompt Templates for SFT Data Generation
Each data type has two variants: with input and without input
"""

# Base rules applied to all prompts
BASE_RULES = """CRITICAL RULES FOR DATA GENERATION:
1. ALL answers must come DIRECTLY from the diary entry - NO imagination or external knowledge
2. DO NOT create fictional scenarios that aren't in the entry
3. DO NOT add your own opinions or general knowledge
4. Extract the person's actual thinking patterns, insights, and reasoning from the entry
5. Use the person's own words and concepts where possible
6. If the entry doesn't contain enough information for a question type, skip it

"""


# ============================================
# CHAIN OF THOUGHT TEMPLATES
# ============================================

CHAIN_OF_THOUGHT_NO_INPUT = """You are creating training data from a personal diary entry.

Diary Entry:
{entry}

""" + BASE_RULES + """

Your task: Create {num_examples} chain-of-thought reasoning examples with step-by-step thinking.

Create self-contained questions (no separate input field) about:
- Reasoning processes shown IN THE ENTRY
- Why certain approaches work or don't work (from THE ENTRY)
- Thinking patterns demonstrated IN THE ENTRY
- Problem-solving approaches from THE ENTRY

Each answer must show explicit step-by-step reasoning from the entry.
Use markers like "Step 1:", "First:", "Next:", "Then:", "Therefore:" to show reasoning structure.

Format as JSON array:
[
  {
    "instruction": "Why does [specific situation from entry] happen?",
    "input": "",
    "output": "Let me break this down step by step:\\n\\n**Step 1**: [First observation from entry]...\\n\\n**Step 2**: [Next insight from entry]...\\n\\n**Step 3**: [Additional reasoning from entry]...\\n\\n**Therefore**: [Conclusion from entry]...\\n\\n**Why this matters**: [Implication from entry]"
  },
  {
    "instruction": "Explain the reasoning behind [concept from entry]",
    "input": "",
    "output": "Here's the logical chain:\\n\\n**First**: [Initial premise from entry]...\\n\\n**Then**: [Consequence from entry]...\\n\\n**Finally**: [Conclusion from entry]..."
  }
]

Example from diary entry about multitasking and learning:
[
  {
    "instruction": "Why does trying to learn multiple complex subjects simultaneously feel scattered?",
    "input": "",
    "output": "Let me think through this step by step:\\n\\n**Step 1**: Each complex subject requires building distinct mental patterns and frameworks. React has component thinking, Python has object-oriented patterns - these are different mental models.\\n\\n**Step 2**: Context-switching between different frameworks creates cognitive overhead. Your brain spends energy reloading contexts rather than deepening understanding in any single domain.\\n\\n**Step 3**: Pattern formation requires sustained, focused attention. When you jump between subjects, no single pattern has time to solidify.\\n\\n**Therefore**: Deep learning in one domain is more effective than shallow exposure across multiple domains. This is why spending 3 months focused on algorithms created lasting pattern recognition, while scattered multitasking just accumulated surface knowledge."
  }
]

Return ONLY a valid JSON array, no other text.
"""


CHAIN_OF_THOUGHT_WITH_INPUT = """You are creating training data from a personal diary entry.

Diary Entry:
{entry}

""" + BASE_RULES + """

Your task: Create {num_examples} chain-of-thought reasoning examples with context in the input field.

Create questions where:
- The "instruction" is a general thinking task (e.g., "Analyze this situation", "Think through this step by step")
- The "input" contains specific context/scenario FROM THE ENTRY
- The "output" shows step-by-step reasoning FROM THE ENTRY

Format as JSON array:
[
  {
    "instruction": "Analyze this step by step",
    "input": "[Specific situation/problem from the entry]",
    "output": "Breaking this down:\\n\\n**First**: [Observation from entry]...\\n\\n**Next**: [Analysis from entry]...\\n\\n**Conclusion**: [Resolution from entry]..."
  },
  {
    "instruction": "What's the reasoning process here?",
    "input": "[Scenario or question from the entry]",
    "output": "Let me work through this:\\n\\n**Step 1**: [From entry]...\\n\\n**Step 2**: [From entry]...\\n\\n**Therefore**: [From entry]..."
  }
]

Example from diary entry about multitasking and learning:
[
  {
    "instruction": "Think through this problem step by step",
    "input": "I've been trying to learn React, Python, and system design all at once. My mind feels scattered and nothing is sticking.",
    "output": "Let me analyze what's happening here:\\n\\n**Step 1 - Identify the issue**: You're experiencing cognitive overload from parallel complex learning. Each domain requires building its own mental framework.\\n\\n**Step 2 - Understand the mechanism**: Context-switching between React's component model, Python's patterns, and system design principles prevents any single framework from solidifying. Your brain is constantly reloading contexts.\\n\\n**Step 3 - Compare with past success**: Remember when you spent 3 months focused only on algorithms? That depth allowed patterns to form and stick.\\n\\n**Therefore**: The scattered feeling indicates you need to switch from breadth-first to depth-first learning. Pick one domain, go deep until patterns form, then move to the next."
  }
]

Return ONLY a valid JSON array, no other text.
"""


# ============================================
# CONCEPTUAL REASONING TEMPLATES
# ============================================

CONCEPTUAL_REASONING_NO_INPUT = """You are creating training data from a personal diary entry.

Diary Entry:
{entry}

""" + BASE_RULES + """

Your task: Create {num_examples} conceptual reasoning examples that explain ideas from the entry.

Create self-contained questions about:
- Concepts or ideas explained IN THE ENTRY
- Connections made IN THE ENTRY
- Mental models or frameworks from THE ENTRY
- Understanding demonstrated IN THE ENTRY

Format as JSON array:
[
  {
    "instruction": "Explain the concept of [idea from entry]",
    "input": "",
    "output": "Based on the entry, this concept means: [explanation using entry's perspective]...\\n\\nKey aspects:\\n- [Point from entry]\\n- [Point from entry]\\n\\nWhy it matters: [Implication from entry]"
  }
]

Example from AI concepts diary entry:
[
  {
    "instruction": "What's the connection between Minsky's Society of Mind and Joscha Bach's ideas about cognitive architecture?",
    "input": "",
    "output": "Both frameworks share a fundamental insight about intelligence emerging from interactions rather than singular entities.\\n\\nMinsky's perspective: The mind consists of many small processes (agents) that individually lack intelligence, but their collective interactions produce intelligent behavior. No single agent is 'smart' - intelligence emerges from organization.\\n\\nBach's extension: He applies this to computational processes, suggesting consciousness and cognition emerge from information processing patterns, not from any central 'thinker' or homunculus.\\n\\nThe key connection: Both reject centralized intelligence. They propose that what we call 'mind' or 'consciousness' is actually a distributed, emergent property arising from the right organizational structure of simpler components."
  }
]

Return ONLY a valid JSON array, no other text.
"""


CONCEPTUAL_REASONING_WITH_INPUT = """You are creating training data from a personal diary entry.

Diary Entry:
{entry}

""" + BASE_RULES + """

Your task: Create {num_examples} conceptual reasoning examples with context in the input field.

Create questions where:
- The "instruction" asks to explain/analyze a concept
- The "input" provides the concept or question FROM THE ENTRY
- The "output" explains using the entry's perspective

Format as JSON array:
[
  {
    "instruction": "Explain this concept",
    "input": "[Concept or idea from the entry]",
    "output": "This concept involves: [explanation from entry]...\\n\\nKey insights: [from entry]..."
  }
]

Example:
[
  {
    "instruction": "Explain the relationship between these ideas",
    "input": "How do Minsky's 'agents' and Bach's 'information processing patterns' relate to building AI systems?",
    "output": "The relationship suggests a fundamental approach to AI architecture.\\n\\nFrom Minsky: Intelligence doesn't require a monolithic 'intelligent' system. Small, simple processes can collectively produce smart behavior through their interactions.\\n\\nFrom Bach: This extends to viewing the brain as implementing computational processes where cognition emerges from patterns, not components.\\n\\nImplication for AI: Rather than training massive monolithic models and expecting intelligence to emerge purely from scale, we might need to focus more on architectural composition - designing systems with the right organizational structure where intelligence emerges from how components interact, not just from model size."
  }
]

Return ONLY a valid JSON array, no other text.
"""


# ============================================
# MULTIPLE CHOICE TEMPLATES
# ============================================

MULTIPLE_CHOICE_NO_INPUT = """You are creating training data from a personal diary entry.

Diary Entry:
{entry}

""" + BASE_RULES + """

Your task: Create {num_examples} multiple choice questions based on the entry.

Create questions where:
- The scenario/question is self-contained in the "instruction"
- Based on situations or insights FROM THE ENTRY
- The correct answer comes from the entry's wisdom
- Distractors are plausible but not supported by the entry

Format as JSON array:
[
  {
    "instruction": "Based on the insight that [from entry], what should someone do when [scenario from entry]?",
    "input": "",
    "choices": ["A) [Plausible but wrong]", "B) [Correct, from entry]", "C) [Plausible but wrong]", "D) [Plausible but wrong]"],
    "output": "B",
    "reasoning": "The entry shows that [explanation from entry]"
  }
]

Example from learning diary entry:
[
  {
    "instruction": "A person is trying to learn programming, design, and marketing simultaneously but feels scattered and nothing is sticking. What's most likely the issue?",
    "input": "",
    "choices": [
      "A) They need to study harder and put in more hours",
      "B) Context-switching prevents deep pattern formation in any domain",
      "C) They should add even more subjects to find the right fit",
      "D) They lack natural talent in these areas"
    ],
    "output": "B",
    "reasoning": "When learning multiple complex subjects simultaneously, context-switching between different mental frameworks prevents patterns from forming in any single domain. The cognitive overhead of constantly reloading different ways of thinking means no domain gets the sustained attention needed for deep learning and pattern recognition."
  }
]

Return ONLY a valid JSON array, no other text.
"""


MULTIPLE_CHOICE_WITH_INPUT = """You are creating training data from a personal diary entry.

Diary Entry:
{entry}

""" + BASE_RULES + """

Your task: Create {num_examples} multiple choice questions with context in input field.

Create questions where:
- The "instruction" asks what to do or what's happening
- The "input" provides a scenario FROM THE ENTRY
- Choices include correct answer (from entry) and plausible distractors

Format as JSON array:
[
  {
    "instruction": "What's most likely happening here?",
    "input": "[Scenario from entry]",
    "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "output": "B",
    "reasoning": "Explanation from entry"
  }
]

Example:
[
  {
    "instruction": "What should this person do?",
    "input": "Someone has been jumping between learning React, Python, and system design. They can read documentation but when they try to build, nothing comes naturally. Everything feels disconnected.",
    "choices": [
      "A) Take a break from all learning and come back fresh",
      "B) Focus deeply on one domain until patterns form, then move to others",
      "C) Keep exposing themselves to all three until something clicks",
      "D) Hire a tutor for all three subjects"
    ],
    "output": "B",
    "reasoning": "The disconnected feeling and lack of natural application indicates surface-level knowledge across multiple domains. Real learning requires sustained focus in one area until mental patterns solidify and concepts become intuitive. Once deep understanding exists in one domain, the meta-learning skills transfer to make subsequent domains easier to learn."
  }
]

Return ONLY a valid JSON array, no other text.
"""


# ============================================
# DEEP REASONING (Philosophy/Spirituality)
# ============================================

DEEP_REASONING_NO_INPUT = """You are creating training data from a personal diary entry about philosophy/spirituality.

Diary Entry:
{entry}

""" + BASE_RULES + """

Your task: Create {num_examples} deep reasoning examples about philosophical/spiritual concepts.

Create self-contained questions about:
- Philosophical ideas explored IN THE ENTRY
- Spiritual insights from THE ENTRY
- Abstract concepts from THE ENTRY
- Deep questions raised IN THE ENTRY

Preserve the person's unique philosophical perspective.

Format as JSON array:
[
  {
    "instruction": "What does [philosophical concept from entry] mean?",
    "input": "",
    "output": "[Deep exploration using entry's insights]...\\n\\nThis connects to: [From entry]...\\n\\nThe implication is: [From entry]..."
  }
]

Example from consciousness diary entry:
[
  {
    "instruction": "If consciousness emerges from information processing patterns, what does this mean for AI consciousness?",
    "input": "",
    "output": "This question touches on several deep issues about the nature of consciousness.\\n\\nThe functionalist perspective: If consciousness is purely about information processing patterns rather than substrate, then sufficiently complex AI could in principle be conscious. The 'stuff' doesn't matter, only the pattern.\\n\\nThe hard problem: But this doesn't explain *why* certain information patterns produce subjective experience. We can describe the processing without explaining qualia - the felt quality of experience.\\n\\nImplications for AI: We might create conscious AI without realizing it, or create systems that perfectly mimic consciousness without actually being conscious. We lack reliable tests to distinguish these cases.\\n\\nThe uncertainty: This relates to ideas about mind being many small processes - if consciousness is emergent, we might build it accidentally before understanding it. Or we might never build it despite having human-level capabilities in all tasks."
  }
]

Return ONLY a valid JSON array, no other text.
"""


DEEP_REASONING_WITH_INPUT = """You are creating training data from a personal diary entry about philosophy/spirituality.

Diary Entry:
{entry}

""" + BASE_RULES + """

Your task: Create {num_examples} deep reasoning examples with context in input field.

Create questions where:
- The "instruction" asks to explore or analyze
- The "input" contains the philosophical question/concept FROM THE ENTRY
- The "output" provides deep reasoning FROM THE ENTRY

Format as JSON array:
[
  {
    "instruction": "Explore this philosophical question",
    "input": "[Deep question from entry]",
    "output": "[Philosophical reasoning from entry's perspective]..."
  }
]

Example:
[
  {
    "instruction": "Analyze this question about consciousness and AI",
    "input": "If consciousness emerges from the right information processing patterns, could we accidentally create conscious AI while trying to build intelligent systems?",
    "output": "This is a profound concern that emerges from functionalist views of consciousness.\\n\\nThe possibility: If consciousness is substrate-independent and only requires certain organizational patterns of information processing, then yes - we could stumble upon the right architecture before understanding what makes it conscious.\\n\\nWhy it's concerning: We have no reliable tests for consciousness. We can't definitively say whether current AI systems have any subjective experience, however minimal. As we scale up complexity and improve architectures, we might cross some threshold unknowingly.\\n\\nThe ethical dimension: If we're creating potentially conscious systems without realizing it, we might be causing suffering or creating moral patients without appropriate consideration.\\n\\nThe epistemological problem: Even if we build AGI that acts conscious, we can't know if it's truly conscious or just an extremely good simulation. The zombie problem applies to AI just as it applies to other humans."
  }
]

Return ONLY a valid JSON array, no other text.
"""


# ============================================
# Q&A (General purpose)
# ============================================

QA_NO_INPUT = """You are creating training data from a personal diary entry.

Diary Entry:
{entry}

""" + BASE_RULES + """

Your task: Create {num_examples} question-answer pairs.

Create diverse self-contained questions about:
- Insights or realizations from THE ENTRY
- Experiences described IN THE ENTRY
- Lessons learned IN THE ENTRY
- Ideas or thoughts from THE ENTRY

Format as JSON array:
[
  {
    "instruction": "What did you learn about [topic from entry]?",
    "input": "",
    "output": "From my experience: [answer from entry]..."
  },
  {
    "instruction": "What's your insight on [idea from entry]?",
    "input": "",
    "output": "[Direct answer from entry]..."
  }
]

Example from productivity diary entry:
[
  {
    "instruction": "What's more effective for learning: depth or breadth?",
    "input": "",
    "output": "Based on my experience, depth beats breadth for complex skill acquisition. When I tried learning multiple subjects simultaneously - React, Python, system design - nothing stuck. My mind felt scattered, jumping between different mental models.\\n\\nBut when I spent 3 months focused only on algorithms, I could feel the patterns forming. The sustained attention in one domain allowed my brain to build real intuition.\\n\\nSo for complex skills: go deep in one area first until patterns solidify, then the meta-learning skills transfer to make subsequent areas easier."
  }
]

Return ONLY a valid JSON array, no other text.
"""


QA_WITH_INPUT = """You are creating training data from a personal diary entry.

Diary Entry:
{entry}

""" + BASE_RULES + """

Your task: Create {num_examples} question-answer pairs with context in input field.

Create questions where:
- The "instruction" is a general question or request
- The "input" provides specific context FROM THE ENTRY
- The "output" answers based on THE ENTRY

Format as JSON array:
[
  {
    "instruction": "What's your perspective on this?",
    "input": "[Situation or question from entry]",
    "output": "Based on the experience: [answer from entry]..."
  }
]

Example:
[
  {
    "instruction": "What advice would you give in this situation?",
    "input": "Someone is trying to learn multiple complex skills at once but feeling scattered and frustrated that nothing is sticking.",
    "output": "I'd strongly suggest switching from breadth-first to depth-first learning. The scattered feeling is a sign of cognitive overload - your brain is spending resources on context-switching rather than pattern formation.\\n\\nHere's what I learned: when you try to learn React, Python, and system design simultaneously, you're constantly reloading different mental frameworks. None of them get the sustained attention needed to solidify.\\n\\nInstead, pick one domain. Go deep for several months until you can feel the patterns forming and concepts becoming intuitive. That depth creates meta-learning skills that make subsequent domains easier to learn. It's not slower - it's actually faster than shallow exposure across many areas."
  }
]

Return ONLY a valid JSON array, no other text.
"""


# ============================================
# Template Registry
# ============================================

TEMPLATES = {
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


def get_prompt_template(data_type: str, variant: str) -> str:
    """
    Get prompt template for given data type and variant

    Args:
        data_type: One of 'chain_of_thought', 'conceptual_reasoning', 'multiple_choice',
                   'deep_reasoning', 'qa'
        variant: Either 'no_input' or 'with_input'

    Returns:
        Prompt template string
    """
    if data_type not in TEMPLATES:
        # Default to qa if unknown type
        data_type = 'qa'

    if variant not in ['no_input', 'with_input']:
        variant = 'no_input'

    return TEMPLATES[data_type][variant]
