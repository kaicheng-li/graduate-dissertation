"""Answer prompt; retrieval is implemented by construct_memory_bank.top_k_per_speaker."""
from typing import Dict, Any, List

RERANK_PROMPT = """You are an intelligent memory assistant tasked with retrieving
accurate information from conversation memories.
# CONTEXT:
You have access to memories from two speakers in a conversation.
These memories contain timestamped information that may be relevant
to answering the question.
# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago"),
calculate the actual date based on the memory timestamp.
6. Always convert relative time references to specific dates, months, or years.
7. Focus only on the content of the memories. Do not confuse character names
8. The answer should be less than 5-6 words.
9. IMPORTANT: Select memories you found that are useful for answering the questions,
and output it before you answer questions.
10. IMPORTANT: Output the final answer after **Answer:**
# APPROACH (Think step by step):
1. Examine all relevant memories
2. Examine the timestamps carefully
3. Look for explicit mentions that answer the question
4. Convert relative references if needed
5. Formulate a concise answer
6. Double-check the answer correctness
7. Ensure the final answer is specific
8. First output the memories that you found are important before you answer questions
"""

def build_answer_input(question: str, memories_by_speaker: Dict[str, List[Dict[str, Any]]]) -> str:
    """Attach runtime retrieved memories to the canonical Answer prompt."""
    sections = []
    for speaker, memories in memories_by_speaker.items():
        memory_lines = "\n".join(
            f"- {memory.get('timestamp', '')}: {memory.get('text', '')}"
            for memory in memories
        )
        sections.append(
            f"Memories for user {speaker}:\n"
            f"{memory_lines}\n"
            f"... (In total {len(memories)} relevant memories from {speaker}'s Memory Bank are provided) ..."
        )
    return (
        f"{RERANK_PROMPT}\n\n"
        + "\n\n".join(sections)
        + f"\n\nQuestion: {question}\n"
    )
