"""Fact extraction prompt and input formatting used by data preparation."""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional
from memfactory.locomo_split import data_text


def validate_facts(facts, location, text_key='text'):
    """Keep valid facts and report original positions of malformed records."""
    from memfactory.diagnostics import log_data_issue
    if not isinstance(facts, list):
        log_data_issue(location, facts, 'expected a fact list', 'use_empty_facts')
        return []
    valid = []
    for index, fact in enumerate(facts):
        try:
            if not isinstance(fact, dict):
                raise ValueError('expected a fact object')
            data_text(fact.get(text_key), f'{location}[{index}].{text_key}')
        except ValueError as exc:
            log_data_issue(f'{location}[{index}]', fact, exc, 'skip_fact')
            continue
        valid.append(fact)
    return valid


@dataclass
class ConversationMessage:
    """对话消息"""
    role: str           # user / assistant / system
    content: str        # 消息内容
    timestamp: Optional[str] = None # 时间戳
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

def format_conversation(messages: List[ConversationMessage]) -> str:
    """格式化对话为字符串"""
    lines = []
    for msg in messages:
        if not msg.timestamp or msg.timestamp == "" or msg.timestamp == " ":
            lines.append(f"{msg.role}: {msg.content}")
        else:
            lines.append(f"{msg.role}: [{msg.timestamp}] {msg.content}")
    return "\n".join(lines)

EXTRACTION_PROMPT_EN = """You are a memory extraction expert.
Your task is to extract memories from the user's perspective based on the conversation between the user and the assistant. This means identifying information the user might remember—including the user's own experiences, thoughts, plans, or statements and actions made by others (such as the assistant) that affect the user or are acknowledged by the user.

Please perform the following operations:
1. Identify information reflecting the user's experiences, beliefs, concerns, decisions, plans, or responses—including meaningful information from the assistant acknowledged or responded to by the user.

2. Clearly parse all references to time, people, and events:
   - If possible, convert relative time expressions (e.g., "yesterday", "next Friday") to absolute dates using message timestamps.
   - Clearly distinguish between event time and message time.
   - If specific locations are mentioned, include them.
   - Resolve all pronouns, aliases, and vague references to full names or clear identities.

3. Always write in the third person perspective, using "User" to refer to the user, rather than the first person.

4. Do not omit any information the user might remember.
   - Include all key experiences, thoughts, emotional reactions, and plans.
   - Prioritize completeness and fidelity over brevity.

Return a valid JSON object with the following structure:

{{
  "memory_list": [
    {{
      "key": "<string, unique and concise memory title>",
      "memory_type": "<string, 'LongTermMemory' or 'UserMemory'>",
      "value": "<detailed, independent, and unambiguous memory statement>",
      "tags": ["<list of relevant topic keywords>"]
    }}
  ],
  "summary": "<paragraph naturally summarizing the above memories from the user's perspective, 120-200 words>"
}}

Conversation:
{conversation}

Your output:"""

def build_extract_input(turn: Dict[str, Any]) -> str:
    """Attach a single dialogue turn to the reference extraction prompt.

    The turn dict ({speaker, text, timestamp}) is formatted as a one-message
    conversation, matching the reference prompt's {conversation} slot.
    """
    message = ConversationMessage(
        role=str(turn.get("speaker") or "user"),
        content=data_text(turn.get("text", ""), 'extract turn.text', allow_empty=True),
        timestamp=str(turn.get("timestamp") or ""),
    )
    conversation = format_conversation([message])
    return EXTRACTION_PROMPT_EN.format(conversation=conversation)
