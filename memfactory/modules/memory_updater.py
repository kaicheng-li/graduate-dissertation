"""Original Manager instructions shared by training, validation and test."""
import json
from typing import Dict, Any, List

# UPDATE_MEMORY_PROMPT-old one
"""You are a smart memory manager which controls the memory of a system.
You can perform four operations: (1) add into the memory, (2) update the
memory, (3) delete from the memory, and (4) no change.
Based on the above four operations, the memory will change.
Compare newly retrieved facts with the existing memory. For each new fact,
decide whether to:- ADD: Add it to the memory as a new element- UPDATE: Update an existing memory element- DELETE: Delete an existing memory element- NONE: Make no change (if the fact is already present or irrelevant)
1. **Add**: If the retrieved facts contain new information not present
in the memory, then you have to add it by generating a new ID in the id field.- Example:
Old Memory:
[
{"id" : "0", "text" : "User is a software engineer"}
]
Retrieved facts: ["Name is John"]
New Memory:
{
"memory" : [
{"id" : "0", "text" : "User is a software engineer", "event" : "NONE"},
{"id" : "1", "text" : "Name is John", "event" : "ADD"}
]
}
2. **Update**: If the retrieved facts contain information that is already
present in the memory but the information is totally different, then
you have to update it.
If the retrieved fact contains information that conveys the same thing as
the memory, keep the version with more detail.
Example (a)– if the memory contains "User likes to play cricket" and the
retrieved fact is "Loves to play cricket with friends", then update the
memory with the retrieved fact.
Example (b)– if the memory contains "Likes cheese pizza" and the
retrieved fact is "Loves cheese pizza", then do NOT update it because they
convey the same information.
Important: When updating, keep the same ID and preserve old_memory.- Example:
Old Memory:
[
{"id" : "0", "text" : "I really like cheese pizza"},
{"id" : "2", "text" : "User likes to play cricket"}
]
Retrieved facts: ["Loves chicken pizza", "Loves to play cricket with friends"]
New Memory:
{
"memory" : [
{"id" : "0", "text" : "Loves cheese and chicken pizza", "event" : "UPDATE",
"old_memory" : "I really like cheese pizza"},
{"id" : "2", "text" : "Loves to play cricket with friends", "event" : "UPDATE",
"old_memory" : "User likes to play cricket"}
]
}
3. **Delete**: If the retrieved facts contain information that contradicts
the memory, delete it. When deleting, return the same IDs — do not generate new IDs.- Example:
Old Memory:
[
{"id" : "1", "text" : "Loves cheese pizza"}
]
Retrieved facts: ["Dislikes cheese pizza"]
New Memory:
{
"memory" : [
{"id" : "1", "text" : "Loves cheese pizza", "event" : "DELETE"}
]
}
4. **No Change**: If the retrieved facts are already present, make no change.- Example:
Old Memory:
[
{"id" : "0", "text" : "Name is John"}
]
Retrieved facts: ["Name is John"]
New Memory:
{
"memory" : [
{"id" : "0", "text" : "Name is John", "event" : "NONE"}
]
}"""

UPDATE_MEMORY_PROMPT = """You are a smart memory manager which controls the memory of a system.
You can perform four operations: (1) Add into the memory, (2) UPDATE the
memory, (3) DELETE from the memory, and (4) NONE change.

Compare each newly retrieved fact with the existing memory. For each new fact, decide whether to:
- ADD: If the retrieved facts contain new information not present in the memory, then you have to add it by generating a new ID in the id field.
- UPDATE: If the retrieved facts contain information that is already present in the memory but the information is totally different, then you have to update it;If the retrieved fact contains information that conveys the same thing as the memory, keep the version with more detail.
- DELETE: If the retrieved facts contain information that contradicts the memory, delete it. When deleting, return the same IDs — do not generate new IDs.
- NONE: If the retrieved facts are already present, make no change.

Note:Old Memory and Retrieved facts are two different sets. Old Memory items have IDs. Retrieved facts do NOT have IDs.

Rules:
1. Always output a single valid JSON object with exactly one top-level key: "memory".
2. The "memory" value must be a complete array containing every existing memory element exactly once, plus any newly added elements.
3. For every existing memory element, output exactly one entry with one of these events: "NONE", "UPDATE", or "DELETE". NONE, UPDATE and DELETE must use an id that already exists in Old Memory
4. For each new fact that should be added, append a new entry with event "ADD". Never assign an new ID to a retrieved fact unless the action is ADD.
5. UPDATE must include: id, text, event, old_memory. Keep the same id and set old_memory to the original memory text.
6. DELETE must include: id, text, event. Keep the original text and same id.
7. NONE and ADD must include: id, text, event. Do not include old_memory for NONE or ADD.
8. If multiple facts affect the same existing memory element, output that element only once. If any fact contradicts it, Delete takes priority. Otherwise UPDATE into the most complete text.
9. Do not output explanations, reasoning, comments, markdown, code fences, or any text before/after the JSON.

Output schema:
{
  "memory": [
    {
      "id": "string",
      "text": "string",
      "event": "ADD | UPDATE | DELETE | NONE",
      "old_memory": "string (only for UPDATE)"
    }
  ]
}

Example:
Old Memory:
[
  {"id": "0", "text": "User is a software engineer"},
  {"id": "1", "text": "Loves cheese pizza"},
  {"id": "2", "text": "User likes to play cricket"}
]
Retrieved facts:
["Name is John", "Dislikes cheese pizza", "Loves to play cricket with friends", "User is a software engineer"]

New Memory:
{
  "memory": [
    {"id": "0", "text": "User is a software engineer", "event": "NONE"},
    {"id": "1", "text": "Loves cheese pizza", "event": "DELETE"},
    {"id": "2", "text": "Loves to play cricket with friends", "event": "UPDATE", "old_memory": "User likes to play cricket"},
    {"id": "3", "text": "Name is John", "event": "ADD"}
  ]
}

Now process the given Old Memory and Retrieved facts. Output ONLY the JSON object."""

def build_manager_input(old_memory: List[Dict[str, Any]], retrieved_facts: List[Any]) -> str:
    """Match the prompt example: id/text memories and plain fact strings."""
    return (
        UPDATE_MEMORY_PROMPT
        + "\n\nOld Memory:\n"
        + json.dumps([{"id": memory["id"], "text": memory["text"]}
                      for memory in old_memory], ensure_ascii=False, indent=2)
        + "\n\nRetrieved facts:\n"
        + json.dumps([fact["text"] if isinstance(fact, dict) else fact
                      for fact in retrieved_facts], ensure_ascii=False, indent=2)
        + "\n\nNew Memory:\n"
    )
