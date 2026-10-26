"""One Manager transition shared by continuous training, validation and test."""
import json
from time import perf_counter
from typing import Any, Callable
from memfactory.manager_env import validate_and_apply
from memfactory.modules.memory_updater import build_manager_input
from memfactory.modules.memory_extractor import validate_facts
from scripts.construct_memory_bank import parse_manager_output, top_k
ManagerGenerate = Callable[[str], str]

def manager_transition(
    state: dict[str, Any],
    memory_bank: list[dict[str, Any]],
    manager_generate: ManagerGenerate,
    manager_top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    facts = validate_facts(state.get('facts', []),
                           f"{state.get('dialogue_id','?')}.turn[{state.get('turn_index','?')}].facts")
    if not facts:
        return list(memory_bank), {"valid": True, "actions": 0, "error": None,
                                   "facts": [], "visible_memories": [], "decisions": []}
    retrieval_started = perf_counter()
    query = " ".join(str(fact["text"]) for fact in facts)
    visible = top_k(query, memory_bank, manager_top_k)
    prompt = build_manager_input(visible, facts)
    timing = {'retrieval_prompt_seconds': perf_counter() - retrieval_started}
    generation_started = perf_counter()
    raw = manager_generate(prompt)
    timing['generator_callback_seconds'] = perf_counter() - generation_started
    generation = getattr(manager_generate, 'last_generation', None)
    if not isinstance(generation, dict):
        generation = None
    parsing = {}
    try:
        decisions = parse_manager_output(raw, parsing)
    except (ValueError, json.JSONDecodeError) as exc:
        return list(memory_bank), {
            "valid": False, "actions": 0, "error": f"parse: {exc}", "raw": raw,
            "facts": facts, "visible_memories": visible, "prompt": prompt,
            "generation": generation, "timing": timing, "parsing": parsing,
        }
    result = validate_and_apply(
        memory_bank,
        visible,
        decisions,
        dialogue_id=str(state["dialogue_id"]),
        source_turn=int(state['turn_index']),
        speaker=str(state.get("speaker", "")),
        timestamp=str(state.get("timestamp", "")),
    )
    return result.memory_bank, {
        "valid": result.valid,
        "actions": result.applied_actions,
        "error": result.error,
        "rejected_ids": list(result.rejected_ids),
        "rejected_actions": list(result.rejected_actions),
        "action_repairs": list(result.repairs),
        "id_mappings": list(result.id_mappings),
        "partial_applied": bool(not result.valid and result.applied_actions),
        "raw": raw,
        "facts": facts, "visible_memories": visible, "prompt": prompt, "decisions": decisions,
        "generation": generation, "timing": timing, "parsing": parsing,
    }
