"""Memory Manager action validation with isolated rejection and recorded repairs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


ALLOWED_EVENTS = {"ADD", "UPDATE", "DELETE", "NONE"}


@dataclass(frozen=True)
class ActionResult:
    valid: bool
    memory_bank: list[dict[str, Any]]
    applied_actions: int
    error: str | None = None
    rejected_ids: tuple[str, ...] = ()
    rejected_actions: tuple[dict[str, Any], ...] = ()
    repairs: tuple[dict[str, Any], ...] = ()
    id_mappings: tuple[dict[str, Any], ...] = ()


def validate_and_apply(
    memory_bank: list[dict[str, Any]],
    visible_memories: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    *,
    dialogue_id: str,
    source_turn: int,
    speaker: str,
    timestamp: str,
) -> ActionResult:
    """Plan against the input snapshot, then commit independent target changes.

    ADD IDs are allocated above the largest decimal ID in the whole input bank.
    Model ADD IDs are output-local aliases. Visible existing IDs are used as-is
    and take precedence over aliases; hidden IDs cannot be targeted.
    Unique aliases can refer to ADDs regardless of output order. Existing targets
    are never inferred from old_memory or text. One valid UPDATE wins over DELETEs
    for the same target, regardless of output order. Other repeated mutations
    leave only that target unchanged.
    Missing/NONE entries preserve memory. Errors never roll back other targets.
    """
    original = deepcopy(memory_bank)
    if not isinstance(decisions, list):
        return ActionResult(False, original, 0, "memory must be a list")
    by_id = {str(m['id']): m for m in original}
    visible = {str(m['id']): by_id[str(m['id'])] for m in visible_memories
               if str(m['id']) in by_id}
    errors, repairs, mappings = [], [], []
    rows, additions, aliases, mutations = [], {}, {}, {}
    numeric_ids = [int(k) for k in by_id if k.isascii() and k.isdecimal()]
    next_id = max(numeric_ids, default=-1) + 1

    def reject(index, memory_id, message):
        errors.append(dict(index=index, id=memory_id, error=message))

    # Allocate all ADDs before resolving any reference: array order cannot turn
    # a valid alias into an unknown ID, nor affect another existing target.
    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            reject(index, '', f"action {index} is not an object")
            continue
        raw_event = decision.get('event')
        event = raw_event.strip().upper() if isinstance(raw_event, str) else ''
        raw_id = decision.get('id')
        model_id = str(raw_id).strip() if isinstance(raw_id, (str, int)) and not isinstance(raw_id, bool) else ''
        raw_text = decision.get('text')
        text = raw_text.strip() if isinstance(raw_text, str) else ''
        if event not in ALLOWED_EVENTS:
            reject(index, model_id, f"unsupported event {event!r}")
            continue
        if event in ('ADD', 'UPDATE') and not text:
            reject(index, model_id, f"{event} requires non-empty text")
            continue
        if event == 'ADD':
            actual_id = str(next_id)
            next_id += 1
            additions[actual_id] = dict(id=actual_id, text=text, speaker=speaker,
                                       timestamp=timestamp, source_turn=source_turn)
            mappings.append(dict(index=index, event=event, model_id=model_id, bank_id=actual_id))
            if model_id:
                aliases.setdefault(model_id, set()).add(actual_id)
        else:
            rows.append((index, event, model_id, text))

    # Resolve every non-ADD against the same snapshot. Generated numeric IDs
    # alone are not references: the model only knows the IDs it emitted/saw.
    for index, event, model_id, text in rows:
        target = model_id if model_id in visible else None
        candidates = aliases.get(model_id, set()) if model_id not in by_id else set()
        if target is None and len(candidates) == 1:
            target = next(iter(candidates))
            repairs.append(dict(index=index, kind='resolve_add_alias',
                                original_id=model_id, resolved_id=target))
        elif target is None and len(candidates) > 1:
            # Repeated ADD placeholders do not invalidate the ADDs. Only the
            # ambiguous reference is skipped; never guess which new item it meant.
            reject(index, model_id, f"ambiguous ADD alias {model_id!r}")
            continue
        if target is None:
            if event == 'NONE':
                repairs.append(dict(index=index, kind='ignore_unseen_none', id=model_id))
            else:
                reject(index, model_id, f"{event} id {model_id!r} was not visible in the prompt")
            continue
        mappings.append(dict(index=index, event=event, model_id=model_id, bank_id=target))
        if event != 'NONE':
            mutations.setdefault(target, []).append((index, event, text))

    # Select a single final change per target before mutating the bank.
    selected = {}
    for target, actions in mutations.items():
        updates = [a for a in actions if a[1] == 'UPDATE']
        deletes = [a for a in actions if a[1] == 'DELETE']
        if len(updates) == 1 and deletes:
            selected[target] = updates[0]
            repairs.append(dict(kind='update_over_delete', id=target,
                                selected_index=updates[0][0],
                                ignored_indices=[a[0] for a in deletes]))
            continue
        if len(actions) > 1:
            for index, event, text in actions:
                reject(index, target, f"duplicate target id {target!r}")
            continue
        selected[target] = actions[0]

    bank, applied = [], 0
    for entry in original + list(additions.values()):
        target = str(entry['id'])
        action = selected.get(target)
        if action and action[1] == 'DELETE':
            # ADD+DELETE cancels the new entry without modifying the input bank.
            applied += int(target not in additions)
            continue
        if action:
            entry.update(text=action[2], timestamp=timestamp, source_turn=source_turn)
        applied += int(target in additions or action is not None)
        bank.append(entry)
    errors.sort(key=lambda r: r['index'])
    mappings.sort(key=lambda r: r['index'])
    error = '; '.join(r['error'] for r in errors) if errors else None
    rejected = tuple(sorted({r['id'] for r in errors if r['id']}))
    return ActionResult(not errors, bank, applied, error, rejected, tuple(errors),
                        tuple(repairs), tuple(mappings))
