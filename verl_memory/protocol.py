"""Framework-independent bank protocol; no alternate prompt or action executor."""
import hashlib
import json
import math
from pathlib import Path

from memfactory.diagnostics import EvidenceStore, log_manager_parse_error
from memfactory.modules.memory_updater import UPDATE_MEMORY_PROMPT, build_manager_input
from memfactory.modules.memory_retriever import RERANK_PROMPT, build_answer_input
from memfactory.modules.memory_extractor import validate_facts
from memfactory.locomo_split import data_text, validate_qa_rows


def append_json(path, row):
    path = Path(path)
    line = json.dumps(row, ensure_ascii=False, allow_nan=False)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as f:
            f.write(line+'\n')
    except OSError as exc:
        print(f'LOG_WRITE_WARNING path={path} error={exc} record={line}', flush=True)


def fingerprint():
    return {name: hashlib.sha256(value.encode()).hexdigest() for name, value in
            [('manager_prompt', UPDATE_MEMORY_PROMPT), ('answer_prompt', RERANK_PROMPT)]}


def prompt_for_turn(turn, bank, top_k_fn, k=5):
    facts = validate_facts(turn.get('facts', []), f"turn[{turn.get('turn_index','?')}].facts")
    if not facts:
        return None
    visible = top_k_fn(' '.join(f['text'] for f in facts), bank, k)
    return build_manager_input(visible, facts)


def answer_prompt(question, bank, participants, retrieve, k=30):
    memories = retrieve(question, bank, participants, k)
    return build_answer_input(question, {p: [m for m in memories if m['speaker'] == p]
                                       for p in participants}), memories


def commit_turn(timeline, turn, bank, prompt, raw, generation, root, trajectory_id, epoch, group, rank):
    # Use the exact same retrieval, parser, validation and executor as formal test.
    from evaluation.manager_validation import manager_transition
    def generated(actual_prompt):
        if actual_prompt != prompt:
            print('ROLLOUT_WARNING generated prompt differs from bank transition prompt', flush=True)
        return raw
    generated.last_generation = generation
    state = dict(turn, dialogue_id=timeline['dialogue_id'])
    after, action = manager_transition(state, bank, generated, 5)
    store = EvidenceStore(root)
    log_manager_parse_error(Path(root)/f'manager_errors_rank_{rank}.jsonl', state, action,
                            mode='verl_rollout', epoch=epoch, group=group, trajectory_id=trajectory_id)
    append_json(Path(root)/f'trajectories_rank_{rank}.jsonl', dict(
        event='trajectory_turn', epoch=epoch, group=group, trajectory_id=trajectory_id,
        dialogue_id=timeline['dialogue_id'], turn=turn['turn_index'], valid=action['valid'],
        error=action['error'], actions=action['actions'], bank_size=len(after),
        partial_applied=action.get('partial_applied', False), rejected_ids=action.get('rejected_ids', []),
        rejected_actions=action.get('rejected_actions', []), action_repairs=action.get('action_repairs', []),
        id_mappings=action.get('id_mappings', []), parsing=action.get('parsing', {}),
        bank_before_ref=store.save(bank), memory_bank_ref=store.save(after),
        action_ref=store.save(action, 'manager_action'), generation=generation))
    return after, action


def per_qa_advantages(rewards, epsilon=1e-12):
    """Population-std per QA, then QA mean: preserves existing trajectory objective."""
    result = [0.] * len(rewards)
    for values in zip(*rewards, strict=True):
        mean = sum(values) / len(values)
        var = sum((v-mean)**2 for v in values) / len(values)
        if math.sqrt(var) <= epsilon:
            continue
        for i, value in enumerate(values):
            result[i] += (value-mean) / (math.sqrt(var) + 1e-6) / len(rewards[0])
    return result


def block_advantages(points, terminal, turns, interval, aux_lambda, tau):
    per_turn, records = {}, []
    for previous, current in zip(points, points[1:]):
        previous_scores = previous.get('scores_by_question')
        current_scores = current.get('scores_by_question')
        common_ids = set(previous_scores or {}).intersection(current_scores or {})
        if previous_scores is not None and current_scores is not None:
            # A question entering the eligible set is not evidence that the
            # bank improved. Compare only questions scored at both snapshots.
            if common_ids:
                phi_before = sum(previous_scores[q] for q in common_ids) / len(common_ids)
                phi_after = sum(current_scores[q] for q in common_ids) / len(common_ids)
                delta = phi_after - phi_before
            else:
                phi_before = phi_after = None
                delta = 0.0
            common_count = len(common_ids)
        else:
            # Compatibility for old traces/tests that only persisted phi.
            phi_before = previous.get('phi')
            phi_after = current.get('phi')
            delta = ((phi_after - phi_before)
                     if phi_before is not None and phi_after is not None else 0.0)
            common_count = None
        auxiliary = aux_lambda * max(-1., min(1., delta/tau))
        for turn in range(previous['after_turn'], current['after_turn']):
            per_turn[turn] = terminal + auxiliary
        records.append(dict(start_turn=previous['after_turn'], end_turn=current['after_turn'],
                            phi_before=phi_before, phi_after=phi_after,
                            common_question_count=common_count, delta=delta,
                            auxiliary_advantage=auxiliary, terminal_advantage=terminal))
    return per_turn, records


def validate_timelines(rows):
    from copy import deepcopy
    from memfactory.diagnostics import log_data_issue
    if not isinstance(rows, list):
        log_data_issue('timelines', rows, 'expected a timeline list', 'use_empty_dataset')
        return []
    valid, dialogue_ids = [], set()
    for index, source in enumerate(rows):
        location = f'timelines[{index}]'
        try:
            if not isinstance(source, dict):
                raise ValueError('expected a timeline object')
            t = deepcopy(source)
            did = data_text(t.get('dialogue_id'), location+'.dialogue_id')
            if did in dialogue_ids:
                raise ValueError('duplicate dialogue timeline')
            for key in ('turns', 'participants'):
                if not isinstance(t.get(key), list) or not t[key]:
                    raise ValueError(f'{did}.{key}: expected a non-empty list')
            for person in t['participants']:
                data_text(person, did+'.participants')
            t['participants'] = list(dict.fromkeys(t['participants']))
            for i, turn in enumerate(t['turns']):
                where = f'{did}.turns[{i}]'
                if not isinstance(turn, dict):
                    log_data_issue(where, turn, 'invalid turn object', 'preserve_turn_as_noop')
                    turn = dict(turn_index=i, speaker='', text='', facts=[])
                    t['turns'][i] = turn
                if not isinstance(turn.get('text', ''), str):
                    log_data_issue(where+'.text', turn.get('text'), 'invalid source text', 'use_empty_text')
                    turn['text'] = ''
                if turn.get('speaker') not in t['participants']:
                    log_data_issue(where, turn, 'unknown speaker', 'preserve_turn_as_noop')
                    turn['facts'] = []
                turn['facts'] = validate_facts(turn.get('facts'), where+'.facts')
            if [turn.get('turn_index') for turn in t['turns']] != list(range(len(t['turns']))):
                raise ValueError('Turns must be contiguous and chronological; cannot guess source alignment')
            t['questions'] = validate_qa_rows(t.get('questions'), did+'.questions')
            if not t['questions']:
                raise ValueError('No usable QA remains for reward/evaluation')
        except (ValueError, TypeError) as exc:
            log_data_issue(location, source, exc, 'skip_dialogue')
            continue
        dialogue_ids.add(did)
        valid.append(t)
    return valid


def load_bank(root, ref):
    import gzip
    raw = gzip.decompress((Path(root)/ref['path']).read_bytes())
    return json.loads(raw)


# Teacher-forced gold scoring is shared with the vLLM worker, not a second trainer.
from dataclasses import dataclass
ANSWER_PREFIX = "**Answer:** "

@dataclass(frozen=True)
class Prepared:
    input_ids: tuple[int, ...]
    gold_start: int
    gold_tokens: int
    crossing_boundary_tokens: int

def prepare_input(tokenizer, tokenize_chat_prompt, prompt, gold, max_prompt_tokens=4096):
    # The runtime helper defines the actual generation prompt and its truncation.
    prompt_ids = tokenize_chat_prompt(tokenizer, prompt, max_prompt_tokens)["input_ids"][0].tolist()
    completion = ANSWER_PREFIX + gold
    encoded = tokenizer(completion, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = list(encoded["input_ids"]), list(encoded["offset_mapping"])
    boundary = len(ANSWER_PREFIX)
    scored = [i for i, (start, end) in enumerate(offsets) if end > boundary]
    crossing = 0
    for i in scored:
        start, end = offsets[i]
        if start < boundary < end:
            # Byte-level tokenizers often combine the separator space and gold.
            crossing += 1
    start = len(prompt_ids) + scored[0]
    return Prepared(tuple(prompt_ids + ids), start, len(scored), crossing)
