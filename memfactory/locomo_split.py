"""Official LoCoMo dialogue-level split used by Memory-R1.

The published 1:1:8 split is a split of complete dialogues, not a random
split of QA rows.  Keep the IDs explicit so a reordered JSON file cannot
silently change the train/validation/test boundary.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from memfactory.diagnostics import log_data_issue


SPLIT_DIALOGUE_IDS = {
    "train": ("conv-26",),
    "validation": ("conv-30",),
    "test": ("conv-41", "conv-42", "conv-43", "conv-44", "conv-47", "conv-48", "conv-49", "conv-50"),
}


def data_text(value: Any, location: str, *, allow_empty: bool = False) -> str:
    """Validate source text without turning null/containers into training text."""
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f'{location}: expected {"a string" if allow_empty else "non-empty text"}, got {value!r}')
    return value


def answer_text(value: Any, location: str) -> str:
    # Numeric gold answers and multi-item answers occur in LoCoMo. Retain the
    # existing comma-joined list convention, but reject null/bool/objects.
    if isinstance(value, list):
        if not value:
            raise ValueError(f'{location}: empty answer list')
        items = []
        for i,item in enumerate(value):
            if isinstance(item,list):
                raise ValueError(f'{location}[{i}]: nested answer lists are not supported')
            items.append(answer_text(item,f'{location}[{i}]'))
        return ', '.join(items)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return str(value)
    return data_text(value, location)


def questions_with_raw_ids(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Freeze QA identity before any category filtering.

    LoCoMo rows normally lack an explicit question_id.  The stable fallback is
    the index in the raw ``qa`` array, never the index in a filtered view.
    """
    questions = []
    location = f"dialogue {sample.get('sample_id', sample.get('dialogue_id', '?'))} qa"
    raw_questions = sample.get('qa', [])
    if not isinstance(raw_questions, list):
        log_data_issue(location, raw_questions, 'expected a QA list', 'use_empty_qa')
        return []
    ids = set()
    for raw_index, question in enumerate(raw_questions):
        if not isinstance(question, dict):
            log_data_issue(f'{location}[{raw_index}]', question, 'expected a QA object', 'skip_qa')
            continue
        row = dict(question)
        qid = question.get('question_id', raw_index)
        if qid is None: qid = raw_index
        if isinstance(qid, bool) or not isinstance(qid, (str,int)) or not str(qid).strip():
            log_data_issue(f'{location}[{raw_index}]', question, 'invalid question_id', 'skip_qa')
            continue
        row['question_id'] = str(qid)
        if row['question_id'] in ids:
            log_data_issue(f'{location}[{raw_index}]', question, 'duplicate question_id', 'skip_qa')
            continue
        ids.add(row['question_id'])
        questions.append(row)
    return questions


def validate_qa_rows(rows, location, question_key='question', answer_key='answer'):
    """Exclude only unusable QA; keep stable source IDs and record exclusions."""
    if not isinstance(rows, list):
        log_data_issue(location, rows, 'expected a QA list', 'use_empty_qa')
        return []
    valid, seen = [], set()
    for index, row in enumerate(rows):
        where = f'{location}[{index}]'
        try:
            if not isinstance(row, dict):
                raise ValueError('expected a QA object')
            question = data_text(row.get(question_key), where+'.'+question_key)
            answer = answer_text(row.get(answer_key), where+'.'+answer_key)
            qid = row.get('question_id', index)
            if qid is None:
                qid = index
            if isinstance(qid, bool) or not isinstance(qid, (str,int)) or not str(qid).strip():
                raise ValueError('invalid question_id')
            key = (row.get('dialogue_id'), str(qid))
            if key in seen:
                raise ValueError('duplicate question_id')
        except (ValueError, TypeError) as exc:
            log_data_issue(where, row, exc, 'skip_qa')
            continue
        seen.add(key)
        valid.append(dict(row, **{question_key:question, answer_key:answer, 'question_id':str(qid)}))
    return valid


def select_locomo_split(payload: Any, split: str) -> list[dict[str, Any]]:
    samples = payload if isinstance(payload, list) else [payload]
    if split not in SPLIT_DIALOGUE_IDS:
        raise ValueError(f"Unknown LoCoMo split: {split}")

    by_id = {}
    for index,sample in enumerate(samples):
        if not isinstance(sample,dict):
            raise ValueError(f'samples[{index}]: expected an object, got {sample!r}')
        did = data_text(sample.get('sample_id',sample.get('dialogue_id')), f'samples[{index}].dialogue_id')
        if did in by_id:
            raise ValueError(f'samples[{index}]: duplicate dialogue_id {did!r}')
        by_id[did] = sample
    expected_ids = SPLIT_DIALOGUE_IDS[split]
    missing = [dialogue_id for dialogue_id in expected_ids if dialogue_id not in by_id]
    if missing:
        raise ValueError(
            f"LoCoMo {split} split is missing dialogue IDs: {', '.join(missing)}"
        )
    selected = [by_id[dialogue_id] for dialogue_id in expected_ids]
    result = []
    for sample in selected:
        row = dict(sample)
        row["qa"] = [
            question for question in questions_with_raw_ids(sample)
            if int(question.get("category", 0)) != 5
        ]
        result.append(row)
    return result


def question_keys(path: str, split: str) -> set[tuple[str, str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    keys = set()
    for sample in select_locomo_split(payload, split):
        dialogue_id = str(sample.get("sample_id", sample.get("dialogue_id", "dialogue")))
        for question in sample.get("qa", []):
            keys.add((dialogue_id, str(question["question_id"])))
    return keys
