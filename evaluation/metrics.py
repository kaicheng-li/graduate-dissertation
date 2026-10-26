"""Memory-R1/LoCoMo answer metrics and category metadata."""

from __future__ import annotations

import json
import math
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from memfactory.locomo_split import questions_with_raw_ids, data_text, answer_text


LOCOMO_CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}


def normalize_text(text: str) -> str:
    """Use the LoCoMo/MemBench SQuAD-style normalization."""
    text = data_text(text, 'metric text', allow_empty=True).lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def f1_score(prediction: str, gold: str) -> float:
    """LoCoMo token F1 used for evaluation and the Memory-R2-style RL reward.

    The overlap numerator counts unique token values, while precision and
    recall retain the original token-list lengths. Keeping this implementation
    here makes training and evaluation use one definition.
    """
    prediction_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not prediction_tokens or not gold_tokens:
        return float(prediction_tokens == gold_tokens)
    overlap = len(set(prediction_tokens) & set(gold_tokens))
    if not overlap:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu1_score(prediction: str, gold: str) -> float:
    """Memory-R1 BLEU-1: clipped unigram precision with brevity penalty."""
    prediction_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not prediction_tokens or not gold_tokens:
        return 0.0
    gold_counts = Counter(gold_tokens)
    prediction_counts = Counter(prediction_tokens)
    clipped = sum(min(count, gold_counts[token]) for token, count in prediction_counts.items())
    precision = clipped / len(prediction_tokens)
    brevity_penalty = (
        1.0
        if len(prediction_tokens) >= len(gold_tokens)
        else math.exp(1.0 - len(gold_tokens) / len(prediction_tokens))
    )
    return brevity_penalty * precision


def exact_match(prediction: str, gold: str) -> float:
    # Preserve the baseline EM normalization, distinct from token F1/BLEU-1.
    normalized_prediction = re.sub(r"[^a-z0-9]+", " ", data_text(prediction,'prediction',allow_empty=True).lower()).strip()
    normalized_gold = re.sub(r"[^a-z0-9]+", " ", data_text(gold,'gold',allow_empty=True).lower()).strip()
    return float(bool(normalized_prediction) and normalized_prediction == normalized_gold)


def load_question_metadata(raw_data_path: str | None) -> dict[tuple[str, str], dict[str, Any]]:
    if not raw_data_path:
        return {}
    payload = json.loads(Path(raw_data_path).read_text(encoding="utf-8"))
    samples = payload if isinstance(payload, list) else [payload]
    metadata = {}
    for sample in samples:
        dialogue_id = str(sample.get("sample_id", sample.get("dialogue_id", "")))
        for question in questions_with_raw_ids(sample):
            category = int(question.get("category", 0))
            metadata[(dialogue_id, str(question["question_id"]))] = {
                "category": category,
                "category_name": LOCOMO_CATEGORY_NAMES.get(category, "unknown"),
            }
    return metadata


def add_question_metadata(
    row: dict[str, Any], metadata: dict[tuple[str, str], dict[str, Any]]
) -> None:
    key = (str(row.get("dialogue_id", "")), str(row.get("question_id", "")))
    values = metadata.get(key, {})
    category = int(row.get("category", 0) or 0)
    if category not in LOCOMO_CATEGORY_NAMES:
        category = int(values.get("category", 0))
    row["category"] = category
    row["category_name"] = values.get(
        "category_name", LOCOMO_CATEGORY_NAMES.get(category, "unknown")
    )


def score_record(record: dict[str, Any]) -> None:
    prediction = record.get('prediction')
    prediction = '' if prediction is None else data_text(prediction,'prediction',allow_empty=True)
    gold = answer_text(record.get('gold'), 'gold')
    record["em"] = exact_match(prediction, gold)
    record["f1"] = f1_score(prediction, gold)
    record["bleu1"] = bleu1_score(prediction, gold)


def mean(records: Iterable[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return sum(values) / len(values) if values else None


def aggregate_answer_metrics(
    records: list[dict[str, Any]], value_keys: tuple[str, ...]
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups["overall"] = records
    for record in records:
        category_name = record.get("category_name")
        if not category_name or category_name == "unknown":
            category_name = LOCOMO_CATEGORY_NAMES.get(int(record.get("category", 0) or 0), "unknown")
        groups[str(category_name)].append(record)
    return {
        group_name: {
            "count": len(group),
            **{key: mean(group, key) for key in value_keys},
        }
        for group_name, group in groups.items()
    }


def extract_answer(text: str) -> str:
    if text is None:
        return ''
    if not isinstance(text, str):
        raise ValueError('Answer completion must be text')
    tagged = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if tagged:
        return tagged.group(1).strip()
    # Prefer a standalone final-answer label over a label inside quoted memory.
    label = r'(?:\*\*(?:Final\s+)?Answer\s*:\*\*|\*\*(?:Final\s+)?Answer\*\*\s*:|(?:Final\s+)?Answer\s*:)'
    marked = list(re.finditer(r'^\s*(?:#{1,6}\s+)?'+label, text, re.IGNORECASE | re.MULTILINE))
    if not marked:
        marked = list(re.finditer(label, text, re.IGNORECASE))
    answer = text[marked[-1].end():] if marked else text
    # Stop at an explicit explanation section, not at an arbitrary newline or
    # a word count: lists, dates and multiline answers must remain intact.
    explanation = r'(?:Explanation|Reasoning|Rationale|Justification|Evidence|Analysis)'
    boundary = re.search(
        r'(?:\*\*\s*'+explanation+r'\s*(?::\s*\*\*|\*\*\s*:)|'
        r'^[ \t]*(?:#{1,6}[ \t]+)?'+explanation+r'[ \t]*:|'
        r'^[ \t]*#{1,6}[ \t]+'+explanation+r'[ \t]*$)',
        answer, re.IGNORECASE | re.MULTILINE)
    return answer[:boundary.start()].strip() if boundary else answer.strip()
