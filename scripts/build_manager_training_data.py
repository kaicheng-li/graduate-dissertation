"""Extract and cache teacher facts for chronological dialogue timelines; no memory construction."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memfactory.modules.memory_extractor import build_extract_input, validate_facts
from memfactory.locomo_split import select_locomo_split, questions_with_raw_ids, data_text, answer_text, validate_qa_rows

DEFAULT_MAX_TOKENS = 8192
DEFAULT_MAX_RETRIES = 2
COMPACT_JSON_INSTRUCTION = """

IMPORTANT: Your prior response was too long or incomplete. Return ONLY one
complete JSON object. Consolidate duplicate or related facts into concise,
self-contained memories. Keep every memory text under 180 characters and the
entire response under 6000 characters. Do not repeat source dialogue or add
explanations.
"""
LOGGER = logging.getLogger("memory_r1.data_builder")


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _session_number(name: str) -> int:
    try:
        return int(name.split("_", 1)[1])
    except (IndexError, ValueError):
        return 10**9


def normalize_locomo_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Flatten sessions and preserve QA without assigning a question time."""
    conversation = sample.get("conversation", {})
    if not isinstance(conversation, dict):
        raise ValueError("LoCoMo sample conversation must be an object.")

    dialogue_id = data_text(sample.get('sample_id', sample.get('dialogue_id', 'dialogue')), 'dialogue_id')
    turns = []
    for name,value in conversation.items():
        if re.fullmatch(r'session_\d+', name) and not isinstance(value,list):
            raise ValueError(f'{dialogue_id}.{name}: expected a turn list, got {value!r}')
    session_names = sorted(
        (name for name in conversation if re.fullmatch(r'session_\d+', name)),
        key=_session_number,
    )
    for session_name in session_names:
        for local_index, raw_turn in enumerate(conversation[session_name]):
            location = f'{dialogue_id}.{session_name}[{local_index}]'
            if not isinstance(raw_turn, dict):
                raise ValueError(f'{location}: expected a turn object, got {raw_turn!r}')
            turn_index = len(turns)
            dia_id = data_text(raw_turn.get("dia_id") or f"D{_session_number(session_name)}:{local_index}", location+'.dia_id')
            turns.append({
                "speaker": data_text(raw_turn.get("speaker"), location+'.speaker'),
                "text": data_text(raw_turn.get("text", ""), location+'.text', allow_empty=True),
                "timestamp": data_text(raw_turn.get("timestamp") or conversation.get(f"{session_name}_date_time") or "",
                                       location+'.timestamp', allow_empty=True),
                "dia_id": dia_id,
            })

    questions = []
    for qa_index, raw_qa in enumerate(validate_qa_rows(questions_with_raw_ids(sample), dialogue_id+".qa")):
        location = f"{dialogue_id}.qa[{raw_qa['question_id']}]"
        evidence = raw_qa.get("evidence") or []
        if isinstance(evidence, str):
            evidence = [evidence]
        if not isinstance(evidence,list):
            raise ValueError(f'{location}.evidence: expected a list or string')
        normalized_evidence = []
        for item in evidence:
            item_text = data_text(item, location+'.evidence item')
            # LoCoMo releases use several equivalent encodings: a list of
            # IDs, semicolon-separated IDs, and space-separated IDs.  Extract
            # the dialogue IDs uniformly so QA-to-turn linking is lossless.
            ids = re.findall(r"\bD\d+:\d+\b", item_text)
            if ids:
                normalized_evidence.extend(ids)
            else:
                normalized_evidence.extend(
                    part.strip() for part in item_text.split(";") if part.strip()
                )
        evidence = normalized_evidence
        answer = answer_text(raw_qa.get("answer"), location+'.answer')
        questions.append({
            "question_id": str(raw_qa.get("question_id", qa_index)),
            "question": data_text(raw_qa.get("question"), location+'.question'),
            "answer": answer,
            "evidence": evidence,
            "category": int(raw_qa.get("category", 0)),
        })

    participants = [data_text(value, dialogue_id+'.'+key) for key in ("speaker_a", "speaker_b")
                    if (value := conversation.get(key)) is not None]
    if not participants:
        participants = list(dict.fromkeys(turn["speaker"] for turn in turns if turn["speaker"]))
    return {
        "dialogue_id": dialogue_id,
        "participants": participants,
        "turns": turns,
        "questions": questions,
    }


def load_dialogues(payload: Any) -> list[dict[str, Any]]:
    samples = payload if isinstance(payload, list) else [payload]
    if not all(isinstance(sample, dict) for sample in samples):
        raise ValueError("Input must be raw LoCoMo JSON object or list of objects.")
    return [normalize_locomo_sample(sample) if "conversation" in sample else sample for sample in samples]


def validate_questions(dialogue):
    dialogue['questions'] = validate_qa_rows(dialogue.get('questions'),
                                            f"{dialogue['dialogue_id']}.questions")


class FactExtractor:
    def __init__(
        self,
        cache_path: str,
        model: str,
        base_url: str,
        api_key: str,
        max_tokens: int,
        max_retries: int,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.cache = read_json(cache_path) if self.cache_path.exists() else {}
        if not isinstance(self.cache,dict):
            raise ValueError(f'{cache_path}: expected a fact-cache object')
        self._facts_by_turn = {}
        for cached_key, value in self.cache.items():
            if cached_key.startswith("facts:"):
                record_key, separator, _ = cached_key.rpartition(":")
                if separator:
                    self._facts_by_turn.setdefault(record_key, value)
        LOGGER.info(
            "teacher=%s base_url=%s cache=%s cached_items=%d max_tokens=%d max_retries=%d",
            model,
            base_url,
            cache_path,
            len(self.cache),
            max_tokens,
            max_retries,
        )

    def _facts_record(self, key: str) -> Any | None:
        return self.cache.get(key, self._facts_by_turn.get(key))


    def _client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the openai package to call the vLLM server.") from exc

        return OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def _parse_json_content(self, content: str, label: str) -> dict[str, Any]:
        if "```json" in content:
            LOGGER.debug("JSON code fence detected: %s", label)
            content = content.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in content:
            LOGGER.debug("Generic code fence detected: %s", label)
            content = content.split("```", 1)[1].split("```", 1)[0]
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end < start:
            LOGGER.error("JSON object missing or truncated: %s response_tail=%r", label, content[-500:])
            raise ValueError("Teacher response does not contain a JSON object.")
        json_text = content[start : end + 1]
        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError:
            LOGGER.error("JSON parse failed: %s response_tail=%r", label, json_text[-500:])
            raise
        if not isinstance(payload, dict):
            raise ValueError("Teacher response JSON must be an object.")
        return payload

    def _json_completion(self, prompt: str, label: str) -> dict[str, Any]:
        """Call the OpenAI-compatible endpoint and retry malformed JSON responses."""
        client = self._client()
        for attempt in range(self.max_retries + 1):
            retry_prompt = prompt if attempt == 0 else prompt + COMPACT_JSON_INSTRUCTION
            LOGGER.info(
                "teacher request start: %s attempt=%d/%d prompt_chars=%d",
                label,
                attempt + 1,
                self.max_retries + 1,
                len(retry_prompt),
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": retry_prompt}],
                temperature=0,
                top_p=1,
                max_tokens=self.max_tokens,
                stream=False,
            )
            choice = response.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            # gpt-oss may expose chain-of-thought in reasoning_content. The
            # builder parses only the final answer in message.content.
            content = choice.message.content or ""
            if finish_reason == "length":
                LOGGER.warning(
                    "teacher response truncated: %s attempt=%d/%d content_chars=%d tail=%r",
                    label,
                    attempt + 1,
                    self.max_retries + 1,
                    len(content),
                    content[-300:],
                )
                error: Exception = ValueError("Teacher response reached max_tokens.")
            elif not content.strip():
                LOGGER.warning(
                    "teacher response empty: %s attempt=%d/%d finish_reason=%s reasoning_chars=%d",
                    label,
                    attempt + 1,
                    self.max_retries + 1,
                    finish_reason,
                    len(getattr(choice.message, "reasoning_content", "") or ""),
                )
                error = ValueError("Teacher returned empty final content.")
            else:
                LOGGER.info(
                    "teacher request done: %s attempt=%d/%d finish_reason=%s content_chars=%d",
                    label,
                    attempt + 1,
                    self.max_retries + 1,
                    finish_reason,
                    len(content),
                )
                try:
                    return self._parse_json_content(content, label)
                except (ValueError, json.JSONDecodeError) as exc:
                    error = exc

            if attempt < self.max_retries:
                LOGGER.warning(
                    "teacher request retrying with compact output: %s reason=%s",
                    label,
                    error,
                )
                continue
            raise error

        raise AssertionError("NVIDIA retry loop exited unexpectedly.")

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary_path.replace(self.cache_path)

    def extract_facts(self, dialogue_id: str, turn_index: int, turn: dict[str, Any]) -> list[dict[str, Any]]:
        """LLMExtract(di): the teacher extracts the turn's memory-relevant facts.

        Extraction is deterministic per turn and cached, so Algorithm 5 can
        consume the extracted facts across all rollout trajectories without
        re-running the teacher.
        """
        key = f"facts:{dialogue_id}:{turn_index}"
        record = self._facts_record(key)
        if record is not None:
            LOGGER.info("cache hit: facts dialogue=%s turn=%d", dialogue_id, turn_index)
            return validate_facts(record, f'{self.cache_path}:{key}')

        payload = self._json_completion(
            build_extract_input(turn),
            f"facts dialogue={dialogue_id} turn={turn_index}",
        )
        raw_facts = validate_facts(payload.get("memory_list"), f'teacher {key}.memory_list', 'value')

        facts = [
            {
                "speaker": str(turn.get("speaker", "")),
                "timestamp": str(turn.get("timestamp", "")),
                "key": data_text(fact.get("key") or "", f'{key}.key', allow_empty=True),
                "memory_type": data_text(fact.get("memory_type") or "", f'{key}.memory_type', allow_empty=True),
                "tags": fact.get("tags") or [],
                "text": fact["value"].strip(),
            }
            for fact in raw_facts
        ]

        self.cache[key] = facts
        self._save_cache()
        return facts



def build(
    input_path: str,
    output_path: str,
    cache_path: str,
    model: str,
    base_url: str,
    api_key: str,
    max_tokens: int,
    max_retries: int,
    split: str,
) -> None:
    dialogues = load_dialogues(select_locomo_split(read_json(input_path), split))
    builder = FactExtractor(cache_path, model, base_url, api_key, max_tokens, max_retries)
    rows = []
    LOGGER.info("loaded input=%s dialogues=%d", input_path, len(dialogues))

    for dialogue_index, dialogue in enumerate(dialogues, start=1):
        validate_questions(dialogue)
        dialogue_id = str(dialogue["dialogue_id"])
        turns = dialogue["turns"]
        if not isinstance(turns,list): raise ValueError(f'{dialogue_id}.turns: expected a list')
        for i,turn in enumerate(turns):
            location = f'{dialogue_id}.turns[{i}]'
            if not isinstance(turn,dict): raise ValueError(f'{location}: expected an object')
            data_text(turn.get('text',''),location+'.text',allow_empty=True)
            data_text(turn.get('speaker'),location+'.speaker')
        LOGGER.info(
            "dialogue start %d/%d id=%s turns=%d questions=%d",
            dialogue_index,
            len(dialogues),
            dialogue_id,
            len(turns),
            len(dialogue.get("questions", [])),
        )
        for turn_index, turn in enumerate(turns):
            LOGGER.info("facts progress dialogue=%s turn=%d/%d", dialogue_id, turn_index + 1, len(turns))
            turn["facts"] = (
                builder.extract_facts(dialogue_id, turn_index, turn)
                if turn.get("text", "").strip()
                else []
            )
        for turn_index, turn in enumerate(turns):
            turn['turn_index'] = turn_index
        rows.append(dict(dialogue_id=dialogue_id, participants=dialogue.get('participants', []),
                         dialogue_turns=turns, questions=dialogue.get('questions', []),
                         metadata=dict(fact_extractor=model)))
        LOGGER.info("dialogue done id=%s tuples=%d", dialogue_id, len(turns))

    write_jsonl(output_path, rows)
    LOGGER.info("wrote %d dialogues with cached facts to %s", len(rows), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract teacher facts; no teacher memory bank.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible vLLM endpoint (default: VLLM_BASE_URL or http://127.0.0.1:8000/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("VLLM_API_KEY", "EMPTY"),
        help="vLLM API key (default: VLLM_API_KEY or EMPTY)",
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--log-file", help="Optional path for a persistent build log")
    args = parser.parse_args()
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.max_retries < 0:
        parser.error("--max-retries cannot be negative")
    build(
        args.input,
        args.output,
        args.cache,
        args.model,
        args.base_url,
        args.api_key,
        args.max_tokens,
        args.max_retries,
        args.split,
    )


if __name__ == "__main__":
    main()
