"""Shared retrieval and Manager JSON parsing used by every bank rollout."""
import json
import math
import os
import re
from typing import Any

_embedding_model = None
_embedding_cache = {}

def _embeddings(texts: list[str]):
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        model_name = os.environ.get("MEMORY_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        device = os.environ.get("MEMORY_EMBEDDING_DEVICE", "cpu")
        _embedding_model = SentenceTransformer(model_name, device=device)

    missing = list(dict.fromkeys(text for text in texts if text not in _embedding_cache))
    if missing:
        vectors = _embedding_model.encode(missing, normalize_embeddings=True, convert_to_numpy=True)
        _embedding_cache.update(zip(missing, vectors))
    return [_embedding_cache[text] for text in texts]

def top_k(query: str, memory_bank: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    if not memory_bank:
        return []
    texts = [query] + [str(memory.get("text", "")) for memory in memory_bank]
    vectors = _embeddings(texts)
    scores = [float(vector @ vectors[0]) for vector in vectors[1:]]
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    return [memory_bank[index] for index, _ in ranked[:k]]

def top_k_per_speaker(
    query: str,
    memory_bank: list[dict[str, Any]],
    participants: list[str],
    k: int,
) -> list[dict[str, Any]]:
    return [
        memory
        for participant in participants
        for memory in top_k(
            query,
            [entry for entry in memory_bank if entry["speaker"] == participant],
            k,
        )
    ]

def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-finite JSON value {value!r}")


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        _reject_constant(value)
    return result


def _object_fragment(text, start):
    """Locate one object and repair punctuation outside quoted strings only."""
    stack, output, repairs = [], [], []
    quoted = escaped = False
    skip_until = start
    for index in range(start, len(text)):
        if index < skip_until:
            continue
        char = text[index]
        if quoted:
            if ord(char) < 32 and not escaped:
                output.append(json.dumps(char)[1:-1])
                repairs.append(dict(kind="escape_control_character", offset=index))
            else:
                output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if text.startswith('//', index) or text.startswith('/*', index):
            line_comment = text.startswith('//', index)
            end = text.find('\n' if line_comment else '*/', index + 2)
            if end < 0 and not line_comment:
                return None, len(text), repairs
            skip_until = len(text) if end < 0 else end + (0 if line_comment else 2)
            output.append(' ')
            repairs.append(dict(kind='remove_comment', offset=index))
            continue
        # An explicit omitted array entry is not an operation. Keep all fully
        # specified actions; omitted bank entries remain unchanged in executor.
        marker = '...' if text.startswith('...', index) else '…' if char == '…' else None
        if marker and stack == ['}', ']'] and ''.join(output).rstrip().endswith(('[', ',')):
            end = index + len(marker)
            while end < len(text) and text[end].isspace():
                end += 1
            if end < len(text) and text[end] in ',]':
                skip_until = end + (text[end] == ',')
                repairs.append(dict(kind='skip_ellipsis_entry', offset=index))
                continue
        if char == '"':
            quoted = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or stack.pop() != char:
                return None, index + 1, repairs
            previous = len(output) - 1
            while previous >= 0 and output[previous].isspace():
                previous -= 1
            if previous >= 0 and output[previous] == ',':
                del output[previous]
                repairs.append(dict(kind="remove_trailing_comma", offset=index))
        elif text.startswith("```", index):
            # A closing code fence can follow otherwise complete JSON content.
            break
        output.append(char)
        if not stack:
            return "".join(output), index + 1, repairs
    if quoted or escaped or not stack:
        return None, len(text), repairs
    fragment = "".join(output).rstrip()
    # Never turn a dangling comma into a finished operation list. Missing keys,
    # values, colons, etc. also remain invalid under the JSON decoder below.
    if not fragment or fragment[-1] not in '"}]':
        return None, len(text), repairs
    suffix = "".join(reversed(stack))
    repairs.append(dict(kind="append_closing_delimiters", text=suffix))
    return fragment + suffix, len(text), repairs


def parse_manager_output(text: str, diagnostics: dict | None = None) -> list[Any]:
    """Recover bounded syntax errors, never fabricate action fields or text.

    Prose/fences surrounding an object are allowed. Multiple differing memory
    objects are ambiguous and rejected, rather than selecting an echoed example.
    The original output is retained by the shared transition/evidence logger.
    """
    if diagnostics is None:
        diagnostics = {}
    diagnostics.update(repaired=False, repairs=[])
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Memory Manager output is empty or not text.")
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object, parse_constant=_reject_constant,
                               parse_float=_finite_float)
    candidates = []
    index = 0
    while index < len(text):
        if text[index] != "{":
            index += 1
            continue
        start = index
        fragment, end, repairs = _object_fragment(text, start)
        try:
            payload = decoder.decode(fragment) if fragment is not None else None
        except json.JSONDecodeError:
            payload = None
        except RecursionError as exc:
            raise ValueError("Memory JSON nesting is too deep.") from exc
        except ValueError as exc:
            diagnostics['failure'] = str(exc)
            raise
        if isinstance(payload, dict) and "memory" in payload:
            memory = payload['memory']
            if not isinstance(memory, list):
                raise ValueError("memory must be a list")
            # Preserve bad entries and their indices for per-action rejection.
            # A null/string entry must not hide neighboring valid operations.
            candidates.append((memory, start, end, repairs))
            index = end
        elif re.match(r'\{\s*"memory"\s*:', text[start:]):
            raise ValueError("Memory JSON is incomplete or malformed; action content was not guessed.")
        else:
            # Also supports a memory field preceded by metadata in the object.
            index = start + 1
    if not candidates:
        raise ValueError("Memory Manager output does not contain a valid memory JSON object.")
    if any(item[0] != candidates[0][0] for item in candidates[1:]):
        raise ValueError("Multiple conflicting memory JSON objects in Manager output.")
    memory, start, end, _ = candidates[-1]
    repairs = [repair for item in candidates for repair in item[3]]
    diagnostics.update(repaired=bool(repairs), repairs=repairs,
                       object_start=start, object_end=end, object_count=len(candidates))
    return memory
