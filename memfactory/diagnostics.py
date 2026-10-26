"""Immutable compressed evidence plus incremental, portable trace references."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone


def log_data_issue(location, value, error, action='skip_record'):
    """Record recoverable source-data defects without stopping valid work."""
    record = dict(event='data_issue', location=location, action=action,
                  error=str(error), raw=value)
    line = json.dumps(record, ensure_ascii=False, default=repr)
    print('DATA_ISSUE '+line, flush=True)
    directory = os.environ.get('MEMORY_DATA_ISSUES_DIR')
    if directory:
        try:
            path = Path(directory)/f'data_issues_pid_{os.getpid()}.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a', encoding='utf-8') as handle:
                handle.write(line+'\n')
        except OSError as exc:
            print(f'DATA_ISSUE log write failed: {exc}', flush=True)


def log_manager_parse_error(path, state, action, **context):
    """Persist errors and syntax/action repairs with the original output."""
    action_repairs = action.get('action_repairs', [])
    repaired = action.get('parsing', {}).get('repaired', False) or bool(action_repairs)
    valid = action.get('valid', True)
    if valid and not repaired:
        return
    error_type = (('action_repair' if action_repairs else 'syntax_repair') if valid else
                  'parse' if str(action.get('error') or '').startswith('parse:') else 'action_validation')
    path = Path(path)
    if valid:
        path = path.with_name(path.name.replace('manager_errors', 'manager_repairs'))
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = action.get('raw', '')
    start = raw.find('{')
    diagnosis = {'first_object_offset': start}
    if start >= 0:
        try:
            payload, end = json.JSONDecoder().raw_decode(raw, start)
            diagnosis.update(first_object_type=type(payload).__name__, first_object_end=end)
            if isinstance(payload, dict):
                diagnosis.update(first_object_keys=list(payload),
                                 memory_field_type=type(payload.get('memory')).__name__)
        except json.JSONDecodeError as exc:
            diagnosis.update(message=exc.msg, position=exc.pos, line=exc.lineno, column=exc.colno)
    generation = action.get('generation')
    generation = generation if isinstance(generation, dict) else None
    event = (f'manager_{error_type}' if valid else
             'manager_parse_error' if error_type == 'parse' else 'manager_action_error')
    record = dict(event=event, valid=valid,
        error_type=error_type, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        rank=int(os.environ.get('RANK', '0')), **context,
        dialogue_id=state.get('dialogue_id'),
        source_turn=state.get('turn_index'),
        question_id=state.get('question_id'), error=action.get('error'), raw_output=raw,
        prompt=action.get('prompt'), facts=action.get('facts'),
        visible_memories=action.get('visible_memories'), generation=generation,
        decisions=action.get('decisions'), timing=action.get('timing'),
        applied_actions=action.get('actions', 0),
        rejected_ids=action.get('rejected_ids', []),
        rejected_actions=action.get('rejected_actions', []), action_repairs=action_repairs,
        id_mappings=action.get('id_mappings', []),
        partial_applied=action.get('partial_applied', False),
        parsing=action.get('parsing', {}), json_diagnostics=diagnosis)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    print(f"{event} dialogue={record['dialogue_id']} turn={record['source_turn']} "
          f"output_tokens={(generation or {}).get('output_tokens')} "
          f"hit_token_limit={(generation or {}).get('hit_token_limit')} "
          f"applied_actions={record['applied_actions']} partial_applied={record['partial_applied']} "
          f"rejected_ids={record['rejected_ids']} error={record['error']!r} "
          f"repairs={[r['kind'] for r in action_repairs]} details={path}", flush=True)


class EvidenceStore:
    def __init__(self, root):
        self.root = Path(root)

    def save(self, value, kind="memory_bank"):
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        relative = Path("evidence") / kind / digest[:2] / (digest + ".json.gz")
        path = self.root / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            # Concurrent ranks may write the same object; replacement is atomic.
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
                pending = Path(handle.name)
                handle.write(gzip.compress(raw, mtime=0))
            try:
                os.replace(pending, path)
            finally:
                pending.unlink(missing_ok=True)
        return dict(path=relative.as_posix(), sha256=digest)

