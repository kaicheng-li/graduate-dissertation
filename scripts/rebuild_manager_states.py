"""Build chronological dialogue timelines from cached teacher facts; no policy calls."""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from memfactory.locomo_split import SPLIT_DIALOGUE_IDS, select_locomo_split
from memfactory.modules.memory_extractor import validate_facts
from scripts.build_manager_training_data import normalize_locomo_sample
EVIDENCE_PATTERN = re.compile(r"\bD(\d+):(\d+)\b")

def normalize_evidence(value: Any) -> list[str]:
    """Extract and canonicalize evidence IDs ('D30:05' -> 'D30:5')."""
    items = value if isinstance(value, list) else [value]
    ids = []
    for item in items:
        for session, turn in EVIDENCE_PATTERN.findall(str(item)):
            ids.append(f"D{int(session)}:{int(turn)}")
    return ids

def flatten_dialogue(sample: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int], list[str]]:
    """Flatten LoCoMo sessions to a turn list; map canonical dia_id -> index."""
    normalized = normalize_locomo_sample(dict(sample, qa=[]))
    turns = normalized['turns']
    dia_to_index = {}
    for index, turn in enumerate(turns):
        canonical = normalize_evidence(turn['dia_id'])
        turn['turn_index'] = index
        if canonical:
            turn['dia_id'] = canonical[0]
            if canonical[0] in dia_to_index:
                raise ValueError(f"{normalized['dialogue_id']}: duplicate turn ID {canonical[0]}")
            dia_to_index[canonical[0]] = index
    return turns, dia_to_index, normalized['participants']

def load_cached_facts(jsonl_path: Path, dialogue_id: str) -> dict[int, list[dict[str, Any]]]:
    """Recover per-turn teacher facts from the existing Algorithm-1 JSONL."""
    facts_by_turn: dict[int, list[dict[str, Any]]] = {}
    with jsonl_path.open(encoding="utf-8") as handle:
        for line_number,line in enumerate(handle,1):
            if not line.strip():
                continue
            row = json.loads(line)
            location = f'{jsonl_path}:{line_number}'
            if not isinstance(row,dict):
                raise ValueError(f'{location}: expected an object, got {row!r}')
            if str(row.get("dialogue_id")) != dialogue_id:
                continue
            turns = row.get('dialogue_turns')
            if not isinstance(turns,list): raise ValueError(f'{location}.dialogue_turns: expected a list')
            for index,turn in enumerate(turns):
                if not isinstance(turn,dict): raise ValueError(f'{location}.dialogue_turns[{index}]: expected an object')
                turn_index = int(turn.get("turn_index", -1))
                if turn_index < 0: raise ValueError(f'{location}.dialogue_turns[{index}]: invalid turn_index')
                facts = validate_facts(turn.get('facts'), f'{location}.turn[{turn_index}].facts')
                if turn_index in facts_by_turn and facts_by_turn[turn_index] != facts:
                    raise ValueError(f'{location}: conflicting cached facts for {dialogue_id} turn {turn_index}')
                facts_by_turn[turn_index] = facts
    return facts_by_turn

def rebuild_split(samples, facts_path, split):
    timelines = []
    for sample in select_locomo_split(samples, split):
        normalized = normalize_locomo_sample(sample)
        did = normalized['dialogue_id']
        turns, _, participants = flatten_dialogue(sample)
        facts = load_cached_facts(Path(facts_path), did)
        for t in turns:
            if t['text'].strip() and t['turn_index'] not in facts:
                raise ValueError(f"Missing cached facts: {did} turn {t['turn_index']}")
            t['facts'] = facts.get(t['turn_index'], [])
        questions = normalized['questions']
        timelines.append(dict(dialogue_id=did, participants=participants, turns=turns, questions=questions))
    if {t['dialogue_id'] for t in timelines} != set(SPLIT_DIALOGUE_IDS[split]):
        raise ValueError(f'Missing dialogues for {split}')
    return timelines

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',required=True)
    p.add_argument('--facts-jsonl',default='data/manager_training_data_vllm.jsonl')
    p.add_argument('--output-dir',default='data/stage0')
    p.add_argument('--splits',nargs='+',choices=list(SPLIT_DIALOGUE_IDS),default=['train','validation'])
    args=p.parse_args()
    samples=json.loads(Path(args.raw).read_text(encoding='utf-8'))
    root=Path(args.output_dir); root.mkdir(parents=True,exist_ok=True)
    for split in args.splits:
        rows=rebuild_split(samples,args.facts_jsonl,split)
        target=root/f'dialogue_timelines_{split}.json'
        target.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'{split}: {len(rows)} dialogues, {sum(len(t["turns"]) for t in rows)} turns, '
              f'{sum(len(t["questions"]) for t in rows)} QA -> {target}',flush=True)

if __name__=='__main__': main()
