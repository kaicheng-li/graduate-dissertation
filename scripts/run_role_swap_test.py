"""Run a Manager/Answer role-swap test with independent dialogues on four GPUs."""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from statistics import mean

from evaluation.metrics import aggregate_answer_metrics
from verl_memory.protocol import validate_timelines


ROOT = Path(__file__).resolve().parents[1]


def divide_dialogues(timelines, count):
    shards = [[] for _ in range(count)]
    loads = [0] * count
    capacity = (len(timelines) + count - 1) // count
    for row in sorted(timelines, key=lambda item: len(item['turns']), reverse=True):
        index = min((i for i in range(count) if len(shards[i]) < capacity),
                    key=lambda i: (loads[i], i))
        shards[index].append(row)
        loads[index] += len(row['turns'])
    return shards


def run_phase(phase, args, devices, output):
    print(f'ROLE_SWAP_PHASE start={phase} gpus={devices}', flush=True)
    running = []
    for index, device in enumerate(devices):
        shard = output / f'gpu_{index}'
        command = [sys.executable, '-m', 'evaluation.evaluate_role_swap',
                   '--phase', phase, '--timeline', str(shard / 'timeline.json'),
                   '--output-dir', str(shard),
                   '--max-prompt-tokens', str(args.max_prompt_tokens),
                   '--max-new-tokens', str(args.max_new_tokens),
                   '--gpu-memory-utilization', str(args.gpu_memory_utilization),
                   '--batch-size', str(args.batch_size)]
        for key in ('manager_model', 'manager_adapter', 'answer_model', 'answer_adapter'):
            value = getattr(args, key)
            if value:
                command.extend(('--' + key.replace('_', '-'), value))
        log_path = shard / f'{phase}.log'
        log = log_path.open('w', encoding='utf-8')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, PYTHONUNBUFFERED='1')
        process = subprocess.Popen(command, cwd=ROOT, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace', bufsize=1,
                                   start_new_session=True)
        def relay(stream, destination, label):
            try:
                for line in stream:
                    destination.write(line)
                    destination.flush()
                    print(f'[{label}] {line}', end='', flush=True)
            finally:
                stream.close()
                destination.close()
        thread = threading.Thread(target=relay, args=(process.stdout, log, f'{phase} gpu={device}'),
                                  daemon=True)
        thread.start()
        running.append((process, thread, log_path))
    failures = []
    try:
        for process, thread, path in running:
            code = process.wait()
            thread.join()
            if code:
                failures.append(str(path))
    except KeyboardInterrupt:
        print(f'ROLE_SWAP_PHASE interrupted={phase}; stopping GPU workers', flush=True)
        for process, _, _ in running:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process, thread, _ in running:
            process.wait()
            thread.join()
        raise
    if failures:
        raise RuntimeError(f'{phase} failed; inspect {failures}')
    print(f'ROLE_SWAP_PHASE complete={phase}', flush=True)


def merge_results(output, shards):
    banks = [bank for index in range(len(shards))
             for bank in json.loads((output / f'gpu_{index}' / 'manager_banks.json').read_text(encoding='utf-8'))]
    records = [json.loads(line) for index in range(len(shards))
               for line in (output / f'gpu_{index}' / 'results.jsonl').read_text(encoding='utf-8').splitlines()
               if line.strip()]
    (output / 'manager_banks.json').write_text(json.dumps(banks, ensure_ascii=False), encoding='utf-8')
    (output / 'results.jsonl').write_text(''.join(
        json.dumps(row, ensure_ascii=False) + '\n' for row in records), encoding='utf-8')
    summary = dict(questions=len(records),
                   **{key: mean(row[key] for row in records) for key in ('f1', 'em', 'bleu1')},
                   by_category=aggregate_answer_metrics(records, ('f1', 'em', 'bleu1')))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('ROLE_SWAP_SUMMARY ' + json.dumps(summary), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manager-model', required=True)
    parser.add_argument('--manager-adapter')
    parser.add_argument('--answer-model', required=True)
    parser.add_argument('--answer-adapter')
    parser.add_argument('--answer-only', action='store_true')
    parser.add_argument('--timeline', default='data/stage0/dialogue_timelines_test.json')
    parser.add_argument('--gpus', default=os.environ.get('TEST_GPUS', '0,1,2,3'))
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-prompt-tokens', type=int, default=4096)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.8)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args(argv)
    devices = [value.strip() for value in args.gpus.split(',') if value.strip()]
    output = Path(args.output_dir).resolve()
    timelines = validate_timelines(json.loads(Path(args.timeline).read_text(encoding='utf-8')))
    print('ROLE_SWAP_START ' + json.dumps(dict(manager_model=args.manager_model,
          manager_adapter=args.manager_adapter, answer_model=args.answer_model,
          answer_adapter=args.answer_adapter, gpus=devices, dialogues=len(timelines),
          output_dir=str(output))), flush=True)
    shards = divide_dialogues(timelines, len(devices))
    if not args.answer_only:
        for index, rows in enumerate(shards):
            shard = output / f'gpu_{index}'
            shard.mkdir(parents=True, exist_ok=True)
            (shard / 'timeline.json').write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
            print(f'ROLE_SWAP_SHARD gpu={devices[index]} '
                  f'dialogues={[r["dialogue_id"] for r in rows]}', flush=True)
        run_phase('manager', args, devices, output)
    run_phase('answer', args, devices, output)
    merge_results(output, shards)


if __name__ == '__main__':
    main()
