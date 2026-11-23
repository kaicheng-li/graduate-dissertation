"""Test full inference models or LoRA adapters in either memory-agent role.

Run --phase manager first, then --phase answer with the same --output-dir.
This path uses the shared bank transition, retrieval and QA metrics without
initializing the training-only FSDP actor.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from threading import Event, Thread
from time import monotonic

from evaluation.manager_validation import manager_transition
from evaluation.metrics import (aggregate_answer_metrics, bleu1_score, exact_match,
                                extract_answer, f1_score)
from memfactory.chat import tokenize_chat_prompt
from scripts.construct_memory_bank import top_k, top_k_per_speaker
from verl_memory.protocol import answer_prompt, prompt_for_turn, validate_timelines


def final_text(raw: str, is_gpt_oss: bool) -> str:
    marker = '<|channel|>final<|message|>'
    if marker not in raw:
        # Never mistake GPT-OSS analysis for a Manager action or a QA answer.
        return '' if is_gpt_oss else raw
    value = raw.rsplit(marker, 1)[1]
    for ending in ('<|return|>', '<|end|>', '<|eot_id|>', '<|im_end|>'):
        value = value.split(ending, 1)[0]
    return value.strip()


def load_engine(model: str, adapter: str | None, max_model_len: int,
                gpu_memory_utilization: float, tensor_parallel_size: int):
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.truncation_side = 'left'
    is_gpt_oss = AutoConfig.from_pretrained(model).model_type == 'gpt_oss'
    generation = GenerationConfig.from_pretrained(model)
    sampling_defaults = dict(top_k=getattr(generation, 'top_k', 50) or -1,
                             top_p=getattr(generation, 'top_p', None) or 1.,
                             repetition_penalty=getattr(generation, 'repetition_penalty', None) or 1.)
    options = dict(model=model, dtype='bfloat16', tensor_parallel_size=tensor_parallel_size,
                   max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                   enable_prefix_caching=False, enable_chunked_prefill=is_gpt_oss,
                   enable_lora=bool(adapter), generation_config='auto')
    request = None
    if adapter:
        config = json.loads((Path(adapter) / 'adapter_config.json').read_text(encoding='utf-8'))
        rank = max(8, 2 ** (int(config['r']) - 1).bit_length())
        options['max_lora_rank'] = rank
        request = LoRARequest('role_swap', 1, str(Path(adapter).resolve()))
    return LLM(**options), tokenizer, request, is_gpt_oss, sampling_defaults


def generate(llm, tokenizer, adapter, is_gpt_oss, sampling_defaults, prompts: list[str],
             max_prompt_tokens: int, max_new_tokens: int, batch_size: int,
             role: str):
    from vllm import SamplingParams

    options = dict(temperature=0., max_tokens=max_new_tokens,
                   repetition_penalty=sampling_defaults['repetition_penalty'],
                   skip_special_tokens=not is_gpt_oss)
    if role == 'manager':
        options.update(top_k=sampling_defaults['top_k'], top_p=sampling_defaults['top_p'])
        if not is_gpt_oss and tokenizer.eos_token_id is not None:
            options['stop_token_ids'] = [tokenizer.eos_token_id]
    sampling = SamplingParams(**options)
    result = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        inputs = [dict(prompt_token_ids=tokenize_chat_prompt(
            tokenizer, prompt, max_prompt_tokens)['input_ids'][0].tolist())
            for prompt in chunk]
        outputs = llm.generate(inputs, sampling, lora_request=adapter, use_tqdm=False)
        result.extend(final_text(output.outputs[0].text, is_gpt_oss) for output in outputs)
    return result


def manager_phase(args, timelines, root: Path, status):
    status['stage'] = 'loading_manager_model'
    llm, tokenizer, adapter, is_gpt_oss, sampling_defaults = load_engine(
        args.manager_model, args.manager_adapter, args.max_prompt_tokens + args.max_new_tokens,
        args.gpu_memory_utilization, args.tensor_parallel_size)
    states = [dict(timeline=timeline, bank=[], invalid=0) for timeline in timelines]
    max_turns = max(len(timeline['turns']) for timeline in timelines)
    for turn_index in range(max_turns):
        status['stage'] = f'manager_turn_{turn_index + 1}/{max_turns}'
        print(f'ROLE_SWAP_MANAGER_TURN start={turn_index + 1}/{max_turns}', flush=True)
        active = []
        prompts = []
        for state in states:
            timeline = state['timeline']
            if turn_index >= len(timeline['turns']):
                continue
            turn = timeline['turns'][turn_index]
            prompt = prompt_for_turn(turn, state['bank'], top_k)
            active.append((state, turn, prompt))
            if prompt is not None:
                prompts.append(prompt)
        outputs = iter(generate(llm, tokenizer, adapter, is_gpt_oss, sampling_defaults,
                                prompts, args.max_prompt_tokens, args.max_new_tokens,
                                max(1, len(prompts)), 'manager') if prompts else [])
        for state, turn, prompt in active:
            raw = next(outputs) if prompt is not None else ''
            def manager_generate(_prompt):
                return raw
            timeline = state['timeline']
            state['bank'], action = manager_transition(
                turn | {'dialogue_id': timeline['dialogue_id']},
                state['bank'], manager_generate, 5)
            state['invalid'] += int(not action['valid'])
        print(f'ROLE_SWAP_MANAGER_PROGRESS turn={turn_index + 1}/{max_turns} '
              f'active_dialogues={len(active)} generated={len(prompts)}', flush=True)
    records = []
    for state in states:
        timeline = state['timeline']
        bank = state['bank']
        invalid = state['invalid']
        records.append(dict(dialogue_id=timeline['dialogue_id'], bank=bank,
                            invalid_turns=invalid, turns=len(timeline['turns'])))
        print(f"ROLE_SWAP_MANAGER dialogue={timeline['dialogue_id']} "
              f"bank_size={len(bank)} invalid_turns={invalid}", flush=True)
    (root / 'manager_banks.json').write_text(
        json.dumps(records, ensure_ascii=False), encoding='utf-8')


def answer_phase(args, timelines, root: Path, status):
    banks = {row['dialogue_id']: row for row in json.loads(
        (root / 'manager_banks.json').read_text(encoding='utf-8'))}
    status['stage'] = 'loading_answer_model'
    llm, tokenizer, adapter, is_gpt_oss, sampling_defaults = load_engine(
        args.answer_model, args.answer_adapter, args.max_prompt_tokens + args.max_new_tokens,
        args.gpu_memory_utilization, args.tensor_parallel_size)
    records = []
    for timeline in timelines:
        status['stage'] = f"answer_dialogue_{timeline['dialogue_id']}"
        print(f"ROLE_SWAP_ANSWER start_dialogue={timeline['dialogue_id']} "
              f"questions={len(timeline['questions'])}", flush=True)
        bank = banks[timeline['dialogue_id']]['bank']
        prompts = [answer_prompt(q['question'], bank, timeline['participants'],
                                 top_k_per_speaker)[0] for q in timeline['questions']]
        raw_answers = generate(llm, tokenizer, adapter, is_gpt_oss, sampling_defaults, prompts,
                               args.max_prompt_tokens, args.max_new_tokens, args.batch_size, 'answer')
        for q, raw in zip(timeline['questions'], raw_answers, strict=True):
            prediction = extract_answer(raw)
            records.append(dict(dialogue_id=timeline['dialogue_id'], question_id=q['question_id'],
                                question=q['question'], gold=q['answer'], category=q.get('category'),
                                prediction=prediction, raw_answer=raw,
                                f1=f1_score(prediction, q['answer']),
                                em=exact_match(prediction, q['answer']),
                                bleu1=bleu1_score(prediction, q['answer'])))
        (root / 'results.jsonl').write_text(''.join(
            json.dumps(row, ensure_ascii=False) + '\n' for row in records), encoding='utf-8')
        print(f"ROLE_SWAP_ANSWER dialogue={timeline['dialogue_id']} "
              f"questions={len(timeline['questions'])}", flush=True)
    summary = dict(questions=len(records), **{key: mean(row[key] for row in records)
                                             for key in ('f1', 'em', 'bleu1')},
                   by_category=aggregate_answer_metrics(records, ('f1', 'em', 'bleu1')))
    (root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('ROLE_SWAP_SUMMARY ' + json.dumps(summary), flush=True)


def heartbeat(phase, status, stopped):
    started = monotonic()
    while not stopped.wait(10):
        print(f"ROLE_SWAP_HEARTBEAT phase={phase} stage={status['stage']} "
              f'elapsed_seconds={int(monotonic() - started)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('manager', 'answer'), required=True)
    parser.add_argument('--timeline', default='data/stage0/dialogue_timelines_test.json')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--manager-model')
    parser.add_argument('--manager-adapter')
    parser.add_argument('--answer-model')
    parser.add_argument('--answer-adapter')
    parser.add_argument('--max-prompt-tokens', type=int, default=4096)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.8)
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    required = 'manager_model' if args.phase == 'manager' else 'answer_model'
    if not getattr(args, required):
        parser.error(f'--{required.replace("_", "-")} is required for {args.phase} phase')
    os.environ.setdefault('MEMORY_EMBEDDING_DEVICE', 'cpu')
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    timelines = validate_timelines(json.loads(Path(args.timeline).read_text(encoding='utf-8')))
    status = {'stage': 'starting'}
    stopped = Event()
    reporter = Thread(target=heartbeat, args=(args.phase, status, stopped), daemon=True)
    reporter.start()
    try:
        if args.phase == 'manager':
            manager_phase(args, timelines, root, status)
        else:
            answer_phase(args, timelines, root, status)
    finally:
        stopped.set()
        reporter.join()


if __name__ == '__main__':
    main()
