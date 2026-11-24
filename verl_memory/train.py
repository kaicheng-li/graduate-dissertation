"""Real verl FSDP/vLLM training with persistent Memory-R1 banks and frozen Answer."""
import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import sys
from time import perf_counter
import traceback

from verl_memory.protocol import (append_json, answer_prompt, block_advantages, fingerprint,
                                  load_bank, per_qa_advantages, validate_timelines)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def runtime_warning(root, component, error, **context):
    record = dict(component=component, error=str(error), **context)
    print('TRAINING_WARNING '+json.dumps(record, ensure_ascii=False), flush=True)
    try:
        append_json(Path(root)/'runtime_warnings.jsonl', record)
    except OSError as exc:
        print(f'Cannot write warning log: {exc}', flush=True)


def make_config(args, total_steps, gpu_count=4):
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf
    with initialize_config_module(config_module='verl.trainer.config', version_base=None):
        config = compose(config_name='ppo_trainer')
    settings = {
        'actor_rollout_ref.model.path': args.manager_model,
        'actor_rollout_ref.model.lora_rank': 8,
        'actor_rollout_ref.model.lora_alpha': 16,
        'actor_rollout_ref.model.target_modules': ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        'actor_rollout_ref.model.use_remove_padding': True,
        'actor_rollout_ref.model.enable_gradient_checkpointing': True,
        'actor_rollout_ref.actor.strategy': 'fsdp',
        'actor_rollout_ref.actor.fsdp_config.model_dtype': 'bfloat16',
        'actor_rollout_ref.actor.fsdp_config.param_offload': True,
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload': True,
        'actor_rollout_ref.actor.fsdp_config.use_orig_params': True,
        'actor_rollout_ref.actor.fsdp_config.seed': args.seed,
        'actor_rollout_ref.actor.use_dynamic_bsz': True,
        'actor_rollout_ref.actor.ppo_max_token_len_per_gpu': args.train_token_budget,
        'actor_rollout_ref.actor.ppo_mini_batch_size': args.num_generations,
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu': 1,
        'actor_rollout_ref.actor.ppo_epochs': 1,
        'actor_rollout_ref.actor.loss_agg_mode': 'seq-mean-token-sum',
        'actor_rollout_ref.actor.use_rollout_log_probs': True,
        'actor_rollout_ref.actor.use_kl_loss': bool(args.beta),
        'actor_rollout_ref.actor.kl_loss_coef': args.beta,
        'actor_rollout_ref.actor.kl_loss_type': 'low_var_kl',
        'actor_rollout_ref.actor.clip_ratio': .2,
        'actor_rollout_ref.actor.clip_ratio_low': .2,
        'actor_rollout_ref.actor.clip_ratio_high': .2,
        'actor_rollout_ref.actor.clip_ratio_c': 1e9,
        'actor_rollout_ref.actor.entropy_coeff': 0.,
        'actor_rollout_ref.actor.entropy_from_logits_with_chunking': True,
        'actor_rollout_ref.actor.optim.lr': args.learning_rate,
        'actor_rollout_ref.actor.optim.weight_decay': .01,
        'actor_rollout_ref.actor.optim.lr_warmup_steps': 0,
        'actor_rollout_ref.actor.optim.lr_warmup_steps_ratio': 0.,
        'actor_rollout_ref.actor.optim.clip_grad': 1.,
        'actor_rollout_ref.actor.policy_loss.loss_mode': 'vanilla',
        'actor_rollout_ref.actor.optim.lr_scheduler_type': 'constant',
        'actor_rollout_ref.actor.optim.total_training_steps': total_steps,
        'actor_rollout_ref.rollout.name': 'vllm',
        'actor_rollout_ref.rollout.mode': 'async',
        'actor_rollout_ref.rollout.dtype': 'bfloat16',
        'actor_rollout_ref.rollout.checkpoint_engine.backend': 'naive',
        'actor_rollout_ref.rollout.load_format': 'auto',
        'actor_rollout_ref.rollout.n': 1,
        'actor_rollout_ref.rollout.temperature': 1.,
        'actor_rollout_ref.rollout.tensor_model_parallel_size': 1,
        'actor_rollout_ref.rollout.gpu_memory_utilization': args.manager_memory_fraction,
        'actor_rollout_ref.rollout.prompt_length': args.max_prompt_tokens,
        'actor_rollout_ref.rollout.response_length': args.max_new_tokens,
        'actor_rollout_ref.rollout.max_model_len': args.max_prompt_tokens+args.max_new_tokens,
        'actor_rollout_ref.rollout.max_num_seqs': 8,
        'actor_rollout_ref.rollout.max_num_batched_tokens': 8192,
        'actor_rollout_ref.rollout.enable_prefix_caching': True,
        'actor_rollout_ref.rollout.enable_chunked_prefill': True,
        'actor_rollout_ref.rollout.enforce_eager': False,
        'actor_rollout_ref.rollout.cudagraph_capture_sizes': [1,2,4,8],
        'actor_rollout_ref.rollout.free_cache_engine': True,
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu': 1,
        'actor_rollout_ref.rollout.log_prob_use_dynamic_bsz': True,
        'actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu': args.train_token_budget,
        'actor_rollout_ref.nccl_timeout': 7200,
        'algorithm.adv_estimator': 'grpo',
        'data.max_prompt_length': args.max_prompt_tokens,
        'data.max_response_length': args.max_new_tokens,
        'trainer.n_gpus_per_node': gpu_count,
        'trainer.nnodes': 1,
        'trainer.total_epochs': args.epochs,
        'trainer.total_training_steps': total_steps,
    }
    settings['actor_rollout_ref.model.path'] = args.answer_model if args.task == 'answer' else args.manager_model
    adapter = args.answer_adapter if args.task == 'answer' else args.manager_adapter
    if adapter:
        settings['actor_rollout_ref.model.lora_adapter_path'] = str(Path(adapter).resolve())
        adapter_config = json.loads((Path(adapter)/'adapter_config.json').read_text(encoding='utf-8'))
        settings['actor_rollout_ref.model.lora_rank'] = adapter_config['r']
        settings['actor_rollout_ref.model.lora_alpha'] = adapter_config['lora_alpha']
        settings['actor_rollout_ref.model.target_modules'] = adapter_config['target_modules']
    for key, value in settings.items():
        OmegaConf.update(config, key, value, force_add=True)
    OmegaConf.resolve(config)
    return config


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    from memfactory.diagnostics import log_data_issue
    rows = []
    for index, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            log_data_issue(f'{path}:{index}', line, exc, 'skip_jsonl_record')
    return rows


def attach_linked_question_turns(timelines, linked_path):
    """Attach evidence turn metadata without changing the model-facing prompt."""
    linked_path = Path(linked_path)
    if not linked_path.is_file():
        print(f'TRAINING_WARNING linked-question data not found: {linked_path}; '
              'SF intermediate filtering will exclude questions without metadata', flush=True)
        return timelines
    index = {}
    for row in read_jsonl(linked_path):
        dialogue_id = row.get('dialogue_id')
        linked = row.get('linked_questions')
        if not isinstance(dialogue_id, str) or not isinstance(linked, list):
            continue
        for question in linked:
            if not isinstance(question, dict) or question.get('question_id') is None:
                continue
            key = (dialogue_id, str(question['question_id']))
            previous = index.setdefault(key, {'evidence': [], 'linked_turns': []})
            evidence = question.get('evidence')
            if isinstance(evidence, str):
                evidence = [evidence]
            if isinstance(evidence, list):
                previous['evidence'].extend(str(item) for item in evidence)
            if isinstance(question.get('turn_index'), int):
                previous['linked_turns'].append(question['turn_index'])

    for timeline in timelines:
        turn_lookup = {}
        for position, turn in enumerate(timeline.get('turns', [])):
            raw_id = str(turn.get('dia_id', ''))
            match = re.fullmatch(r'D(\d+):(\d+)', raw_id)
            if match:
                turn_lookup[f'D{int(match.group(1))}:{int(match.group(2))}'] = position
        for question in timeline.get('questions', []):
            key = (timeline['dialogue_id'], str(question.get('question_id')))
            metadata = index.get(key, {})
            evidence = metadata.get('evidence') or question.get('evidence') or []
            references = []
            for item in evidence:
                references.extend(re.findall(r'\bD(\d+):(\d+)\b', str(item)))
            references = sorted(set(references))
            evidence_turns = [turn_lookup[f'D{int(session)}:{int(turn)}']
                              for session, turn in references
                              if f'D{int(session)}:{int(turn)}' in turn_lookup]
            # A question becomes eligible only when every evidence reference is
            # known and has entered the bank. No turn-index guessing is allowed.
            question['evidence_turn_indices'] = sorted(set(evidence_turns))
            question['sf_ready_after_turn'] = (
                max(evidence_turns) + 1
                if references and len(evidence_turns) == len(references)
                else None)
    return timelines


def mean(values):
    values = list(values)
    return sum(values)/len(values) if values else None


def canonical_trajectory_id(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value) if value is not None else None


def visible_gpu_tokens(cuda_visible_devices, device_count):
    """Return the driver's CUDA device tokens for manually time-shared actors."""
    if cuda_visible_devices:
        return [token.strip() for token in cuda_visible_devices.split(',') if token.strip()]
    return [str(index) for index in range(device_count)]


def create_answer_workers(ray, worker_class, args, device_tokens):
    cls = ray.remote(num_gpus=0, num_cpus=2)(worker_class)
    return [cls.remote(args.answer_model, args.answer_adapter,
                       args.answer_memory_fraction,
                       args.max_prompt_tokens+args.max_new_tokens,
                       args.seed+index, device)
            for index, device in enumerate(device_tokens)]


def score_requests(workers, requests, args, root, label, gold=False,
                   wake_before=True, sleep_after=True):
    import ray
    result = []
    if not requests:
        return result
    # Keep a stable request identity in driver and worker logs without putting
    # prompt contents into the diagnostic stream.
    requests = [dict(request, _score_index=request.get('_score_index', index))
                for index, request in enumerate(requests)]
    # Teacher-forced prompt-logprob requests retain a logprob entry for every
    # scored token. Keep their RPC batches smaller than free-form QA batches;
    # this avoids a vLLM EngineCore spike when a large SF bank-score phase
    # follows a terminal QA phase.
    batch_size = (getattr(args, 'gold_batch_size', 1) if gold
                  else args.answer_batch_size)
    batch_size = max(1, min(batch_size, args.answer_batch_size))
    width = batch_size*len(workers)
    if wake_before:
        ray.get([worker.wake.remote() for worker in workers])
    failed = False
    try:
        for start in range(0, len(requests), width):
            batches = [requests[start+i*batch_size:start+(i+1)*batch_size]
                       for i in range(len(workers)) if start+i*batch_size < len(requests)]
            batch_meta = []
            for worker_index, batch in enumerate(batches):
                batch_meta.append(dict(worker=worker_index,
                    request_indices=[r.get('_score_index', start+worker_index*batch_size+j)
                                     for j,r in enumerate(batch)],
                    prompt_chars=[len(str(r.get('prompt',''))) for r in batch],
                    gold_chars=[len(str(r.get('gold',''))) for r in batch] if gold else None))
            append_json(Path(root)/'answer_batches.jsonl', dict(
                event='answer_batch', label=label, gold_score=gold, start=start,
                total=len(requests), batch_size=batch_size, workers=batch_meta))
            print(f'verl-answer batch stage={label} start={start} '
                  f'workers={batch_meta}', flush=True)
            jobs = [w.evaluate.remote(batch, gold) for w,batch in zip(workers,batches)]
            for batch in ray.get(jobs):
                result.extend(batch['rows'])
                append_json(Path(root)/'answer_performance.jsonl', dict(event=label,
                            **{k:v for k,v in batch.items() if k != 'rows'}))
            print(f'verl-answer stage={label} completed={min(start+width,len(requests))}/{len(requests)}',flush=True)
    except BaseException:
        failed = True
        raise
    finally:
        # Terminal and snapshot QA share one Answer engine lifecycle.
        if sleep_after or failed:
            try:
                ray.get([worker.sleep.remote() for worker in workers])
            except BaseException as exc:
                # Preserve the EngineCore traceback from the scoring RPC. A
                # dead worker cannot be put to sleep and must not mask it.
                if failed:
                    print(f'ANSWER_SLEEP_WARNING after scoring failure: {exc!r}', flush=True)
                else:
                    raise
    return result


def score_trajectories(trajectories, timeline, workers, args, root, epoch, group,
                       training=True, existing_answers=()):
    from scripts.construct_memory_bank import top_k_per_speaker
    from evaluation.metrics import f1_score, exact_match, bleu1_score
    from memfactory.diagnostics import EvidenceStore
    store = EvidenceStore(root)
    cached = {}
    for row in existing_answers:
        if row.get('dialogue_id') == timeline['dialogue_id']:
            tid = canonical_trajectory_id(row.get('trajectory_id'))
            cached.setdefault(tid, {})[str(row.get('question_id'))] = row
    new_answers = {t['trajectory_id']: {} for t in trajectories}
    requests, identities = [], []
    for trajectory in trajectories:
        trajectory_cached = cached.get(canonical_trajectory_id(trajectory['trajectory_id']), {})
        for q in timeline['questions']:
            if str(q['question_id']) in trajectory_cached:
                continue
            prompt, retrieved = answer_prompt(q['question'],trajectory['bank'],timeline['participants'],top_k_per_speaker)
            requests.append(dict(prompt=prompt,max_prompt_tokens=args.max_prompt_tokens,max_new_tokens=args.max_new_tokens))
            identities.append((trajectory,q,retrieved))
    # Reuse the same greedy decoding path for terminal and snapshot QA.
    sf_phase = training and args.kind == 'sf' and args.aux_lambda != 0
    answer_awake = bool(requests) and sf_phase
    outputs = score_requests(workers,requests,args,root,'terminal_qa',
                             sleep_after=not answer_awake)
    generated = {}
    for request, output, (t,q,retrieved) in zip(requests,outputs,identities,strict=True):
        key = hashlib.sha256(json.dumps([request['prompt'],q['answer']],ensure_ascii=False).encode()).hexdigest()
        generated[key] = output
        row = dict(dialogue_id=timeline['dialogue_id'],question_id=q['question_id'],question=q['question'],
            gold=q['answer'],category=q.get('category'),**output,
            f1=f1_score(output['prediction'],q['answer']),em=exact_match(output['prediction'],q['answer']),
            bleu1=bleu1_score(output['prediction'],q['answer']),answer_words=len(output['prediction'].split()),
            memory_bank_ref=store.save(t['bank']),evidence_root=str(Path(root).resolve()),
            answer_evidence_ref=store.save(dict(**request,retrieved_memories=retrieved,**output),'answer'))
        new_answers[t['trajectory_id']][str(q['question_id'])] = row
        record = dict(epoch=epoch,group=group,trajectory_id=t['trajectory_id'],**row)
        append_json(Path(root)/'answers.jsonl',record)
        if not row['prediction'] or row['hit_token_limit']:
            append_json(Path(root)/'answer_errors.jsonl',dict(error='empty_prediction' if not row['prediction'] else 'token_limit',**record))
    for trajectory in trajectories:
        trajectory_cached = cached.get(canonical_trajectory_id(trajectory['trajectory_id']), {})
        trajectory['answers'] = [trajectory_cached.get(str(q['question_id'])) or
                                 new_answers[trajectory['trajectory_id']][str(q['question_id'])]
                                 for q in timeline['questions']]
    if not training:
        return None,[]
    terminal = per_qa_advantages([[q['f1'] for q in t['answers']] for t in trajectories])
    advantages = {t['trajectory_id']:{turn['turn_index']:a for turn in timeline['turns']}
                  for t,a in zip(trajectories,terminal)}
    credit = []
    if args.kind == 'sf' and args.aux_lambda != 0:
        unique, locations = {},[]
        for t in trajectories:
            for point in t['bank_points']:
                bank = load_bank(root,point['memory_bank_ref'])
                point['scores'] = []
                point['scores_by_question'] = {}
                final_bank = point['after_turn'] == len(timeline['turns'])
                eligible = [q for q in timeline['questions']
                            if final_bank or (q.get('sf_ready_after_turn') is not None and
                                              q['sf_ready_after_turn'] <= point['after_turn'])]
                for q in eligible:
                    prompt,retrieved = answer_prompt(q['question'],bank,timeline['participants'],top_k_per_speaker)
                    key = hashlib.sha256(json.dumps([prompt,q['answer']],ensure_ascii=False).encode()).hexdigest()
                    unique.setdefault(key,dict(prompt=prompt,max_prompt_tokens=args.max_prompt_tokens,
                                               max_new_tokens=args.max_new_tokens))
                    locations.append((t['trajectory_id'],point,q,key,retrieved))
        pending = {key:request for key,request in unique.items() if key not in generated}
        if pending:
            outputs = score_requests(workers,list(pending.values()),args,root,'sf_f1',
                                     wake_before=not answer_awake, sleep_after=True)
            generated.update(zip(pending,outputs,strict=True))
            answer_awake = False
        elif answer_awake:
            # score_requests returns early for an empty request list, so close
            # the deferred Answer phase explicitly when no QA is eligible.
            import ray
            ray.get([worker.sleep.remote() for worker in workers])
            answer_awake = False
        for tid,point,q,key,retrieved in locations:
            output = generated[key]
            score = f1_score(output['prediction'],q['answer'])
            point['scores'].append(score)
            point['scores_by_question'][str(q['question_id'])] = score
            append_json(Path(root)/'bank_scores.jsonl',dict(epoch=epoch,group=group,dialogue_id=timeline['dialogue_id'],
                trajectory_id=tid,after_turn=point['after_turn'],question_id=q['question_id'],**output,
                question=q['question'],gold=q['answer'],f1=score,score_metric='generated_f1',
                memory_bank_ref=point['memory_bank_ref'],score_evidence_ref=store.save(
                    dict(**unique[key],gold=q['answer'],f1=score,score_metric='generated_f1',
                         retrieved_memories=retrieved,**output),'bank_score')))
        for t,a in zip(trajectories,terminal):
            for point in t['bank_points']:
                point['phi'] = mean(point['scores']) if point['scores'] else None
            values,blocks = block_advantages(t['bank_points'],a,len(timeline['turns']),
                args.credit_interval,args.aux_lambda,args.aux_tau)
            advantages[t['trajectory_id']] = values
            credit.extend(dict(trajectory_id=t['trajectory_id'],score_metric='generated_f1',**b) for b in blocks)
    return advantages,credit


def score_test_dialogues(states, workers, args, root, epoch):
    """Score all test dialogues in shared Answer batches, reusing saved QA rows."""
    from scripts.construct_memory_bank import top_k_per_speaker
    from evaluation.metrics import f1_score, exact_match, bleu1_score
    from memfactory.diagnostics import EvidenceStore

    requests, identities = [], []
    cached_by_state = {}
    answers_by_state = {}
    for state in states:
        timeline, trajectory = state['timeline'], state['trajectory']
        cached = {str(row.get('question_id')): row
                  for row in state['existing_answers']
                  if row.get('dialogue_id') == timeline['dialogue_id'] and
                  canonical_trajectory_id(row.get('trajectory_id')) ==
                  canonical_trajectory_id(trajectory['trajectory_id'])}
        cached_by_state[id(state)] = cached
        answers_by_state[id(state)] = {}
        for question in timeline['questions']:
            qid = str(question['question_id'])
            if qid in cached:
                continue
            prompt, retrieved = answer_prompt(question['question'], trajectory['bank'],
                timeline['participants'], top_k_per_speaker)
            requests.append(dict(prompt=prompt, max_prompt_tokens=args.max_prompt_tokens,
                                 max_new_tokens=args.max_new_tokens))
            identities.append((state, question, retrieved))

    outputs = score_requests(workers, requests, args, root, 'terminal_qa')
    for request, output, (state, question, retrieved) in zip(requests, outputs, identities, strict=True):
        timeline, trajectory = state['timeline'], state['trajectory']
        store = EvidenceStore(state['root'])
        row = dict(dialogue_id=timeline['dialogue_id'], question_id=question['question_id'],
            question=question['question'], gold=question['answer'], category=question.get('category'),
            **output, f1=f1_score(output['prediction'], question['answer']),
            em=exact_match(output['prediction'], question['answer']),
            bleu1=bleu1_score(output['prediction'], question['answer']),
            answer_words=len(output['prediction'].split()),
            memory_bank_ref=store.save(trajectory['bank']),
            evidence_root=str(Path(state['root']).resolve()),
            answer_evidence_ref=store.save(dict(**request, retrieved_memories=retrieved,
                                                **output), 'answer'))
        answers_by_state[id(state)][str(question['question_id'])] = row
        record = dict(epoch=epoch, group=state['group'],
                      trajectory_id=trajectory['trajectory_id'], **row)
        append_json(Path(state['root'])/'answers.jsonl', record)
        if not row['prediction'] or row['hit_token_limit']:
            append_json(Path(state['root'])/'answer_errors.jsonl', dict(
                error='empty_prediction' if not row['prediction'] else 'token_limit', **record))

    all_answers = []
    trajectories = []
    for state in states:
        trajectory = state['trajectory']
        cached = cached_by_state[id(state)]
        generated = answers_by_state[id(state)]
        trajectory['answers'] = [cached.get(str(question['question_id'])) or
                                 generated[str(question['question_id'])]
                                 for question in state['timeline']['questions']]
        all_answers.extend(trajectory['answers'])
        trajectories.append(trajectory)
    return all_answers, trajectories


def weight_loss_mask(batch, generations):
    """With seq-mean-token-sum, give every trajectory equal token-mean weight."""
    totals = defaultdict(float)
    mask = batch.batch['response_mask'].float()
    ids = batch.non_tensor_batch['trajectory_id'].tolist()
    for tid,n in zip(ids,mask.sum(-1).tolist()): totals[tid] += n
    for i,tid in enumerate(ids): mask[i] *= len(ids)/(generations*totals[tid])
    batch.batch['response_mask'] = mask


def answer_loss_mask(batch, _generations):
    mask = batch.batch['response_mask'].float()
    batch.batch['response_mask'] = mask/mask.sum(-1,keepdim=True)


def apply_update(actor, batch, advantages, generations, beta, mask_fn):
    """A zero final advantage skips ALL policy/reference/optimizer/scheduler calls."""
    result = dict(skipped_update=not any(v != 0 for v in advantages),entropy=None,
                  log_probability_seconds=0.,update_seconds=0.,metrics={})
    if result['skipped_update']:
        result['skip_reason'] = 'all_final_advantages_zero'
        return result
    import numpy as np
    import torch
    batch.meta_info.update(temperature=1.,global_token_num=batch.batch['attention_mask'].sum(-1).tolist())
    tick = perf_counter()
    old = actor.compute_log_prob(batch)
    batch.batch['old_log_probs'] = old.batch['old_log_probs']
    binary = batch.batch['response_mask'].clone()
    result['entropy'] = float((old.batch['entropys']*binary).sum()/binary.sum())
    if beta:
        batch.batch['ref_log_prob'] = actor.compute_ref_log_prob(batch).batch['ref_log_prob']
    batch.batch['advantages'] = torch.tensor(advantages,dtype=torch.float32)[:,None].expand_as(batch.batch['responses']).clone()
    result['log_probability_seconds'] = perf_counter()-tick
    mask_fn(batch,generations)
    actor.set_group_size(len(batch))
    tick = perf_counter()
    updated = actor.update_actor(batch)
    result['update_seconds'] = perf_counter()-tick
    result['metrics'] = {k:float(np.mean(v)) for k,v in updated.meta_info['metrics'].items()}
    nonfinite = {k:str(v) for k,v in result['metrics'].items() if not __import__('math').isfinite(v)}
    # Preserve non-finite metrics as explicit strings plus JSON nulls.
    # Logging does not change optimizer execution or decide whether to continue.
    if nonfinite:
        result['nonfinite_metrics'] = nonfinite
        for key in nonfinite:
            result['metrics'][key] = None
    return result


def qa_summary(records, trajectories=()):
    from evaluation.metrics import aggregate_answer_metrics
    return dict(qa=len(records),f1=mean(r['f1'] for r in records),em=mean(r['em'] for r in records),
        bleu1=mean(r['bleu1'] for r in records),answer_words=mean(len(r['prediction'].split()) for r in records),
        answer_output_tokens=mean(r['output_tokens'] for r in records),
        answer_token_limit_count=sum(r['hit_token_limit'] for r in records),
        invalid_turns=sum(t['invalid_turns'] for t in trajectories),
        repaired_turns=sum(t.get('repaired_turns', 0) for t in trajectories),
        **{key:sum(t.get(key,0) for t in trajectories) for key in
           ('action_repaired_turns','partial_applied_turns','parse_error_turns','applied_actions','rejected_actions')},
        turns=sum(t['turns'] for t in trajectories),
        paper_metrics=aggregate_answer_metrics(records,('em','f1','bleu1','llm_judge')))


def plot_run(root):
    try:
        from scripts.diagnose_rl import render
        render(root)
    except Exception as exc:
        runtime_warning(root, 'plot', repr(exc), traceback=traceback.format_exc())
    finally:
        # A failed render must not leak open figures across subsequent epochs.
        if 'matplotlib.pyplot' in sys.modules:
            sys.modules['matplotlib.pyplot'].close('all')


def adapter_at(checkpoint):
    checkpoint = Path(checkpoint)
    adapter = checkpoint/'lora_adapter'
    if not any((adapter/name).is_file() for name in ('adapter_model.safetensors', 'adapter_model.bin')):
        runtime_warning(checkpoint.parent, 'adapter_export',
                        f'No exported LoRA weights at {adapter}; keep the saved FSDP checkpoint.',
                        checkpoint=str(checkpoint))
        return None
    return adapter


def validate_answer_rows(rows):
    from memfactory.locomo_split import data_text, validate_qa_rows
    from memfactory.diagnostics import log_data_issue
    valid = []
    for row in validate_qa_rows(rows, 'answer_rows', 'prompt', 'gold'):
        try:
            data_text(row.get('dialogue_id'), 'answer_rows.dialogue_id')
        except ValueError as exc:
            log_data_issue('answer_rows', row, exc, 'skip_qa')
            continue
        if not isinstance(row.get('retrieved_memories'), list):
            log_data_issue('answer_rows.retrieved_memories', row.get('retrieved_memories'),
                           'expected a memory list; prompt is already prepared', 'use_empty_metadata')
            row = dict(row, retrieved_memories=[])
        valid.append(row)
    return valid


def load_data(args):
    if args.task == 'answer':
        train,val = [validate_answer_rows(read_jsonl(p)) for p in (args.train_answer_data,args.validation_answer_data)]
    elif args.task == 'test':
        return [],validate_timelines(read_json(args.timeline))
    else:
        train,val = [validate_timelines(read_json(p)) for p in (args.train_timelines,args.validation_timelines)]
        attach_linked_question_turns(train, args.manager_training_data)
        attach_linked_question_turns(val, args.manager_training_data)
    if {r['dialogue_id'] for r in train} & {r['dialogue_id'] for r in val}:
        print('TRAINING_WARNING training and validation contain overlapping dialogue IDs', flush=True)
    return train,val


def prepare_data(actor, datasets, args, root, actor_gpus):
    from scripts.construct_memory_bank import top_k_per_speaker
    from memfactory.diagnostics import EvidenceStore
    for split,timelines in zip(('train','validation'),datasets):
        target = root/f'answer_{split}.jsonl'
        for group,timeline in enumerate(timelines):
            trace = root/'banks'/split/timeline['dialogue_id']
            partial = actor.build_banks(timeline,actor_gpus,str(trace),0,group,0,True)
            # All ranks participate in FSDP weight synchronization; keep one greedy bank.
            t = min((t for p in partial for t in p['trajectories']),key=lambda t:t['trajectory_id'])
            store = EvidenceStore(trace)
            for q in timeline['questions']:
                prompt,retrieved = answer_prompt(q['question'],t['bank'],timeline['participants'],top_k_per_speaker)
                append_json(target,dict(dialogue_id=timeline['dialogue_id'],question_id=q['question_id'],
                    question=q['question'],gold=q['answer'],category=q.get('category'),prompt=prompt,
                    retrieved_memories=retrieved,memory_bank_ref=store.save(t['bank']),evidence_root=str(trace.resolve())))
        print(f'verl-answer-data split={split} path={target}',flush=True)


def _test_progress(root, group, timeline, epoch):
    """Find one recoverable trajectory for a dialogue in an existing trace."""
    from memfactory.diagnostics import EvidenceStore
    base = Path(root)/'evaluation'/f'epoch_{epoch}'/str(group)
    attempts = sorted(base.glob('attempt_*'),
                      key=lambda p: int(p.name.rsplit('_', 1)[-1]))
    if not attempts:
        attempts = [base/'attempt_0']

    def ref_file(trace, row, key):
        ref = row.get(key)
        if not isinstance(ref, dict) or not ref.get('path'):
            return None
        path = trace/ref['path']
        return ref if path.is_file() else None

    candidates = []
    for trace in attempts:
        rows = defaultdict(lambda: defaultdict(list))
        for path in sorted(trace.glob('trajectories_rank_*.jsonl')):
            for row in read_jsonl(path):
                if row.get('event') != 'trajectory_turn':
                    continue
                try:
                    turn = int(row['turn'])
                except (KeyError, TypeError, ValueError):
                    continue
                tid = canonical_trajectory_id(row.get('trajectory_id'))
                if tid is not None:
                    rows[tid][turn].append(row)

        # Follow before/after hashes where available. This prevents combining
        # divergent rank logs into a bank that was never generated.
        for tid, by_turn in rows.items():
            prefix = []
            previous_hash = None
            for turn in range(len(timeline['turns'])):
                options = by_turn.get(turn, [])
                selected = None
                for row in options:
                    before = ref_file(trace, row, 'bank_before_ref')
                    after = ref_file(trace, row, 'memory_bank_ref')
                    if after is None:
                        continue
                    if turn == 0:
                        if before is None or before.get('sha256') == EvidenceStore(trace).save([]).get('sha256'):
                            selected = row
                            break
                    elif before is not None and before.get('sha256') == previous_hash:
                        selected = row
                        break
                if selected is None:
                    break
                prefix.append(selected)
                previous_hash = selected['memory_bank_ref'].get('sha256')
            candidates.append((len(prefix), int(trace.name.rsplit('_', 1)[-1]),
                               tid, trace, prefix))

    def id_sort(value):
        return (0, value) if isinstance(value, int) else (1, str(value))

    usable = [item for item in candidates if item[0] > 0]
    if usable:
        _, _, trajectory_id, trace, prefix = sorted(
            usable, key=lambda item: (-item[0], -item[1], id_sort(item[2])))[0]
    else:
        # The trace may contain only malformed/incomplete records. Continue in
        # a fresh attempt without deleting the evidence.
        trace = attempts[-1]
        seen = {item[2] for item in candidates}
        trajectory_id = 0
        while trajectory_id in seen:
            trajectory_id += 1
        prefix = []

    trajectory = dict(trajectory_id=trajectory_id, bank=[], bank_points=[], answers=[],
        invalid_turns=0, repaired_turns=0, action_repaired_turns=0,
        partial_applied_turns=0, parse_error_turns=0, applied_actions=0,
        rejected_actions=0, turns=len(timeline['turns']))
    if prefix:
        trajectory['bank'] = load_bank(trace, prefix[-1]['memory_bank_ref'])
        for row in prefix:
            trajectory['invalid_turns'] += int(not row.get('valid', True))
            trajectory['repaired_turns'] += int(row.get('parsing', {}).get('repaired', False))
            trajectory['action_repaired_turns'] += int(bool(row.get('action_repairs')))
            trajectory['partial_applied_turns'] += int(row.get('partial_applied', False))
            trajectory['parse_error_turns'] += int(str(row.get('error') or '').startswith('parse:'))
            trajectory['applied_actions'] += row.get('actions', 0)
            trajectory['rejected_actions'] += len(row.get('rejected_actions', []))

    answers_path = trace/'answers.jsonl'
    answers = read_jsonl(answers_path) if answers_path.is_file() else []
    answers = [row for row in answers
               if row.get('dialogue_id') == timeline['dialogue_id']
               and canonical_trajectory_id(row.get('trajectory_id')) == trajectory_id]
    print(f'TEST_RESUME group={group} dialogue={timeline["dialogue_id"]} '
          f'committed_turns={len(prefix)}/{len(timeline["turns"])} '
          f'cached_answers={len(answers)}/{len(timeline["questions"])} trace={trace}', flush=True)
    return dict(timeline=timeline, trajectory=trajectory, root=trace, group=group,
                start_turn=len(prefix), existing_answers=answers)


def evaluation(actor, frozen, val, args, root, epoch, actor_gpus):
    qa,trajectories = [],[]
    if args.task == 'test':
        progress = [_test_progress(root, group, row, epoch)
                    for group, row in enumerate(val)]
        pending = [state for state in progress
                   if state['start_turn'] < len(state['timeline']['turns'])]
        if pending:
            actor.build_banks_many(pending, epoch, True)
        qa, trajectories = score_test_dialogues(progress, frozen, args, root, epoch)
        summary = qa_summary(qa, trajectories)
        write_json(root/'evaluation'/f'epoch_{epoch}.json',
                   dict(epoch=epoch, component=args.task,
                        roll_forward=dict(summary=summary, records=qa)))
        print('VERL_VALIDATION '+json.dumps(dict(epoch=epoch, **summary)), flush=True)
        return summary, qa

    for group,row in enumerate(val if args.task != 'answer' else [val[i:i+args.answer_batch_size]
                                                           for i in range(0,len(val),args.answer_batch_size)]):
        trace = root/'evaluation'/f'epoch_{epoch}'/str(group)
        attempt = 0
        while (trace/f'attempt_{attempt}').exists(): attempt += 1
        trace = trace/f'attempt_{attempt}'
        if args.task == 'answer':
            partial = actor.sample_answers(row,actor_gpus,str(trace),epoch,group,True)
            # One greedy completion per QA; discard redundant FSDP rank replicas.
            qa.extend(r for p in partial for r in p['records'] if r['candidate'] == 0)
        else:
            partial = actor.build_banks(row,actor_gpus,str(trace),epoch,group,0,True)
            t = min((t for p in partial for t in p['trajectories']),key=lambda t:t['trajectory_id'])
            score_trajectories([t],row,frozen,args,trace,epoch,group,training=False)
            trajectories.append(t)
            qa.extend(t['answers'])
    summary = qa_summary(qa,trajectories)
    mode = 'answer' if args.task == 'answer' else 'roll_forward'
    write_json(root/'evaluation'/f'epoch_{epoch}.json',dict(epoch=epoch,component=args.task,
                **{mode:dict(summary=summary,records=qa)}))
    print('VERL_VALIDATION '+json.dumps(dict(epoch=epoch,**summary)),flush=True)
    return summary,qa


def finish_epoch(actor, frozen, val, args, root, epoch, actor_gpus, step, best, contract):
    """Commit completed training before running the potentially lengthy validation."""
    checkpoint = root/f'epoch_{epoch}'
    actor.save_checkpoint(str(checkpoint), None, step, None)
    state = dict(epoch=epoch, global_step=step, best=best, contract=contract,
                 validation_status='pending' if val else 'disabled')
    write_json(checkpoint/'training_state.json', state)
    adapter = adapter_at(checkpoint)
    print(f'VERL_CHECKPOINT epoch={epoch} path={checkpoint} adapter_ready={adapter is not None}', flush=True)
    if val:
        try:
            summary,_ = evaluation(actor, frozen, val, args, root, epoch, actor_gpus)
        except Exception as exc:
            state['validation_status'] = 'failed'
            state['validation_error'] = repr(exc)
            write_json(checkpoint/'training_state.json', state)
            runtime_warning(root, 'validation', repr(exc), epoch=epoch, traceback=traceback.format_exc())
            raise
        state['validation_status'] = 'completed'
        score = summary['f1']
        if score is not None and score > best:
            best = score
            write_json(root/'best_checkpoint.json', dict(
                epoch=epoch, path=str(adapter or checkpoint), f1=best, component=args.task,
                checkpoint_type='lora_adapter' if adapter is not None else 'fsdp'))
        state['best'] = best
        write_json(checkpoint/'training_state.json', state)
    return best


def resume_state(args, root, data_paths):
    """Epoch-boundary verl checkpoints only; keep interrupted attempts as evidence."""
    fields = ('task','kind','seed','num_generations','learning_rate','beta','credit_interval',
              'aux_lambda','aux_tau','manager_model','manager_adapter','answer_model','answer_adapter',
              'max_prompt_tokens','max_new_tokens','qa_batch_size','answer_batch_size','gold_batch_size',
              'train_token_budget','smoke_turns')
    expected = dict(settings={key:getattr(args,key,None) for key in fields},prompts=fingerprint(),
                    auxiliary_score='generated_f1',
                    data={str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in data_paths})
    if args.resume_from:
        state = read_json(Path(args.resume_from)/'training_state.json')
        if state['contract'] != expected:
            print('TRAINING_WARNING resume configuration or data differs from the saved run; using current arguments', flush=True)
        if Path(args.resume_from).resolve().parent != root:
            print('TRAINING_WARNING resuming into a different output directory', flush=True)
        if (root/'best_checkpoint.json').exists():
            best = read_json(root/'best_checkpoint.json')
            if best['epoch'] > state['epoch']:
                print('TRAINING_WARNING a later completed epoch exists; using the requested checkpoint', flush=True)
        # Archive log tails, preserving errors/banks from the interrupted epoch.
        stamp = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
        for path in root.glob('*.jsonl'):
            retained,interrupted = [],[]
            for r in read_jsonl(path):
                (retained if r.get('epoch',0) <= state['epoch'] else interrupted).append(r)
            if interrupted:
                archive=root/'interrupted'/stamp/path.name
                for r in interrupted: append_json(archive,r)
                path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in retained),encoding='utf-8')
        for path in (root/'evaluation').glob('epoch_*.json'):
            if read_json(path)['epoch'] > state['epoch']:
                archive = root/'interrupted'/stamp/'evaluation'/path.name
                archive.parent.mkdir(parents=True,exist_ok=True)
                path.replace(archive)
        return state
    if root.exists() and any(root.iterdir()):
        print('TRAINING_WARNING output directory is not empty; existing output files may be overwritten', flush=True)
    return dict(epoch=0,global_step=0,best=-1.,contract=expected)


def run(args):
    import numpy as np
    import torch
    if platform.system() != 'Linux' or not torch.cuda.is_available():
        print('TRAINING_WARNING Linux CUDA was not detected; backend initialization will handle the environment', flush=True)
    backend_versions = {name: importlib.metadata.version(name)
                        for name in ('verl', 'vllm', 'torch', 'transformers')}
    print('Training backend versions: '+json.dumps(backend_versions), flush=True)
    if backend_versions['verl'] != '0.9.1':
        print('WARNING: running with unverified backend versions; dependency and API '
              'compatibility have not been established.', flush=True)
    if torch.cuda.device_count() != 4:
        print('TRAINING_WARNING this configuration requests four GPUs; available GPUs differ', flush=True)
    import ray
    from omegaconf import OmegaConf
    from verl import DataProto
    from verl.single_controller.ray import RayClassWithInitArgs,RayResourcePool,RayWorkerGroup
    from verl_memory.workers import MemoryActorWorker,MemoryBackend,FrozenAnswerWorker
    paths = ([args.train_answer_data,args.validation_answer_data] if args.task == 'answer' else
             [args.timeline] if args.task == 'test' else [args.train_timelines,args.validation_timelines])
    root = Path(args.output_dir).resolve()
    state = resume_state(args,root,paths)
    root.mkdir(parents=True,exist_ok=True)
    os.environ['MEMORY_DATA_ISSUES_DIR'] = str(root/'data_issues')
    train,val = load_data(args)
    if (args.task == 'test' and not val) or (args.task != 'test' and not train):
        runtime_warning(root, 'data', 'No usable data remains; training/evaluation was not started')
        return
    if args.smoke_turns:
        train=[dict(train[0],turns=train[0]['turns'][:args.smoke_turns],questions=train[0]['questions'][:2])]
        val=[]
        args.epochs=1
    groups = len(train) if args.task != 'answer' else (len(train)+args.qa_batch_size-1)//args.qa_batch_size
    answer_gpu_tokens = visible_gpu_tokens(os.environ.get('CUDA_VISIBLE_DEVICES'),
                                           torch.cuda.device_count())
    actor_gpus = len(answer_gpu_tokens)
    config=make_config(args,max(1,args.epochs*groups),actor_gpus)
    from scripts.construct_memory_bank import _embeddings
    _embeddings(['Memory retrieval startup'])
    write_json(root/('resume_config.json' if args.resume_from else 'run_config.json'),dict(
        args=vars(args),backend='verl=='+backend_versions['verl'],backend_versions=backend_versions,
        config=OmegaConf.to_container(config,resolve=True),
        precision='BF16 + vLLM + FSDP LoRA',actor_gpus=actor_gpus,
        gpu_schedule='four_gpu_phase_shared',answer_gpu_tokens=answer_gpu_tokens,
        initial_bank='empty',qa='after_final_turn',prompts=fingerprint(),
        reward='per_QA_population_standardization_then_mean' if args.task != 'answer' else 'per_QA_population_standardization',
        auxiliary_score='generated_f1' if args.task == 'manager' and args.kind == 'sf' else None,
        reference='base model with LoRA disabled',old_policy='fixed pre-update teacher-forced token log probabilities',
        epoch_zero_validation=False,intermediate_checkpoints=False))
    ray.init(runtime_env={'env_vars':dict(PYTHONPATH=str(Path(__file__).resolve().parents[1]),
        HF_HUB_OFFLINE=os.environ.get('HF_HUB_OFFLINE','1'),
        MEMORY_EMBEDDING_MODEL=os.environ.get('MEMORY_EMBEDDING_MODEL','sentence-transformers/all-MiniLM-L6-v2'),
        MEMORY_DATA_ISSUES_DIR=str(root/'data_issues'),
        MEMORY_EMBEDDING_DEVICE='cpu',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')})
    try:
        pool=RayResourcePool([actor_gpus],use_gpu=True,max_colocate_count=1,name_prefix='memory_actor')
        actor_group=RayWorkerGroup(resource_pool=pool,ray_cls_with_init=RayClassWithInitArgs(
            ray.remote(MemoryActorWorker),config=config.actor_rollout_ref,role='actor_rollout'))
        actor=MemoryBackend(actor_group,config.actor_rollout_ref,args.seed)
        actor.init_model()
        if args.resume_from:
            actor.load_checkpoint(str(Path(args.resume_from).resolve()),None,False)
            actor.weight_step=state['global_step']
        frozen=[]
        if args.task in ('manager', 'test'):
            # Manager workers retain Ray's four-GPU allocation. Their vLLM engines
            # sleep during Answer scoring, so Answer actors borrow those devices
            # without reserving a second set of Ray GPU resources.
            frozen=create_answer_workers(ray,FrozenAnswerWorker,args,answer_gpu_tokens)
            ray.get([worker.ready.remote() for worker in frozen])
        if args.task == 'prepare':
            prepare_data(actor,(train,val),args,root,actor_gpus)
            return
        if args.task == 'test':
            summary,records=evaluation(actor,frozen,val,args,root,0,actor_gpus)
            target=Path(args.results_file) if args.results_file else root/'results.jsonl'
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
            write_json(target.with_suffix('.json'),summary)
            return
        print(f'verl-training component={args.task} kind={args.kind} actor_gpus={actor_gpus} '
              f'answer_workers={len(frozen)} answer_gpu_tokens={answer_gpu_tokens} '
              f'gpu_schedule=four_gpu_phase_shared '
              f'groups_per_epoch={groups} initial_validation=off credit_interval={args.credit_interval} '
              f'lr={args.learning_rate} resume_epoch={state["epoch"]}',flush=True)
        step,best=state['global_step'],state['best']
        for epoch in range(state['epoch']+1,args.epochs+1):
            order=list(range(len(train)))
            random.Random(args.seed+epoch).shuffle(order)
            batches=([order[i:i+args.qa_batch_size] for i in range(0,len(order),args.qa_batch_size)]
                     if args.task == 'answer' else [[i] for i in order])
            for group,indices in enumerate(batches):
                tick=perf_counter()
                trace=root/'epochs'/f'epoch_{epoch}'/f'group_{group}'
                # A resumed incomplete epoch writes a fresh trace attempt; old evidence remains.
                attempt=0
                while (trace/f'attempt_{attempt}').exists(): attempt+=1
                trace=trace/f'attempt_{attempt}'
                credit=[]
                if args.task == 'answer':
                    rows=[train[i] for i in indices]
                    partial=actor.sample_answers(rows,args.num_generations,str(trace),epoch,group)
                    records=[r for p in partial for r in p['records']]
                    per_question={}
                    for i,row in enumerate(rows):
                        values=[next(r['f1'] for r in records if r['dialogue_id'] == row['dialogue_id']
                                     and str(r['question_id']) == str(row['question_id']) and r['candidate'] == c)
                                for c in range(args.num_generations)]
                        per_question[i]=per_qa_advantages([[v] for v in values])
                    actions=DataProto.concat([p['actions'] for p in partial])
                    advantages=[per_question[int(i)][int(c)] for i,c in zip(
                        actions.non_tensor_batch['row'],actions.non_tensor_batch['candidate'])]
                    rewards=[r['f1'] for r in records]
                    stats=qa_summary(records)
                    signal=mean(float(max(r)-min(r)>1e-12) for r in [[
                        next(r['f1'] for r in records if r['dialogue_id']==row['dialogue_id'] and
                             str(r['question_id'])==str(row['question_id']) and r['candidate']==c)
                        for c in range(args.num_generations)] for row in rows])
                else:
                    timeline=train[indices[0]]
                    interval=args.credit_interval if args.kind == 'sf' and args.aux_lambda else 0
                    partial=actor.build_banks(timeline,args.num_generations,str(trace),epoch,group,interval)
                    trajectories=sorted([t for p in partial for t in p['trajectories']],key=lambda t:t['trajectory_id'])
                    actions_list=[p['actions'] for p in partial if p['actions'] is not None]
                    if not actions_list:
                        runtime_warning(root, 'empty_group', 'No Manager actions; skip update', epoch=epoch, group=group)
                        append_json(root/'training_metrics.jsonl', dict(component=args.task, epoch=epoch, group=group,
                            global_step=step, skipped_update=True, skip_reason='no_manager_actions',
                            total_seconds=perf_counter()-tick))
                        continue
                    actions=DataProto.concat(actions_list)
                    adv,credit=score_trajectories(trajectories,timeline,frozen,args,trace,epoch,group)
                    advantages=[adv[int(tid)][int(turn)] for tid,turn in zip(
                        actions.non_tensor_batch['trajectory_id'],actions.non_tensor_batch['turn'])]
                    records=[q for t in trajectories for q in t['answers']]
                    rewards=[mean(q['f1'] for q in t['answers']) for t in trajectories]
                    stats=qa_summary(records,trajectories)
                    signal=mean(float(max(v)-min(v)>1e-12) for v in zip(*[[r['f1'] for r in t['answers']] for t in trajectories]))
                sampling_scoring_seconds=perf_counter()-tick
                output_tokens=int(actions.batch['response_mask'].sum())
                prompt_tokens=int(actions.batch['attention_mask'].sum())-output_tokens
                lengths=actions.batch['response_mask'].sum(-1)
                result=apply_update(actor,actions,advantages,args.num_generations,args.beta,
                                    answer_loss_mask if args.task == 'answer' else weight_loss_mask)
                if result.get('nonfinite_metrics'):
                    runtime_warning(root, 'nonfinite_metrics', result['nonfinite_metrics'], epoch=epoch, group=group)
                step+=int(not result['skipped_update'])
                finite_advantages = all(__import__('math').isfinite(v) for v in advantages)
                record=dict(component=args.task,epoch=epoch,group=group,attempt=attempt,global_step=step,
                    reward=mean(rewards),trajectory_rewards=rewards,signal_ratio=signal,
                    advantage_mean=mean(advantages) if finite_advantages else None,
                    advantage_abs_mean=mean(abs(v) for v in advantages) if finite_advantages else None,
                    advantage_min=min(advantages) if finite_advantages else None,
                    advantage_max=max(advantages) if finite_advantages else None,
                    output_tokens=output_tokens,prompt_tokens=prompt_tokens,output_tokens_mean=float(lengths.float().mean()),
                    token_limit_count=int((lengths>=args.max_new_tokens).sum()),
                    sampled_sequences=len(actions),sampling_scoring_seconds=sampling_scoring_seconds,
                    total_seconds=perf_counter()-tick,credit_blocks=credit,evidence_root=str(trace.resolve()),
                    **stats,**result)
                append_json(root/'training_metrics.jsonl',record)
                print('VERL_GROUP '+json.dumps({k:v for k,v in record.items() if k not in ('credit_blocks','paper_metrics')}),flush=True)
                del actions,partial
            best=finish_epoch(actor,frozen,val,args,root,epoch,actor_gpus,step,best,state['contract'])
        plot_run(root)
    except BaseException as exc:
        append_json(root/'runtime_errors.jsonl',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        ray.shutdown()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=['manager','answer','prepare','test'],default='manager')
    p.add_argument('--kind',choices=['sa','sf'],default='sa')
    p.add_argument('--manager-model')
    p.add_argument('--manager-adapter')
    p.add_argument('--answer-model')
    p.add_argument('--answer-adapter')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--train-timelines',default='data/stage0/dialogue_timelines_train.json')
    p.add_argument('--validation-timelines',default='data/stage0/dialogue_timelines_validation.json')
    p.add_argument('--train-answer-data',default='data/answer_verl/answer_train.jsonl')
    p.add_argument('--validation-answer-data',default='data/answer_verl/answer_validation.jsonl')
    p.add_argument('--timeline',default='data/stage0/dialogue_timelines_test.json')
    p.add_argument('--manager-training-data',default='data/manager_training_data_vllm.jsonl',
                   help='Per-turn linked_questions metadata used by SF intermediate scoring')
    p.add_argument('--results-file')
    p.add_argument('--resume-from',help='Completed epoch_N directory in the same output-dir; includes FSDP optimizer/RNG')
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--num-generations',type=int,default=4)
    p.add_argument('--qa-batch-size',type=int,default=4)
    p.add_argument('--learning-rate',type=float,default=1e-6)
    p.add_argument('--beta',type=float,default=.01)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-prompt-tokens',type=int,default=4096)
    p.add_argument('--max-new-tokens',type=int,default=2048)
    p.add_argument('--credit-interval',type=int,default=4)
    p.add_argument('--aux-lambda',type=float,default=.1)
    p.add_argument('--aux-tau',type=float,default=.5)
    p.add_argument('--answer-batch-size',type=int,default=16)
    p.add_argument('--gold-batch-size',type=int,default=4,
                   help='Legacy teacher-forced scoring batch size; unused by generated-F1 SF')
    p.add_argument('--train-token-budget',type=int,default=8192)
    p.add_argument('--manager-memory-fraction',type=float,default=.60)
    p.add_argument('--answer-memory-fraction',type=float,default=.65)
    p.add_argument('--check-data-only',action='store_true')
    p.add_argument('--smoke-turns',type=int,default=0,help='Manager engineering check only; no quality claim')
    args=p.parse_args(argv)
    divisor=4
    if (args.num_generations<2 or args.num_generations%divisor or args.epochs<1 or args.answer_batch_size<1
        or args.gold_batch_size<1
        or args.qa_batch_size<1 or args.credit_interval<1 or args.aux_tau<=0 or args.aux_lambda<0
        or args.learning_rate<=0 or args.beta<0 or args.smoke_turns<0
        or args.train_token_budget<args.max_prompt_tokens+args.max_new_tokens):
        print('TRAINING_WARNING unusual group size, scoring interval, hyperparameter or token budget; using supplied arguments', flush=True)
    if args.smoke_turns and args.task != 'manager': print('TRAINING_WARNING --smoke-turns was specified for a non-Manager task', flush=True)
    if args.check_data_only:
        train,val=load_data(args)
        for split,rows in [('train',train),('validation',val)]:
            print(json.dumps(dict(split=split,rows=len(rows),turns=sum(len(r.get('turns',[])) for r in rows),
                                 qa=sum(len(r.get('questions',[r])) for r in rows)),ensure_ascii=False))
        return
    if not args.answer_model: args.answer_model=args.manager_model
    if args.task != 'answer' and not args.manager_model: print('TRAINING_WARNING --manager-model was not supplied', flush=True)
    if args.task in ('manager','answer','test') and not args.answer_model: print('TRAINING_WARNING --answer-model was not supplied', flush=True)
    if args.task == 'prepare' and not args.manager_adapter: print('TRAINING_WARNING --manager-adapter was not supplied for Answer data preparation', flush=True)
    run(args)


if __name__ == '__main__': main()
