"""verl 0.9 model-engine training and native vLLM server rollouts."""
import asyncio
import json
import os
from functools import partial
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch
import ray
from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import set_expandable_segments
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

from memfactory.chat import format_chat_prompt
from memfactory.diagnostics import EvidenceStore
from verl_memory.protocol import append_json, commit_turn, prompt_for_turn
from evaluation.metrics import extract_answer, f1_score, exact_match, bleu1_score


def generation_settings(config):
    for key, default in [('typical_p', 1.), ('no_repeat_ngram_size', 0),
                         ('min_length', 0), ('min_new_tokens', None), ('num_beams', 1)]:
        value = getattr(config, key, default)
        if value not in (default, None, 0) and not (key == 'num_beams' and value == 1):
            print(f'GENERATION_WARNING vLLM does not support {key}={value}; ignored', flush=True)
    return dict(top_k=getattr(config, 'top_k', 50) or -1,
                top_p=getattr(config, 'top_p', 1.),
                repetition_penalty=getattr(config, 'repetition_penalty', 1.))


def trajectory_policy_loss(config, model_output, data, dp_group=None):
    """Use verl's clipped policy loss while retaining trajectory token weights.

    New verl casts response_mask to bool. Pass the independent weights through
    its policy-loss weight argument and apply the same weights to the KL term.
    This is the existing objective, not an importance-sampling correction.
    """
    from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
    from verl.utils.metric import AggregationType, Metric
    log_prob = no_padding_2_padding(model_output['log_probs'], data)
    config.global_batch_info.update(dp_size=data['dp_size'],
        batch_num_tokens=data['batch_num_tokens'], global_batch_size=data['global_batch_size'],
        loss_scale_factor=config.loss_scale_factor)
    fields = ['response_mask', 'old_log_probs', 'advantages', 'memory_loss_weights']
    if config.use_kl_loss:
        fields.append('ref_log_prob')
    padded = data.select(*fields).to_padded_tensor()
    mask, weights = padded['response_mask'].bool(), padded['memory_loss_weights']
    loss, metrics = get_policy_loss_fn('vanilla')(
        old_log_prob=padded['old_log_probs'], log_prob=log_prob,
        advantages=padded['advantages'], response_mask=mask,
        loss_agg_mode='seq-mean-token-sum', config=config, rollout_is_weights=weights)
    metrics = Metric.from_dict(metrics, aggregation=AggregationType.MEAN)
    metrics['actor/pg_loss'] = Metric(value=loss, aggregation=AggregationType.SUM)
    if config.use_kl_loss:
        kl = kl_penalty(logprob=log_prob, ref_logprob=padded['ref_log_prob'],
                        kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(kl*weights, mask, 'seq-mean-token-sum', **config.global_batch_info)
        metrics['actor/kl_loss'] = Metric(value=kl_loss, aggregation=AggregationType.SUM)
        loss = loss + config.kl_loss_coef*kl_loss
    return loss, metrics


def lora_from_named_parameters(model):
    """Export this project's linear LoRA inside FSDP's full-parameter gather.

    PEFT's structural selector can compare wrapped module paths with cleaned
    state keys and discard every tensor. Select the actual A/B parameters,
    then write the standard PEFT checkpoint names without consulting wrappers.
    """
    peft_model = getattr(model, '_fsdp_wrapped_module', model)
    parameters = {name.replace('_fsdp_wrapped_module.', ''): value
                  for name, value in peft_model.named_parameters()}
    state = {}
    for name, value in parameters.items():
        for side in ('A', 'B'):
            suffix = f'.lora_{side}.default.weight'
            if name.endswith(suffix):
                prefix = name[:-len(suffix)]
                state[f'{prefix}.lora_{side}.weight'] = value
                break
    # Own the storage: FSDP restores shards when its gather context exits.
    state = {name: value.detach().cpu().contiguous().clone() for name, value in state.items()}
    return state, peft_model.peft_config['default'].to_dict()


class MemoryActorWorker(ActorRolloutRefWorker):
    """Project loss and explicit LoRA export; model and optimizer stay in verl."""

    def _export_lora(self, stage):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        engine = self.actor.engine
        engine.to('cuda', model=True, optimizer=False, grad=False)
        try:
            with FSDP.summon_full_params(engine.module, recurse=True, writeback=False):
                state, config = lora_from_named_parameters(engine.module)
        finally:
            if engine.is_param_offload_enabled:
                engine.to('cpu', model=True, optimizer=False, grad=False)
            torch.cuda.empty_cache()
        print('VERL_LORA_EXPORT '+json.dumps(dict(rank=self.rank, stage=stage,
            source='gathered_named_parameters', tensors=len(state),
            numel=sum(v.numel() for v in state.values()),
            examples=[dict(name=n, shape=list(v.shape)) for n,v in list(state.items())[:2]])), flush=True)
        return state, config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        result = super().init_model()
        config = omega_conf_to_dataclass(self.config.actor)
        self.actor.set_loss_fn(partial(trajectory_policy_loss, config=config))
        self._export_lora('init')
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps=None, mode='naive'):
        # Base weights are preloaded by vLLM (load_format=auto). Only LoRA is
        # trainable. Gather before waking inference to reduce peak GPU usage.
        set_expandable_segments(False)  # verl's CUDA IPC allocator requirement
        try:
            state, config = self._export_lora('sync')
            self.rollout.sleep_level = 1
            if self.config.rollout.free_cache_engine:
                await self.rollout.resume(tags=['weights'])
            await self.rollout.update_weights(iter(state.items()), peft_config=config,
                base_sync_done=True, global_steps=global_steps)
            del state
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if self.config.rollout.free_cache_engine:
                await asyncio.to_thread(torch.distributed.barrier)
                await self.rollout.resume(tags=['kv_cache'])
            self.base_sync_done = True
        finally:
            set_expandable_segments(True)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        super().save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)
        # All ranks gather; only rank zero writes the loadable adapter.
        state, config = self._export_lora('checkpoint')
        if self.rank == 0:
            from peft import PeftConfig
            from safetensors.torch import save_file
            adapter = Path(local_path)/'lora_adapter'
            adapter.mkdir(parents=True, exist_ok=True)
            PeftConfig.from_peft_type(**config).save_pretrained(adapter)
            save_file(state, str(adapter/'adapter_model.safetensors'), metadata={'format':'pt'})


def pack_completions(tokenizer, prompts, outputs, prompt_length, response_length):
    """Preserve exact sampled tokens, including EOS, with separate padding masks."""
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    ids = torch.full((len(prompts), prompt_length), pad, dtype=torch.long)
    responses = torch.full((len(prompts), response_length), pad, dtype=torch.long)
    prompt_mask, response_mask = torch.zeros_like(ids), torch.zeros_like(responses)
    for i, (prompt, output) in enumerate(zip(prompts, outputs, strict=True)):
        tokens = list(output.token_ids)
        ids[i, -len(prompt):] = torch.tensor(prompt)
        prompt_mask[i, -len(prompt):] = 1
        if tokens:
            responses[i, :len(tokens)] = torch.tensor(tokens)
            response_mask[i, :len(tokens)] = 1
    attention = torch.cat([prompt_mask, response_mask], dim=1)
    return DataProto.from_dict(tensors=dict(prompts=ids, responses=responses,
        input_ids=torch.cat([ids, responses], dim=1), attention_mask=attention,
        position_ids=(attention.cumsum(-1)-1).clamp_min(0), response_mask=response_mask.float()))


class MemoryBackend:
    """Driver coordinator: all ranks train; native servers sample resident banks."""
    def __init__(self, group, config, seed):
        self.group, self.config, self.seed = group, config, seed
        self.replicas = []
        self.generation = 0
        self.weight_step = 0

    def _run(self, coroutines):
        async def gather():
            return await asyncio.gather(*coroutines)
        return asyncio.run(gather())

    def init_model(self):
        from transformers import AutoTokenizer, GenerationConfig
        from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica
        self.group.init_model()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.path)
        self.tokenizer.truncation_side = 'left'
        self.sampling = generation_settings(GenerationConfig.from_pretrained(self.config.model.path))
        model = omega_conf_to_dataclass(self.config.model)
        self.replicas = [vLLMReplica(i, self.config.rollout, model,
                                    gpus_per_node=self.group.world_size)
                         for i in range(self.group.world_size)]
        self._run([r.init_hybrid(self.group) for r in self.replicas])
        self._sleep()

    def _sleep(self):
        self._run([r.sleep() for r in self.replicas])
        print('GPU_PHASE manager_sleep', flush=True)

    def _wake(self):
        # All FSDP ranks must join this RPC. It restores weights, synchronizes
        # LoRA through verl's checkpoint engine, offloads training weights, then
        # restores the KV cache. Never run it once per turn.
        ray.get(self.group.update_weights(global_steps=self.weight_step, mode='naive'))
        print(f'GPU_PHASE manager_wake step={self.weight_step}', flush=True)

    def _generate(self, prompts, seeds, validate=False):
        encoded = [self.tokenizer(format_chat_prompt(self.tokenizer, p),
                    add_special_tokens=False, truncation=True,
                    max_length=self.config.rollout.prompt_length)['input_ids'] for p in prompts]
        jobs = []
        for i, (ids, seed) in enumerate(zip(encoded, seeds, strict=True)):
            options = dict(self.sampling, temperature=0. if validate else 1., seed=seed,
                           max_tokens=self.config.rollout.response_length,
                           stop_token_ids=[self.tokenizer.eos_token_id])
            server = self.replicas[i % len(self.replicas)].servers[0]
            jobs.append(server.generate.remote(ids, options, f'memory-{self.generation}-{i}'))
        outputs = ray.get(jobs)
        self.generation += 1
        batch = pack_completions(self.tokenizer, encoded, outputs,
                                 self.config.rollout.prompt_length, self.config.rollout.response_length)
        return batch, encoded, outputs

    def _tensor_batch(self, batch, **metadata):
        # Keep caller's padded batch intact: old/reference outputs must align
        # with the same token positions when fed into the update.
        data = batch.to_tensordict().clone()
        if 'old_log_probs' in data:
            data['memory_loss_weights'] = data['response_mask'].clone()
            data['response_mask'] = data['response_mask'].bool()
        data = left_right_2_no_padding(data)
        tu.assign_non_tensor(data, temperature=1., **metadata)
        return data

    def compute_log_prob(self, batch):
        data = self._tensor_batch(batch, calculate_entropy=True, compute_loss=False)
        output = self.group.compute_log_prob(data)
        return SimpleNamespace(batch={
            'old_log_probs': no_padding_2_padding(tu.get(output, 'log_probs'), data).float(),
            'entropys': no_padding_2_padding(tu.get(output, 'entropy'), data).float()})

    def compute_ref_log_prob(self, batch):
        data = self._tensor_batch(batch, calculate_entropy=False, compute_loss=False, no_lora_adapter=True)
        output = self.group.compute_log_prob(data)
        return SimpleNamespace(batch={'ref_log_prob':
            no_padding_2_padding(tu.get(output, 'log_probs'), data).float()})

    def set_group_size(self, total_rows):
        self.group_size = total_rows

    def update_actor(self, batch):
        data = self._tensor_batch(batch, calculate_entropy=False, compute_loss=True,
            global_batch_size=len(batch), mini_batch_size=len(batch), epochs=1,
            seed=self.seed, dataloader_kwargs={'shuffle':False})
        output = self.group.update_actor(data)
        metrics = tu.get(output, 'metrics')
        metrics = {key if key.startswith(('actor/', 'perf/')) else 'actor/'+key:value
                   for key,value in metrics.items()}
        self.weight_step += 1
        return SimpleNamespace(meta_info={'metrics':metrics})

    def save_checkpoint(self, *args):
        return self.group.save_checkpoint(*args)

    def load_checkpoint(self, *args):
        return self.group.load_checkpoint(*args)

    def sample_answers(self, rows, count, root, epoch, group, validate=False):
        requests = [(i, row, candidate) for i,row in enumerate(rows) for candidate in range(count)]
        self._wake()
        try:
            batch, prompts, outputs = self._generate([row['prompt'] for _,row,_ in requests],
                [self.seed+epoch*1000003+group*10007+i*101+c for i,_,c in requests], validate)
        finally:
            self._sleep()
        records, store = [], EvidenceStore(root)
        for (index,row,candidate), ids, output in zip(requests,prompts,outputs,strict=True):
            tokens = list(output.token_ids)
            raw = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
            prediction = extract_answer(raw)
            record = dict(dialogue_id=row['dialogue_id'], question_id=row['question_id'],
                question=row['question'], gold=row['gold'], category=row.get('category'),
                candidate=candidate, prediction=prediction, raw_answer=raw,
                em=exact_match(prediction,row['gold']), f1=f1_score(prediction,row['gold']),
                bleu1=bleu1_score(prediction,row['gold']), answer_words=len(prediction.split()),
                output_tokens=len(tokens), hit_token_limit=len(tokens)>=self.config.rollout.response_length,
                prompt_tokens=len(ids), at_prompt_limit=len(ids)>=self.config.rollout.prompt_length,
                memory_bank_ref=row.get('memory_bank_ref'), bank_evidence_root=row.get('evidence_root'),
                evidence_root=str(Path(root).resolve()),
                answer_evidence_ref=store.save(dict(prompt=row['prompt'],
                    retrieved_memories=row['retrieved_memories'], input_ids=ids,
                    completion_ids=tokens, raw_answer=raw), 'answer'))
            append_json(Path(root)/f'answers_rank_{candidate % len(self.replicas)}.jsonl',
                        dict(epoch=epoch,group=group,**record))
            if not prediction or record['hit_token_limit']:
                append_json(Path(root)/'answer_errors.jsonl', dict(epoch=epoch,group=group,
                    error='empty_prediction' if not prediction else 'token_limit',**record))
            records.append(record)
        batch.non_tensor_batch = dict(row=np.array([i for i,_,_ in requests]),
                                     candidate=np.array([c for _,_,c in requests]))
        print(f'verl-answer epoch={epoch} group={group} completed={len(records)} validate={validate}',flush=True)
        return [dict(records=records, actions=None if validate else batch)]

    def build_banks(self, timeline, count, root, epoch, group, credit_interval=0, validate=False):
        from scripts.construct_memory_bank import top_k, _embeddings
        started = perf_counter()
        _embeddings(['Memory retrieval startup'])
        store = EvidenceStore(root)
        trajectories = [dict(trajectory_id=i, bank=[], bank_points=[], invalid_turns=0,
                        repaired_turns=0, action_repaired_turns=0, partial_applied_turns=0,
                        parse_error_turns=0, applied_actions=0, rejected_actions=0,
                        turns=len(timeline['turns'])) for i in range(count)]
        if credit_interval:
            for item in trajectories:
                item['bank_points'].append(dict(after_turn=0, memory_bank_ref=store.save([])))
        batches = []
        self._wake()
        try:
            for turn in timeline['turns']:
                tick = perf_counter()
                prompts = [prompt_for_turn(turn,t['bank'],top_k) for t in trajectories]
                retrieval_seconds = perf_counter()-tick
                if prompts[0] is not None:
                    tick = perf_counter()
                    batch, inputs, outputs = self._generate(prompts,
                        [self.seed+epoch*1000003+group*10007+turn['turn_index']*101+t['trajectory_id']
                         for t in trajectories],validate)
                    elapsed = perf_counter()-tick
                    for item,prompt,ids,output in zip(trajectories,prompts,inputs,outputs,strict=True):
                        tokens = list(output.token_ids)
                        raw = self.tokenizer.decode(tokens,skip_special_tokens=True).strip()
                        generation = dict(prompt_tokens=len(ids), output_tokens=len(tokens), input_ids=ids,
                            completion_ids=tokens, actual_input_text=self.tokenizer.decode(ids,skip_special_tokens=False),
                            raw_completion=self.tokenizer.decode(tokens,skip_special_tokens=False),
                            max_new_tokens=self.config.rollout.response_length,
                            max_prompt_tokens=self.config.rollout.prompt_length,
                            at_prompt_limit=len(ids)>=self.config.rollout.prompt_length,
                            hit_token_limit=len(tokens)>=self.config.rollout.response_length,
                            batch_generate_seconds=elapsed)
                        item['bank'],action = commit_turn(timeline,turn,item['bank'],prompt,raw,generation,
                            root,item['trajectory_id'],epoch,group,item['trajectory_id']%len(self.replicas))
                        item['invalid_turns'] += int(not action['valid'])
                        item['repaired_turns'] += int(action.get('parsing',{}).get('repaired',False))
                        item['action_repaired_turns'] += int(bool(action.get('action_repairs')))
                        item['partial_applied_turns'] += int(action.get('partial_applied',False))
                        item['parse_error_turns'] += int(str(action.get('error') or '').startswith('parse:'))
                        item['applied_actions'] += action['actions']
                        item['rejected_actions'] += len(action.get('rejected_actions',[]))
                    batch.non_tensor_batch = dict(trajectory_id=np.arange(count),
                                                  turn=np.full(count,turn['turn_index']))
                    if not validate:
                        batches.append(batch)
                    append_json(Path(root)/'performance.jsonl',dict(event='manager_batch',epoch=epoch,
                        group=group,turn=turn['turn_index'],sequences=count,
                        output_tokens=sum(len(o.token_ids) for o in outputs),generate_seconds=elapsed,
                        retrieval_seconds=retrieval_seconds))
                else:
                    for item in trajectories:
                        item['bank'],_ = commit_turn(timeline,turn,item['bank'],None,'',None,root,
                            item['trajectory_id'],epoch,group,item['trajectory_id']%len(self.replicas))
                after = turn['turn_index']+1
                if credit_interval and (after%credit_interval==0 or after==len(timeline['turns'])):
                    for item in trajectories:
                        item['bank_points'].append(dict(after_turn=after,memory_bank_ref=store.save(item['bank'])))
                if after==1 or after%5==0 or after==len(timeline['turns']):
                    print(f'verl-bank epoch={epoch} group={group} turn={after}/{len(timeline["turns"])} '
                          f'trajectories={count} elapsed_s={perf_counter()-started:.1f}',flush=True)
        finally:
            self._sleep()
        return [dict(trajectories=trajectories,actions=DataProto.concat(batches) if batches else None,
                     elapsed_seconds=perf_counter()-started)]


    def build_banks_many(self, jobs, epoch, validate=True):
        """Advance independent evaluation dialogues in one batched rollout."""
        from scripts.construct_memory_bank import top_k, _embeddings
        started = perf_counter()
        _embeddings(['Memory retrieval startup'])
        states = []
        for job in jobs:
            timeline = job['timeline']
            trajectory = job['trajectory']
            states.append(dict(timeline=timeline, trajectory=trajectory,
                root=Path(job['root']), group=job['group'], next_turn=job['start_turn']))

        max_remaining = max((len(s['timeline']['turns'])-s['next_turn'] for s in states), default=0)
        woke = False
        try:
            for offset in range(max_remaining):
                active, prompts = [], []
                retrieval_started = perf_counter()
                for state in states:
                    position = state['next_turn']
                    if position >= len(state['timeline']['turns']):
                        continue
                    turn = state['timeline']['turns'][position]
                    prompt = prompt_for_turn(turn, state['trajectory']['bank'], top_k)
                    active.append((state, turn, prompt))
                    if prompt is not None:
                        prompts.append((state, turn, prompt))
                retrieval_seconds = perf_counter()-retrieval_started

                generated = {}
                if prompts:
                    if not woke:
                        self._wake()
                        woke = True
                    generate_started = perf_counter()
                    _, inputs, outputs = self._generate(
                        [prompt for _, _, prompt in prompts],
                        [self.seed+epoch*1000003+state['group']*10007+
                         turn['turn_index']*101+state['trajectory']['trajectory_id']
                         for state, turn, _ in prompts], validate)
                    generate_seconds = perf_counter()-generate_started
                    for index, (state, turn, prompt) in enumerate(prompts):
                        ids = inputs[index]
                        output = outputs[index]
                        tokens = list(output.token_ids)
                        raw = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
                        generation = dict(prompt_tokens=len(ids), output_tokens=len(tokens),
                            input_ids=ids, completion_ids=tokens,
                            actual_input_text=self.tokenizer.decode(ids, skip_special_tokens=False),
                            raw_completion=self.tokenizer.decode(tokens, skip_special_tokens=False),
                            max_new_tokens=self.config.rollout.response_length,
                            max_prompt_tokens=self.config.rollout.prompt_length,
                            at_prompt_limit=len(ids)>=self.config.rollout.prompt_length,
                            hit_token_limit=len(tokens)>=self.config.rollout.response_length,
                            batch_generate_seconds=generate_seconds)
                        generated[id(state)] = (prompt, raw, generation,
                                                index % len(self.replicas))

                for state, turn, prompt in active:
                    trajectory = state['trajectory']
                    if prompt is None:
                        trajectory['bank'], _ = commit_turn(state['timeline'], turn,
                            trajectory['bank'], None, '', None, state['root'],
                            trajectory['trajectory_id'], epoch, state['group'], 0)
                    else:
                        actual_prompt, raw, generation, replica = generated[id(state)]
                        trajectory['bank'], action = commit_turn(state['timeline'], turn,
                            trajectory['bank'], actual_prompt, raw, generation, state['root'],
                            trajectory['trajectory_id'], epoch, state['group'], replica)
                        trajectory['invalid_turns'] += int(not action['valid'])
                        trajectory['repaired_turns'] += int(action.get('parsing', {}).get('repaired', False))
                        trajectory['action_repaired_turns'] += int(bool(action.get('action_repairs')))
                        trajectory['partial_applied_turns'] += int(action.get('partial_applied', False))
                        trajectory['parse_error_turns'] += int(str(action.get('error') or '').startswith('parse:'))
                        trajectory['applied_actions'] += action['actions']
                        trajectory['rejected_actions'] += len(action.get('rejected_actions', []))
                        append_json(state['root']/'performance.jsonl', dict(
                            event='manager_batch', epoch=epoch, group=state['group'],
                            turn=turn['turn_index'], sequences=1,
                            output_tokens=generation['output_tokens'],
                            generate_seconds=generation['batch_generate_seconds'],
                            retrieval_seconds=retrieval_seconds))
                    state['next_turn'] += 1
                if offset == 0 or (offset+1) % 5 == 0 or offset+1 == max_remaining:
                    print(f'verl-test-bank batch_turn={offset+1}/{max_remaining} '
                          f'active_dialogues={len(active)} prompts={len(prompts)} '
                          f'elapsed_s={perf_counter()-started:.1f}', flush=True)
        finally:
            if woke:
                self._sleep()
        return [dict(trajectories=[s['trajectory'] for s in states], actions=None,
                     elapsed_seconds=perf_counter()-started)]


class FrozenAnswerWorker:
    """A manually GPU-bound vLLM worker that sleeps between scoring phases."""
    def __init__(self, model_path, adapter_path, memory_fraction=.80, max_model_len=6144,
                 seed=42, device_token='0'):
        # These actors intentionally reserve no Ray GPU resource; the driver
        # starts them only alongside sleeping Manager vLLM engines.
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_token)
        from transformers import AutoTokenizer, GenerationConfig
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.truncation_side = 'left'
        self.max_model_len = int(max_model_len)
        self.config = GenerationConfig.from_pretrained(model_path)
        self.memory_sampling_settings = generation_settings(self.config)
        adapter = json.loads((Path(adapter_path)/'adapter_config.json').read_text()) if adapter_path else {'r':8}
        # Supported vLLM LoRA ranks are powers of two, with minimum 8.
        rank = max(8, 2**int(np.ceil(np.log2(adapter['r']))))
        self.llm = LLM(model=model_path, dtype='bfloat16', tensor_parallel_size=1,
            gpu_memory_utilization=memory_fraction, max_model_len=max_model_len,
            max_num_batched_tokens=max_model_len, max_num_seqs=4, enable_prefix_caching=False,
            enable_chunked_prefill=False, enable_lora=True, max_lora_rank=rank,
            seed=seed, generation_config='auto', enable_sleep_mode=True)
        self.adapter = LoRARequest('frozen_answer', 1, str(Path(adapter_path).resolve())) if adapter_path else None
        self.llm.sleep(level=1)
        self.sleeping = True
        print(f'GPU_PHASE answer_sleep device={os.environ.get("CUDA_VISIBLE_DEVICES")}', flush=True)

    def wake(self):
        if self.sleeping:
            self.llm.wake_up()
            self.sleeping = False
            self._warmup_gold_path()
            print(f'GPU_PHASE answer_wake device={os.environ.get("CUDA_VISIBLE_DEVICES")}', flush=True)
        return True

    def _warmup_gold_path(self):
        """Compile the LoRA + prompt-logprob path before real SF scoring."""
        from vllm import SamplingParams
        from memfactory.chat import tokenize_chat_prompt
        from verl_memory.protocol import prepare_input
        prepared = prepare_input(self.tokenizer, tokenize_chat_prompt,
                                 'Answer with one word: warmup.', 'warmup', 128)
        self.llm.generate(
            [dict(prompt_token_ids=list(prepared.input_ids))],
            SamplingParams(temperature=0., max_tokens=1, prompt_logprobs=1),
            lora_request=self.adapter, use_tqdm=False)
        print('ANSWER_WARMUP gold_lora_prompt_tokens='+str(len(prepared.input_ids)), flush=True)

    def sleep(self):
        if not self.sleeping:
            self.llm.sleep(level=1)
            self.sleeping = True
            print(f'GPU_PHASE answer_sleep device={os.environ.get("CUDA_VISIBLE_DEVICES")}', flush=True)
        return True

    def ready(self):
        # The driver awaits this before building banks, so constructor failures
        # surface at startup rather than at the first terminal-QA request.
        return True

    def evaluate(self, requests, gold_score=False):
        from vllm import SamplingParams
        from memfactory.chat import tokenize_chat_prompt
        from verl_memory.protocol import prepare_input
        started = perf_counter()
        prepared = []
        if gold_score:
            for r in requests:
                p = prepare_input(self.tokenizer, tokenize_chat_prompt, r['prompt'], r['gold'], r['max_prompt_tokens'])
                prepared.append(p)
            print('ANSWER_GOLD_INPUTS '+json.dumps([
                dict(request_index=r.get('_score_index'), input_tokens=len(p.input_ids),
                     gold_tokens=p.gold_tokens, crossing_boundary_tokens=p.crossing_boundary_tokens,
                     max_model_len=self.max_model_len,
                     over_limit=len(p.input_ids) > self.max_model_len)
                for r,p in zip(requests,prepared)], ensure_ascii=False), flush=True)
            inputs = [dict(prompt_token_ids=list(p.input_ids)) for p in prepared]
            # Teacher forcing: one prefill scores all gold tokens. No free-form answer.
            sampling = SamplingParams(temperature=0., max_tokens=1, prompt_logprobs=1)
        else:
            inputs = [dict(prompt_token_ids=tokenize_chat_prompt(self.tokenizer, r['prompt'],
                r['max_prompt_tokens'])['input_ids'][0].tolist()) for r in requests]
            sampling = SamplingParams(temperature=0., max_tokens=requests[0]['max_new_tokens'],
                                       repetition_penalty=self.memory_sampling_settings['repetition_penalty'])
        try:
            outputs = self.llm.generate(inputs, sampling, lora_request=self.adapter, use_tqdm=False)
        except BaseException as exc:
            meta = [dict(request_index=r.get('_score_index'), input_tokens=len(p.input_ids),
                         gold_tokens=p.gold_tokens, max_model_len=self.max_model_len,
                         over_limit=len(p.input_ids) > self.max_model_len)
                    for r,p in zip(requests, prepared)] if gold_score else [
                        dict(request_index=r.get('_score_index'),
                             input_tokens=len(item['prompt_token_ids']),
                             max_model_len=self.max_model_len,
                             over_limit=len(item['prompt_token_ids']) > self.max_model_len)
                        for r,item in zip(requests, inputs)]
            try:
                memory = dict(allocated=int(torch.cuda.memory_allocated()),
                              reserved=int(torch.cuda.memory_reserved()))
            except Exception:
                memory = {}
            print('ANSWER_ENGINE_FAILURE '+json.dumps(
                dict(gold_score=gold_score, request_meta=meta, memory=memory,
                     error=repr(exc)), ensure_ascii=False), flush=True)
            raise
        result = []
        for i, output in enumerate(outputs):
            if gold_score:
                p = prepared[i]
                values = [output.prompt_logprobs[j][p.input_ids[j]].logprob
                          for j in range(p.gold_start, len(p.input_ids))]
                result.append(dict(ell=float(np.clip(sum(values)/len(values), -10., 0.)), gold_tokens=len(values),
                    input_ids=list(p.input_ids), gold_start=p.gold_start, gold_token_log_probs=values))
            else:
                completion = output.outputs[0]
                result.append(dict(prediction=extract_answer(completion.text), raw_answer=completion.text,
                    input_ids=inputs[i]['prompt_token_ids'], completion_ids=list(completion.token_ids),
                    prompt_tokens=len(inputs[i]['prompt_token_ids']),
                    output_tokens=len(completion.token_ids), hit_token_limit=completion.finish_reason == 'length'))
        return dict(rows=result, elapsed_seconds=perf_counter()-started,
                    requests=len(requests), gold_score=gold_score)
