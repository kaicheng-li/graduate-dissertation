"""CPU boundary tests; Ray/CUDA transport is deliberately replaced with doubles."""
import ast
import asyncio
import importlib.util
import json
import sys
from functools import partial
from pathlib import Path
import tempfile
from time import perf_counter
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import torch

from memfactory.diagnostics import EvidenceStore
from verl_memory.protocol import append_json, commit_turn, prompt_for_turn


class Batch:
    def __init__(self, tensors):
        self.batch, self.non_tensor_batch = tensors, {}

    @classmethod
    def from_dict(cls, tensors):
        return cls(tensors)

    @classmethod
    def concat(cls, batches):
        result = cls({k:torch.cat([b.batch[k] for b in batches]) for k in batches[0].batch})
        result.non_tensor_batch = {k:np.concatenate([b.non_tensor_batch[k] for b in batches])
                                  for k in batches[0].non_tensor_batch}
        return result


def cpu_definitions():
    # Load the actual definitions without importing GPU-only engine modules.
    # This does not assert that CUDA initialization or Ray transport works.
    path = Path(__file__).resolve().parents[1]/'verl_memory/workers.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = {'pack_completions', 'MemoryBackend', 'MemoryActorWorker',
             'trajectory_policy_loss', 'generation_settings', 'lora_from_named_parameters'}
    names.add('MemoryBackend')
    tree.body = [node for node in tree.body if getattr(node, 'name', None) in names]
    scope = dict(torch=torch, np=np, asyncio=asyncio, json=json, Path=Path, SimpleNamespace=SimpleNamespace,
                 DataProto=Batch, ray=SimpleNamespace(get=lambda x:x), perf_counter=perf_counter,
                 EvidenceStore=EvidenceStore, append_json=append_json,
                 commit_turn=commit_turn, prompt_for_turn=prompt_for_turn,
                 format_chat_prompt=lambda tokenizer, prompt:prompt)
    class EngineWorker:
        def init_model(self):
            return 'initialized'

        def save_checkpoint(self, *args):
            self.saved_shards = args
    scope.update(ActorRolloutRefWorker=EngineWorker,
                 partial=partial, omega_conf_to_dataclass=lambda config:config,
                 set_expandable_segments=Mock(),
                 Dispatch=SimpleNamespace(ONE_TO_ALL='all'),
                 register=lambda **kw:lambda fn:fn)
    exec(compile(tree,str(path),'exec'),scope)
    return scope


class BackendBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.scope = cpu_definitions()
        self.tokenizer = SimpleNamespace(pad_token_id=2, eos_token_id=2)

    def pack(self, prompts, outputs):
        return self.scope['pack_completions'](self.tokenizer, prompts,
            [SimpleNamespace(token_ids=x, stop_reason='completed') for x in outputs],4,3)

    def backend(self):
        config = SimpleNamespace(rollout=SimpleNamespace(prompt_length=4,response_length=3))
        return self.scope['MemoryBackend'](Mock(),config,42)

    def test_answer_workers_borrow_visible_devices_without_ray_gpu_reservations(self):
        from verl_memory.train import create_answer_workers, visible_gpu_tokens

        self.assertEqual(visible_gpu_tokens('2,0,3,1', 4), ['2', '0', '3', '1'])
        self.assertEqual(visible_gpu_tokens('', 4), ['0', '1', '2', '3'])
        options = {}
        class RemoteClass:
            def remote(self, *args):
                return SimpleNamespace(args=args)
        class FakeRay:
            def remote(self, **kwargs):
                options.update(kwargs)
                return lambda worker_class: RemoteClass()

        args = SimpleNamespace(answer_model='base', answer_adapter='adapter',
            answer_memory_fraction=.8, max_prompt_tokens=100, max_new_tokens=20, seed=7)
        workers = create_answer_workers(FakeRay(), object, args, ['0', '1', '2', '3'])
        self.assertEqual(options, {'num_gpus': 0, 'num_cpus': 2})
        self.assertEqual(len(workers), 4)
        self.assertEqual([worker.args[-1] for worker in workers], ['0', '1', '2', '3'])
        self.assertEqual([worker.args[-2] for worker in workers], [7, 8, 9, 10])

    def test_answer_scoring_wakes_once_and_sleeps_after_the_whole_phase(self):
        from verl_memory.train import score_requests

        events = []
        class FakeRay:
            @staticmethod
            def get(refs):
                values = []
                for ref in refs:
                    if ref[0] == 'evaluate':
                        values.append(dict(rows=[dict(id=row['id']) for row in ref[2]],
                                           elapsed_seconds=0., requests=len(ref[2]), gold_score=False))
                    else:
                        values.append(True)
                return values

        workers = []
        for index in range(2):
            workers.append(SimpleNamespace(
                wake=SimpleNamespace(remote=lambda i=index: (events.append(('wake', i)) or ('wake', i))),
                evaluate=SimpleNamespace(remote=lambda rows, gold, i=index:
                    (events.append(('evaluate', i)) or ('evaluate', i, rows))),
                sleep=SimpleNamespace(remote=lambda i=index: (events.append(('sleep', i)) or ('sleep', i)))))
        args = SimpleNamespace(answer_batch_size=1, gold_batch_size=1)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'ray': FakeRay}):
            rows = score_requests(workers, [dict(id='a'), dict(id='b'), dict(id='c')],
                                  args, tmp, 'unit_test')
        self.assertEqual(rows, [dict(id='a'), dict(id='b'), dict(id='c')])
        self.assertEqual(events[:2], [('wake', 0), ('wake', 1)])
        self.assertEqual(events[-2:], [('sleep', 0), ('sleep', 1)])

    def test_eos_equal_to_pad_remains_a_training_token(self):
        data = self.pack([[5,6],[7]], [[9,2],[8,9,2]]).batch
        self.assertEqual(data['response_mask'].tolist(),[[1,1,0],[1,1,1]])
        self.assertEqual(data['attention_mask'].tolist(),[[0,0,1,1,1,1,0],[0,0,0,1,1,1,1]])
        self.assertEqual(data['position_ids'][0].tolist(),[0,0,0,1,2,3,3])

    def test_empty_completion_is_not_fabricated(self):
        data = self.pack([[5]], [[]]).batch
        self.assertEqual(data['response_mask'].sum().item(),0)

    def test_native_tensor_and_zip_errors_propagate(self):
        with self.assertRaises(ValueError):
            self.pack([[5],[6]],[[7]])
        with self.assertRaises(RuntimeError):
            self.pack([[5]],[[6,7,8,9]])

    def test_native_requests_keep_order_seeds_and_temperature(self):
        actor=self.backend()
        actor.tokenizer=Mock(side_effect=lambda *a,**k:{'input_ids':[5,6]})
        actor.tokenizer.pad_token_id=actor.tokenizer.eos_token_id=2
        actor.sampling=dict(top_k=20,top_p=.8,repetition_penalty=1.05)
        servers=[SimpleNamespace(generate=Mock()),SimpleNamespace(generate=Mock())]
        for i,server in enumerate(servers):
            server.generate.remote.return_value=SimpleNamespace(token_ids=[10+i,2],stop_reason='completed')
        actor.replicas=[SimpleNamespace(servers=[s]) for s in servers]
        batch,_,_=actor._generate(['a','b','c'],[100,101,102])
        self.assertEqual(batch.batch['responses'][:,0].tolist(),[10,11,10])
        self.assertEqual(servers[0].generate.remote.call_args_list[1].args[1]['seed'],102)
        self.assertEqual(servers[1].generate.remote.call_args.args[1]['temperature'],1.)
        actor._generate(['v'],[7],True)
        self.assertEqual(servers[0].generate.remote.call_args.args[1]['temperature'],0.)
        self.assertEqual(servers[0].generate.remote.call_args.args[2],'memory-1-0')
        self.assertEqual(servers[0].generate.remote.call_args.args[1]['top_p'],.8)

    def test_bank_persists_independently_and_aux_points_include_final(self):
        actor=self.backend()
        actor.replicas=[None,None]
        actor._wake=Mock()
        actor._sleep=Mock()
        actor.tokenizer=SimpleNamespace(decode=lambda ids,**kw:str(ids))
        actor._generate=Mock(return_value=(self.pack([[5]]*4,[[6,2]]*4),
            [[5]]*4,[SimpleNamespace(token_ids=[6,2]) for _ in range(4)]))
        seen=[]
        def prompt(turn,bank,retriever):
            seen.append((turn['turn_index'],list(bank)))
            return 'prompt'
        def commit(timeline,turn,bank,prompt,raw,generation,root,tid,*args):
            return bank+[(tid,turn['turn_index'])],{'valid':True,'actions':1}
        self.scope['prompt_for_turn']=prompt
        self.scope['commit_turn']=commit
        with tempfile.TemporaryDirectory() as tmp, patch('scripts.construct_memory_bank._embeddings'):
            result=actor.build_banks({'turns':[{'turn_index':i} for i in range(5)]},4,tmp,1,0,4)[0]
        for t in result['trajectories']:
            self.assertEqual(t['bank'],[(t['trajectory_id'],i) for i in range(5)])
            self.assertEqual([p['after_turn'] for p in t['bank_points']],[0,4,5])
            self.assertEqual(t['applied_actions'],5)
            self.assertEqual(t['parse_error_turns'],0)
        self.assertEqual(seen[4],(1,[(0,0)]))
        self.assertEqual(result['actions'].non_tensor_batch['trajectory_id'].tolist(),[0,1,2,3]*5)
        actor._wake.assert_called_once()
        actor._sleep.assert_called_once()

    def test_test_dialogues_advance_in_parallel_and_resume_from_saved_turn(self):
        actor=self.backend()
        actor.replicas=[None,None]
        actor._wake=Mock()
        actor._sleep=Mock()
        actor.seed=42
        actor.generation=0
        actor.config=SimpleNamespace(rollout=SimpleNamespace(prompt_length=32,response_length=8))
        actor.tokenizer=SimpleNamespace(decode=lambda ids,**kw: str(ids))
        batches=[]
        def generate(prompts,seeds,validate=False):
            batches.append(list(prompts))
            return None, [[i+1] for i,_ in enumerate(prompts)], [
                SimpleNamespace(token_ids=[i+1]) for i,_ in enumerate(prompts)]
        actor._generate=generate
        seen=[]
        def prompt(turn,bank,retriever):
            seen.append((turn['dialogue_id'],turn['turn_index'],list(bank)))
            return f"{turn['dialogue_id']}:{turn['turn_index']}"
        def commit(timeline,turn,bank,prompt,raw,generation,root,tid,epoch,group,rank):
            return bank+[f"{timeline['dialogue_id']}:{turn['turn_index']}"],dict(
                valid=True,actions=1,action_repairs=[],rejected_actions=[])
        self.scope['prompt_for_turn']=prompt
        self.scope['commit_turn']=commit
        def state(dialogue_id,start,bank):
            turns=[dict(dialogue_id=dialogue_id,turn_index=i,facts=[dict(text='fact')])
                   for i in range(3)]
            timeline=dict(dialogue_id=dialogue_id,turns=turns)
            trajectory=dict(trajectory_id=0,bank=bank,turns=3,invalid_turns=0,
                repaired_turns=0,action_repaired_turns=0,partial_applied_turns=0,
                parse_error_turns=0,applied_actions=0,rejected_actions=0)
            return dict(timeline=timeline,trajectory=trajectory,root=dialogue_id,
                        group=start,start_turn=start)
        jobs=[state('already-started',1,['already-started:0']),state('fresh',0,[])]
        with patch('scripts.construct_memory_bank._embeddings'):
            result=actor.build_banks_many(jobs,epoch=0,validate=True)
        self.assertEqual([len(batch) for batch in batches],[2,2,1])
        self.assertEqual(result[0]['trajectories'][0]['bank'],
                         ['already-started:0','already-started:1','already-started:2'])
        self.assertEqual(result[0]['trajectories'][1]['bank'],
                         ['fresh:0','fresh:1','fresh:2'])
        self.assertEqual(actor._wake.call_count,1)
        self.assertEqual(actor._sleep.call_count,1)
        self.assertNotIn(('already-started',0,[]),seen)

    def test_resume_scans_saved_trajectories_and_does_not_assume_id_zero(self):
        from verl_memory.train import _test_progress

        timeline = dict(dialogue_id='resume-dialogue',
                        turns=[dict(turn_index=i) for i in range(3)],
                        questions=[dict(question_id=0, question='q', answer='a')])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def write_attempt(number, trajectories):
                trace = root/'evaluation'/'epoch_0'/'0'/f'attempt_{number}'
                store = EvidenceStore(trace)
                path = trace/'trajectories_rank_0.jsonl'
                for tid, turns in trajectories.items():
                    previous = []
                    for turn in turns:
                        before = store.save(previous)
                        previous = [dict(id=str(tid), turn=turn)]
                        append_json(path, dict(event='trajectory_turn', trajectory_id=tid,
                            turn=turn, valid=True, actions=1,
                            bank_before_ref=before, memory_bank_ref=store.save(previous)))
                return trace

            write_attempt(0, {'9': [0, 1, 2], '4': [0, 1]})
            latest = write_attempt(1, {'8': [0, 1, 2], '11': [0, 1, 2]})
            append_json(latest/'answers.jsonl', dict(dialogue_id='resume-dialogue',
                trajectory_id='8', question_id=0, prediction='cached'))
            append_json(latest/'answers.jsonl', dict(dialogue_id='resume-dialogue',
                trajectory_id='11', question_id=0, prediction='other'))

            state = _test_progress(root, 0, timeline, 0)
            self.assertEqual(state['trajectory']['trajectory_id'], 8)
            self.assertEqual(state['start_turn'], 3)
            self.assertEqual(state['trajectory']['bank'], [dict(id='8', turn=2)])
            self.assertEqual([row['prediction'] for row in state['existing_answers']], ['cached'])

    def test_resume_does_not_join_divergent_rank_histories(self):
        from verl_memory.train import _test_progress

        timeline = dict(dialogue_id='branching-dialogue',
                        turns=[dict(turn_index=i) for i in range(3)], questions=[])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root/'evaluation'/'epoch_0'/'5'/'attempt_0'
            store = EvidenceStore(trace)
            empty = []
            first = [dict(id='6', turn=0)]
            correct_second = [dict(id='6', turn=1)]
            wrong_second = [dict(id='6', turn='wrong-branch')]
            third = [dict(id='6', turn=2)]
            rows = {
                'trajectories_rank_0.jsonl': [
                    dict(event='trajectory_turn', trajectory_id=6, turn=0,
                         bank_before_ref=store.save(empty), memory_bank_ref=store.save(first)),
                    # This duplicate row has a turn number gap in its bank chain.
                    dict(event='trajectory_turn', trajectory_id=6, turn=1,
                         bank_before_ref=store.save(empty), memory_bank_ref=store.save(wrong_second)),
                ],
                'trajectories_rank_1.jsonl': [
                    dict(event='trajectory_turn', trajectory_id=6, turn=1,
                         bank_before_ref=store.save(first), memory_bank_ref=store.save(correct_second)),
                    dict(event='trajectory_turn', trajectory_id=6, turn=2,
                         bank_before_ref=store.save(correct_second), memory_bank_ref=store.save(third)),
                ],
            }
            for filename, records in rows.items():
                for record in records:
                    append_json(trace/filename, record)

            state = _test_progress(root, 5, timeline, 0)
            self.assertEqual(state['trajectory']['trajectory_id'], 6)
            self.assertEqual(state['start_turn'], 3)
            self.assertEqual(state['trajectory']['bank'], third)

    def test_failed_generation_releases_rollout_memory(self):
        actor=self.backend()
        actor._wake=Mock()
        actor._sleep=Mock()
        actor._generate=Mock(side_effect=RuntimeError('server failed'))
        with self.assertRaisesRegex(RuntimeError,'server failed'):
            actor.sample_answers([{'prompt':'p'}],4,'unused',1,0)
        actor._sleep.assert_called_once()

    def test_epoch_export_gathers_on_all_ranks_but_only_rank_zero_writes(self):
        from safetensors.torch import load_file
        worker = self.scope['MemoryActorWorker']()
        params = {'base_model.model.proj.lora_A.weight':torch.arange(6.).reshape(2,3)}
        worker._export_lora = Mock(return_value=(params,{'r':2}))
        config = Mock()
        peft = SimpleNamespace(PeftConfig=SimpleNamespace(from_peft_type=Mock(return_value=config)))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules,{'peft':peft}):
            worker.rank = 1
            worker.save_checkpoint(tmp,None,3,None)
            self.assertFalse((Path(tmp)/'lora_adapter').exists())
            worker.rank = 0
            worker.save_checkpoint(tmp,None,3,None)
            loaded = load_file(str(Path(tmp)/'lora_adapter/adapter_model.safetensors'))
            torch.testing.assert_close(loaded[next(iter(params))],params[next(iter(params))])
            self.assertEqual(worker.saved_shards,(tmp,None,3,None))
            config.save_pretrained.assert_called_once_with(Path(tmp)/'lora_adapter')
            self.assertEqual(worker._export_lora.call_count,2)
            worker._export_lora.assert_called_with('checkpoint')

    def test_initialization_uses_the_same_export_as_sync(self):
        worker=self.scope['MemoryActorWorker']()
        worker.rank=1
        worker.config=SimpleNamespace(actor=SimpleNamespace())
        worker.actor=Mock()
        worker._export_lora=Mock()
        self.assertEqual(worker.init_model(),'initialized')
        worker._export_lora.assert_called_once_with('init')

    def test_explicit_parameter_export_bypasses_empty_state_dict_and_owns_storage(self):
        a=torch.nn.Parameter(torch.ones(2,3))
        b=torch.nn.Parameter(torch.zeros(3,2))
        model=Mock()
        model.named_parameters.return_value=iter([
            ('base_model.model._fsdp_wrapped_module.proj.lora_A.default.weight',a),
            ('base_model.model._fsdp_wrapped_module.proj.lora_B.default._fsdp_wrapped_module.weight',b),
            ('base_model.model.proj.base_layer.weight',torch.nn.Parameter(torch.ones(3,3),requires_grad=False))])
        model.peft_config={'default':SimpleNamespace(to_dict=lambda:{'r':2})}
        model.state_dict.return_value={}
        state,config=self.scope['lora_from_named_parameters'](SimpleNamespace(_fsdp_wrapped_module=model))
        self.assertEqual(len(state),2)
        self.assertEqual(config,{'r':2})
        model.state_dict.assert_not_called()
        with torch.no_grad():
            a.zero_()
        torch.testing.assert_close(state['base_model.model.proj.lora_A.weight'],torch.ones(2,3))
        self.assertTrue(all('_fsdp_wrapped_module' not in k and '.default.' not in k for k in state))

    @unittest.skipUnless(importlib.util.find_spec('peft'), 'optional real PEFT dependency')
    def test_real_peft_nested_wrappers_export_and_reload_identical_outputs(self):
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.block=torch.nn.Sequential(torch.nn.Linear(3,4,bias=False))
            def forward(self,x): return self.block(x)

        class Wrapped(torch.nn.Module):
            # Reproduce nested FSDP module paths on CPU, not CUDA collectives.
            def __init__(self,module):
                super().__init__()
                self._fsdp_wrapped_module=module
            def forward(self,*args,**kwargs): return self._fsdp_wrapped_module(*args,**kwargs)
            def __getattr__(self,name):
                try: return super().__getattr__(name)
                except AttributeError: return getattr(self._fsdp_wrapped_module,name)

        torch.manual_seed(42)
        base=Tiny()
        base_state={k:v.detach().clone() for k,v in base.state_dict().items()}
        config=LoraConfig(r=2,lora_alpha=4,target_modules=['0'])
        model=get_peft_model(base,config)
        layer=model.base_model.model.block[0]
        with torch.no_grad(): layer.lora_B.default.weight.fill_(.3)
        expected=get_peft_model_state_dict(model)
        x=torch.randn(5,3)
        expected_output=model(x).detach()
        layer.lora_A.default=Wrapped(layer.lora_A.default)
        layer.lora_B.default=Wrapped(layer.lora_B.default)
        model.base_model.model.block=Wrapped(model.base_model.model.block)
        with patch.object(model,'state_dict',side_effect=AssertionError('must use gathered parameters')):
            state,_=self.scope['lora_from_named_parameters'](Wrapped(model))
        self.assertEqual(state.keys(),expected.keys())
        for name in state:
            torch.testing.assert_close(state[name],expected[name])
            self.assertNotEqual(state[name].data_ptr(),expected[name].data_ptr())
        restored_base=Tiny()
        restored_base.load_state_dict(base_state)
        restored=get_peft_model(restored_base,config)
        set_peft_model_state_dict(restored,state)
        torch.testing.assert_close(restored(x),expected_output)

    def test_sync_sends_exported_weights_before_waking_kv_cache(self):
        worker=self.scope['MemoryActorWorker']()
        worker.config=SimpleNamespace(rollout=SimpleNamespace(free_cache_engine=True))
        events=[]
        state={'proj.lora_A.weight':torch.ones(2,3)}
        def export(stage):
            events.append(stage)
            return state,{'r':2}
        async def resume(tags): events.extend(tags)
        async def update(weights,**kwargs):
            events.append('transfer')
            self.assertEqual(dict(weights),state)
            self.assertEqual(kwargs,dict(peft_config={'r':2},base_sync_done=True,global_steps=5))
        worker._export_lora=export
        worker.rollout=SimpleNamespace(resume=resume,update_weights=update)
        with patch('torch.cuda.synchronize'),patch('torch.cuda.empty_cache'),patch('torch.distributed.barrier'):
            asyncio.run(worker.update_weights(global_steps=5))
        self.assertEqual(events,['sync','weights','transfer','kv_cache'])
        self.assertTrue(worker.base_sync_done)

    def test_empty_export_cannot_replace_the_current_adapter(self):
        worker=self.scope['MemoryActorWorker']()
        worker._export_lora=Mock(side_effect=RuntimeError('Empty LoRA export'))
        worker.rollout=SimpleNamespace(resume=AsyncMock(),update_weights=AsyncMock())
        with self.assertRaisesRegex(RuntimeError,'Empty LoRA export'):
            asyncio.run(worker.update_weights())
        worker.rollout.update_weights.assert_not_called()


if __name__ == '__main__':
    unittest.main()
