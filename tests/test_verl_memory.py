"""CPU verification of framework boundaries, bank parity and loss weighting."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from memfactory.manager_env import validate_and_apply
from memfactory.modules.memory_updater import build_manager_input
from scripts.construct_memory_bank import parse_manager_output
from verl_memory.protocol import (block_advantages, commit_turn, per_qa_advantages,
                                  prompt_for_turn, validate_timelines)
from verl_memory.train import (apply_update, answer_loss_mask, finish_epoch, plot_run,
                               resume_state, weight_loss_mask, write_json,
                               attach_linked_question_turns)


class VerlMemoryTests(unittest.TestCase):
    def test_raw_nulls_and_nonobjects_are_not_silently_coerced_or_dropped(self):
        from copy import deepcopy
        from scripts.build_manager_training_data import normalize_locomo_sample
        from scripts.rebuild_manager_states import flatten_dialogue
        sample=dict(sample_id='conv-26',conversation={'speaker_a':'A','session_1':[
            dict(speaker='A',text='hello')]},qa=[dict(question='q',answer='a',category=1)])
        for field in ('question','answer'):
            bad=deepcopy(sample); bad['qa'][0][field]=None
            with self.subTest(field=field):
                self.assertEqual(normalize_locomo_sample(bad)['questions'], [])
        for bad_turn in (None,'not an object',dict(speaker='A',text=None)):
            bad=deepcopy(sample); bad['conversation']['session_1'][0]=bad_turn
            for fn in (normalize_locomo_sample,flatten_dialogue):
                with self.subTest(turn=bad_turn,fn=fn),self.assertRaisesRegex(ValueError,'session_1'):
                    fn(bad)
        bad=deepcopy(sample); bad['qa'].append(None)
        self.assertEqual(len(normalize_locomo_sample(bad)['questions']), 1)

    def test_numeric_and_list_gold_and_optional_null_metadata_are_preserved(self):
        from scripts.build_manager_training_data import normalize_locomo_sample
        sample=dict(sample_id='conv-26',conversation={'speaker_a':'A','session_1_date_time':'yesterday',
            'session_1':[dict(speaker='A',text='',timestamp=None,dia_id=None)]},
            qa=[dict(question='q',answer=[2023,'Paris'],category=1,evidence=None,question_id=None)])
        result=normalize_locomo_sample(sample)
        self.assertEqual(result['questions'][0]['answer'],'2023, Paris')
        self.assertEqual(result['questions'][0]['evidence'],[])
        self.assertEqual(result['questions'][0]['question_id'],'0')
        self.assertEqual(result['turns'][0]['text'],'')
        self.assertEqual(result['turns'][0]['timestamp'],'yesterday')
        self.assertEqual(result['turns'][0]['dia_id'],'D1:0')
        for answer in ([None],{'answer':'a'},True,[],[[]]):
            sample['qa'][0]['answer']=answer
            with self.subTest(answer=answer):
                self.assertEqual(normalize_locomo_sample(sample)['questions'], [])

    def test_bad_facts_are_logged_and_isolated_in_cache_and_rollout(self):
        from scripts.build_manager_training_data import FactExtractor
        from scripts.rebuild_manager_states import load_cached_facts
        turn=dict(speaker='A',text='hello',timestamp='today',turn_index=0)
        for facts in ([None],[dict(text=None)],[dict(text={'value':'a'})]):
            with self.subTest(facts=facts),tempfile.TemporaryDirectory() as tmp:
                cache=Path(tmp)/'cache.json'
                cache.write_text(json.dumps({'facts:d:0':facts}))
                extractor=FactExtractor(str(cache),'teacher','unused','unused',100,0)
                self.assertEqual(extractor.extract_facts('d',0,turn), [])
                rows=Path(tmp)/'facts.jsonl'
                rows.write_text(json.dumps(dict(dialogue_id='d',dialogue_turns=[dict(turn,facts=facts)])))
                self.assertEqual(load_cached_facts(rows,'d'), {0: []})
                self.assertIsNone(prompt_for_turn(dict(turn,facts=facts),[],lambda q,b,k:b))
        with tempfile.TemporaryDirectory() as tmp:
            cache=Path(tmp)/'new.json'
            extractor=FactExtractor(str(cache),'teacher','unused','unused',100,0)
            extractor._json_completion=Mock(return_value={'memory_list':[{'value':None}]})
            self.assertEqual(extractor.extract_facts('d',0,turn), [])
            self.assertTrue(cache.exists())

    def test_loaded_answer_rows_reject_null_gold_and_nonobject_records(self):
        from verl_memory.train import validate_answer_rows
        for row in (None,dict(dialogue_id='d',question_id='0',prompt='p',gold=None,retrieved_memories=[])):
            with self.subTest(row=row):
                self.assertEqual(validate_answer_rows([row]), [])

    def test_recoverable_data_is_logged_without_changing_valid_qa_ids(self):
        import os
        from memfactory.modules.memory_extractor import validate_facts
        from scripts.build_manager_training_data import normalize_locomo_sample
        from verl_memory.train import validate_answer_rows, read_jsonl
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'MEMORY_DATA_ISSUES_DIR':tmp}):
            self.assertEqual(validate_facts([dict(text='good'), None, dict(text=None)], 'facts'), [dict(text='good')])
            sample = dict(sample_id='d', conversation={'speaker_a':'A','session_1':[dict(speaker='A',text='hi')]},
                qa=[dict(question='q',answer=None),dict(question='q',answer=0),
                    dict(question='q'),dict(question='q',answer=['Paris',2023])])
            questions = normalize_locomo_sample(sample)['questions']
            self.assertEqual([q['question_id'] for q in questions], ['1','3'])
            self.assertEqual([q['answer'] for q in questions], ['0','Paris, 2023'])
            timeline = dict(dialogue_id='d',participants=['A'],questions=questions,
                turns=[dict(turn_index=0,speaker='A',text=None,facts=[None,dict(text='good')]),
                       None,dict(turn_index=2,speaker='A',text='next',facts=None)])
            cleaned = validate_timelines([timeline])[0]
            self.assertEqual([t['turn_index'] for t in cleaned['turns']], [0,1,2])
            self.assertEqual([t['facts'] for t in cleaned['turns']], [[dict(text='good')],[],[]])
            rows = validate_answer_rows([dict(dialogue_id='d',question_id='7',prompt='p',gold=None),
                dict(dialogue_id='d',question_id='8',prompt='p',gold=0,retrieved_memories=None)])
            self.assertEqual([r['question_id'] for r in rows], ['8'])
            self.assertEqual(rows[0]['gold'], '0')
            path=Path(tmp)/'input.jsonl'
            path.write_text('{bad json}\n{"value":1}\n')
            self.assertEqual(read_jsonl(path), [dict(value=1)])
            records=[json.loads(line) for file in Path(tmp).glob('data_issues_pid_*.jsonl')
                     for line in file.read_text(encoding='utf-8').splitlines()]
            self.assertTrue(any(r['location']=='facts[1]' and r['raw'] is None for r in records))
            self.assertTrue(any(r['action']=='skip_qa' and r['raw'].get('question_id')=='0' for r in records
                                if isinstance(r['raw'],dict)))
            self.assertTrue(any(r['action']=='skip_jsonl_record' for r in records))

    def test_json_comments_and_ellipsis_keep_explicit_actions_and_literal_text(self):
        raw = '''{"memory":[...,
          {"id":"a","event":"UPDATE","old_memory":"old","text":"https://x /*literal*/ ..."},
          // omitted unchanged entries
          …,
          {"event":"ADD", /* explanation */ "text":"new"}, ...]}'''
        diagnostics = {}
        decisions = parse_manager_output(raw, diagnostics)
        self.assertEqual([d['event'] for d in decisions], ['UPDATE', 'ADD'])
        self.assertEqual(decisions[0]['text'], 'https://x /*literal*/ ...')
        self.assertEqual({r['kind'] for r in diagnostics['repairs']},
                         {'remove_comment', 'skip_ellipsis_entry', 'remove_trailing_comma'})

    def test_unknown_update_does_not_cancel_independent_adds(self):
        result = validate_and_apply([], [], [dict(event='ADD', text='one'),
            dict(id='new_id', event='UPDATE', text='unresolved'), dict(event='ADD', text='two')],
            dialogue_id='d', source_turn=1, speaker='A', timestamp='today')
        self.assertFalse(result.valid)
        self.assertEqual([m['text'] for m in result.memory_bank], ['one', 'two'])
        self.assertEqual(result.applied_actions, 2)
        self.assertEqual(result.rejected_actions[0]['index'], 1)

    def test_matching_text_never_retargets_unknown_ids(self):
        bank = [dict(id='a', text='old', speaker='A')]
        for event in ('UPDATE', 'DELETE', 'NONE'):
            for extra in ({}, dict(old_memory='old')):
                with self.subTest(event=event, extra=extra):
                    result = validate_and_apply(bank, bank,
                        [dict(id='new_id', event=event, text='old', **extra)],
                        dialogue_id='d', source_turn=1, speaker='A', timestamp='today')
                    self.assertEqual(result.valid, event == 'NONE')
                    self.assertEqual(result.memory_bank, bank)
                    self.assertEqual(result.applied_actions, 0)
                    self.assertEqual(result.id_mappings, ())
                    if event != 'NONE':
                        self.assertEqual(result.rejected_ids, ('new_id',))

    def test_unknown_update_cannot_override_valid_delete(self):
        bank = [dict(id='a', text='old')]
        result = validate_and_apply(bank, bank,
            [dict(id='a', event='DELETE'),
             dict(id='new_id', event='UPDATE', text='new', old_memory='old'),
             dict(event='ADD', text='independent')],
            dialogue_id='d', source_turn=1, speaker='A', timestamp='today')
        self.assertFalse(result.valid)
        self.assertEqual([m['text'] for m in result.memory_bank], ['independent'])
        self.assertEqual(result.applied_actions, 2)
        self.assertEqual(result.rejected_ids, ('new_id',))
        self.assertEqual(result.repairs, ())

    def test_unknown_update_does_not_create_duplicate_target_zero(self):
        bank = [dict(id='0', text='Caroline greeted Mel', speaker='Caroline')]
        actions = [dict(id='0', event='UPDATE', text=bank[0]['text'], old_memory=bank[0]['text']),
                   dict(id='4', event='UPDATE', text='Melanie greets Caroline',
                        old_memory=bank[0]['text'])]
        for decisions in (actions, list(reversed(actions))):
            result = validate_and_apply(bank, bank, decisions,
                dialogue_id='conv-26', source_turn=1, speaker='Melanie', timestamp='today')
            self.assertFalse(result.valid)
            self.assertEqual(result.applied_actions, 1)
            self.assertEqual(result.rejected_ids, ('4',))
            self.assertEqual(result.error, "UPDATE id '4' was not visible in the prompt")
            self.assertEqual(result.memory_bank[0]['text'], bank[0]['text'])
            self.assertEqual([(m['model_id'], m['bank_id']) for m in result.id_mappings], [('0', '0')])

    def test_unseen_none_is_logged_without_creating_memory(self):
        timeline = dict(dialogue_id='test', participants=['A'])
        turn = dict(turn_index=0, speaker='A', timestamp='today', facts=[dict(text='fact')])
        with tempfile.TemporaryDirectory() as tmp:
            prompt = prompt_for_turn(turn, [], lambda q,b,k:b)
            raw = '{"memory":[{"id":"new-1","event":"NONE","text":"not an ADD"}]}'
            bank, action = commit_turn(timeline, turn, [], prompt, raw, None, tmp, 0, 1, 0, 0)
            self.assertEqual(bank, [])
            self.assertTrue(action['valid'])
            row = json.loads((Path(tmp)/'manager_repairs_rank_0.jsonl').read_text())
            self.assertEqual(row['event'], 'manager_action_repair')
            self.assertEqual(row['action_repairs'][0]['kind'], 'ignore_unseen_none')
            self.assertEqual(row['raw_output'], raw)

    def test_plot_failure_preserves_saved_metrics_and_does_not_abort(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root/'training_metrics.jsonl'
            metrics.write_text('{"reward":0.5}\n')
            with patch('scripts.diagnose_rl.render', side_effect=RuntimeError('plot failed')):
                plot_run(root)
            self.assertEqual(metrics.read_text(), '{"reward":0.5}\n')
            warning = json.loads((root/'runtime_warnings.jsonl').read_text())
            self.assertEqual(warning['component'], 'plot')
            self.assertIn('plot failed', warning['error'])

    def test_epoch_checkpoint_survives_validation_failure(self):
        actor = Mock()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def save(path, *args):
                write_json(Path(path)/'lora_adapter/adapter_config.json', {})
                (Path(path)/'lora_adapter/adapter_model.safetensors').write_bytes(b'test')
            actor.save_checkpoint.side_effect = save
            def fail_validation(*args):
                state = json.loads((root/'epoch_1/training_state.json').read_text())
                self.assertEqual(state['global_step'], 3)
                self.assertEqual(state['validation_status'], 'pending')
                actor.save_checkpoint.assert_called_once()
                raise RuntimeError('validation worker failed')
            with patch('verl_memory.train.evaluation', side_effect=fail_validation):
                with self.assertRaisesRegex(RuntimeError, 'validation worker failed'):
                    finish_epoch(actor, [], [1], SimpleNamespace(task='manager'), root, 1, 2, 3, -1., {})
            state = json.loads((root/'epoch_1/training_state.json').read_text())
            self.assertEqual(state['validation_status'], 'failed')
            self.assertEqual(state['global_step'], 3)
            self.assertTrue((root/'epoch_1/lora_adapter/adapter_model.safetensors').exists())

    def test_missing_adapter_export_keeps_checkpoint_and_truthful_best_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, actor = Path(tmp), Mock()
            def save(path, *args):
                Path(path).mkdir(parents=True)
                (Path(path)/'model_world_size_2_rank_0.pt').write_bytes(b'test')
            actor.save_checkpoint.side_effect = save
            with patch('verl_memory.train.evaluation', return_value=({'f1':.4}, [])):
                best = finish_epoch(actor, [], [1], SimpleNamespace(task='manager'), root, 1, 2, 1, .3, {})
            self.assertEqual(best, .4)
            selected = json.loads((root/'best_checkpoint.json').read_text())
            self.assertEqual(selected['checkpoint_type'], 'fsdp')
            self.assertEqual(selected['path'], str(root/'epoch_1'))
            self.assertTrue((root/'epoch_1/training_state.json').exists())
            self.assertIn('adapter_export', (root/'runtime_warnings.jsonl').read_text())

    def test_manager_json_recovers_closers_fences_and_trailing_commas(self):
        expected = [dict(event='ADD', text='literal }, ] and comma, survive')]
        complete = json.dumps(dict(memory=expected))
        cases = [complete[:-1], complete[:-2], complete[:-3],
                 '\ufeffExplanation\n```json\n'+complete[:-1]+'\n```\nExplanation after output.',
                 'Explanation\n```json\n'+complete[:-3]+',},]}\n```\nDone.']
        for raw in cases:
            with self.subTest(raw=raw):
                diagnostics = {}
                self.assertEqual(parse_manager_output(raw, diagnostics), expected)
                self.assertTrue(diagnostics['repaired'])
                self.assertTrue(diagnostics['repairs'])

    def test_manager_json_quoted_punctuation_escapes_and_unicode_are_unchanged(self):
        expected = [dict(event='ADD',text='A says "hello". \\ path; }, ] 中文\nnext line')]
        raw = 'Before\n```json\n'+json.dumps(dict(memory=expected),ensure_ascii=False)+'\n```\nAfter'
        diagnostics = {}
        self.assertEqual(parse_manager_output(raw,diagnostics), expected)
        self.assertFalse(diagnostics['repaired'])

    def test_manager_json_repairs_literal_control_characters_without_changing_text(self):
        diagnostics = {}
        raw = '{"memory":[{"event":"ADD","text":"line1\nline2\ttab"}]}'
        self.assertEqual(parse_manager_output(raw,diagnostics),
                         [dict(event='ADD',text='line1\nline2\ttab')])
        self.assertTrue(diagnostics['repaired'])
        self.assertEqual(len(diagnostics['repairs']),2)

    def test_invalid_action_field_types_cannot_become_memory_text(self):
        for text in (None,True,23,[],dict(fact='not a string')):
            with self.subTest(text=text):
                result = validate_and_apply([],[],[dict(event='ADD',text=text)],
                    dialogue_id='a',source_turn=0,speaker='A',timestamp='today')
                self.assertFalse(result.valid)
                self.assertEqual(result.memory_bank,[])

    def test_manager_json_rejects_truncation_instead_of_salvaging_partial_actions(self):
        cases = ['{"memory":[{"event":"ADD","text":"half',
                 '{"memory":[{"event":"ADD","text":',
                 '{"memory":[{"event":"ADD","text":"done"},',
                 '{"memory":[{"event":"ADD","text":"done","id"',
                 '{"memory":[{"event":"UPDATE","text":"done","id":1',
                 '{"memory":[{"event":"ADD","text":"done"}]}\n{"memory":[{"event":',
                 '{"memory":[{"event":"ADD","text":"bad\\q"}]}',
                 '{"memory":[{"event":"ADD","text":"done"}]]']
        for raw in cases:
            with self.subTest(raw=raw),self.assertRaises(ValueError):
                parse_manager_output(raw)

    def test_manager_json_preserves_invalid_entries_for_local_rejection(self):
        decisions = parse_manager_output('{"memory":[{"event":"ADD","text":"done"},null,"bad",{}]}')
        self.assertEqual(decisions[1:], [None,'bad',{}])
        result = validate_and_apply([],[],decisions,dialogue_id='d',source_turn=0,speaker='A',timestamp='t')
        self.assertFalse(result.valid)
        self.assertEqual([m['text'] for m in result.memory_bank], ['done'])
        self.assertEqual([r['index'] for r in result.rejected_actions], [1,2,3])

    def test_manager_json_does_not_accept_ambiguous_fields_or_nonfinite_values(self):
        cases = ['{"memory":{}}', '{"memory":[],"memory":[{}]}',
                 '{"memory":[{"event":"ADD","event":"DELETE","text":"done"}]}',
                 '{"memory":[{"event":"ADD","text":NaN}]}',
                 '{"memory":[{"event":"ADD","text":1e999}]}',
                 '{"memory":[]}\n{"memory":[{"event":"ADD","text":"different"}]}']
        for raw in cases:
            with self.subTest(raw=raw),self.assertRaises(ValueError):
                parse_manager_output(raw)

    def test_manager_json_accepts_identical_repeated_objects_and_metadata(self):
        raw = '{"reason":"comparison", "memory":[]}'
        diagnostics = {}
        self.assertEqual(parse_manager_output(raw+'\n'+raw,diagnostics), [])
        self.assertEqual(diagnostics['object_count'],2)

    def test_repaired_manager_outputs_and_errors_share_train_test_transition(self):
        from evaluation.manager_validation import manager_transition
        timeline = dict(dialogue_id='test',participants=['A'])
        turn = dict(turn_index=0,speaker='A',timestamp='today',facts=[dict(text='fact')])
        cases = [('{"memory":[{"event":"ADD","text":"complete"}',True),
                 ('{"memory":[{"text":"missing event"}',False),
                 ('{"memory":[{"event":"ADD","text":"cut',False)]
        with patch('evaluation.manager_validation.top_k',lambda q,b,k:b):
            for raw, valid in cases:
                with self.subTest(raw=raw),tempfile.TemporaryDirectory() as tmp:
                    prompt = prompt_for_turn(turn,[],lambda q,b,k:b)
                    expected, _ = manager_transition(dict(turn,dialogue_id='test'),[],lambda p:raw,5)
                    bank, action = commit_turn(timeline,turn,[],prompt,raw,None,tmp,0,1,0,0)
                    self.assertEqual(bank,expected)
                    self.assertEqual(action['valid'],valid)
                    log = Path(tmp)/('manager_repairs_rank_0.jsonl' if valid else 'manager_errors_rank_0.jsonl')
                    row = json.loads(log.read_text())
                    self.assertEqual(row['raw_output'],raw)
                    self.assertEqual(row['valid'],valid)
                    if valid:
                        self.assertEqual(bank[0]['text'],'complete')
                        self.assertTrue(row['parsing']['repaired'])
                        self.assertTrue(row['parsing']['repairs'])
                    else:
                        self.assertEqual(bank,[])

    def test_optimizer_error_propagates_and_nonfinite_metrics_are_only_logged(self):
        class Batch:
            def __init__(self):
                self.batch=dict(attention_mask=torch.ones(2,3),response_mask=torch.ones(2,2),
                                responses=torch.ones(2,2,dtype=torch.long))
                self.meta_info={}
            def __len__(self): return 2
        actor=Mock()
        actor.compute_log_prob.return_value=SimpleNamespace(batch={
            'old_log_probs':torch.zeros(2,2),'entropys':torch.ones(2,2)})
        native_error=RuntimeError('native optimizer failure')
        actor.update_actor.side_effect=native_error
        with self.assertRaises(RuntimeError) as caught:
            apply_update(actor,Batch(),[1.,-1.],2,0.,answer_loss_mask)
        self.assertIs(caught.exception,native_error)
        actor.update_actor.side_effect=None
        actor.update_actor.return_value=SimpleNamespace(meta_info={'metrics':{'actor/pg_loss':[float('nan')]}})
        result=apply_update(actor,Batch(),[1.,-1.],2,0.,answer_loss_mask)
        self.assertFalse(result['skipped_update'])
        self.assertIsNone(result['metrics']['actor/pg_loss'])
        self.assertEqual(result['nonfinite_metrics']['actor/pg_loss'],'nan')
        json.dumps(result,allow_nan=False)

    def test_zero_final_advantage_never_calls_actor_or_weights_mask(self):
        actor, mask = Mock(), Mock()
        result = apply_update(actor,None,[0.,0.,0.,0.],4,.01,mask)
        self.assertTrue(result['skipped_update'])
        self.assertEqual(actor.mock_calls,[])
        mask.assert_not_called()

    def test_nonzero_update_uses_saved_old_probabilities_and_optional_reference(self):
        class Batch:
            def __init__(self):
                self.batch = dict(attention_mask=torch.ones(4,3),response_mask=torch.ones(4,2),
                                  responses=torch.ones(4,2,dtype=torch.long))
                self.meta_info = {}
            def __len__(self): return 4
        for beta in (0.,.01):
            batch, actor = Batch(), Mock()
            old = torch.full((4,2),-2.)
            actor.compute_log_prob.return_value = SimpleNamespace(batch={'old_log_probs':old,'entropys':torch.ones(4,2)})
            actor.compute_ref_log_prob.return_value = SimpleNamespace(batch={'ref_log_prob':torch.full((4,2),-3.)})
            actor.update_actor.return_value = SimpleNamespace(meta_info={'metrics':{'actor/lr':[1e-6,1e-6]}})
            result = apply_update(actor,batch,[1.,-1.,1.,-1.],4,beta,answer_loss_mask)
            self.assertFalse(result['skipped_update'])
            self.assertIs(batch.batch['old_log_probs'],old)
            torch.testing.assert_close(batch.batch['advantages'][:,0],torch.tensor([1.,-1.,1.,-1.]))
            torch.testing.assert_close(batch.batch['response_mask'].sum(-1),torch.ones(4))
            actor.update_actor.assert_called_once_with(batch)
            self.assertEqual(actor.compute_ref_log_prob.call_count,int(bool(beta)))

    def test_teacher_facts_do_not_enter_bank_without_manager_add(self):
        timeline = dict(dialogue_id='test',participants=['A'])
        turn = dict(turn_index=0,speaker='A',timestamp='today',facts=[dict(text='teacher fact')])
        with tempfile.TemporaryDirectory() as tmp:
            prompt = prompt_for_turn(turn,[],lambda q,b,k:b)
            bank,action = commit_turn(timeline,turn,[],prompt,'{"memory":[]}',None,tmp,0,1,0,0)
            self.assertTrue(action['valid'])
            self.assertEqual(bank,[])

    def test_resume_keeps_error_evidence_and_excludes_uncommitted_metrics(self):
        args = dict(task='manager',kind='sf',seed=42,num_generations=4,learning_rate=1e-6,
            beta=.01,credit_interval=4,aux_lambda=.1,aux_tau=.5,manager_model='manager',
            manager_adapter=None,answer_model='answer',answer_adapter='adapter',max_prompt_tokens=4096,
            max_new_tokens=2048,qa_batch_size=4,train_token_budget=8192,smoke_turns=0,resume_from=None)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'; data=Path(tmp)/'data.json'; data.write_text('[]')
            state=resume_state(SimpleNamespace(**args),root,[data])
            state.update(epoch=1,global_step=1)
            write_json(root/'epoch_1/training_state.json',state)
            write_json(root/'evaluation/epoch_2.json',dict(epoch=2))
            (root/'training_metrics.jsonl').write_text('{"epoch":1}\n{"epoch":2}\n')
            evidence=root/'epochs/epoch_2/group_0/attempt_0/error.json'
            write_json(evidence,dict(raw_output='malformed output'))
            args['resume_from']=str(root/'epoch_1')
            restored=resume_state(SimpleNamespace(**args),root,[data])
            self.assertEqual(restored['global_step'],1)
            self.assertEqual(json.loads((root/'training_metrics.jsonl').read_text()),{'epoch':1})
            self.assertFalse((root/'evaluation/epoch_2.json').exists())
            self.assertTrue(list((root/'interrupted').glob('*/evaluation/epoch_2.json')))
            self.assertEqual(json.loads(evidence.read_text())['raw_output'],'malformed output')
            args['answer_adapter']='different_adapter'
            with patch('builtins.print') as warning:
                continued=resume_state(SimpleNamespace(**args),root,[data])
            self.assertEqual(continued['global_step'],1)
            self.assertTrue(any('configuration or data differs' in str(c) for c in warning.call_args_list))

    def test_data_preparation_accepts_qa_without_time_or_evidence(self):
        from scripts.build_manager_training_data import normalize_locomo_sample, validate_questions
        dialogue=normalize_locomo_sample(dict(sample_id='a',conversation={'speaker_a':'A',
            'session_1':[dict(speaker='A',text='hello')]},qa=[dict(question='q',answer='a',category=1)]))
        validate_questions(dialogue)
        self.assertEqual(len(dialogue['questions']),1)
        self.assertNotIn('target_turn',dialogue['questions'][0])

    def test_bank_transition_matches_formal_test_with_none_and_rejection(self):
        from evaluation.manager_validation import manager_transition
        timeline = dict(dialogue_id='test',participants=['A'])
        bank = [dict(id='id1',text='original',speaker='A',timestamp='yesterday',source_turn=0)]
        cases = [
            [dict(id='id1',event='UPDATE',text='updated',old_memory='original')],
            [dict(id='id1',event='NONE',text='ignored')],
            [dict(id='id1',event='NONE'),dict(id='id1',event='UPDATE',text='updated')],
            [dict(id='id1',event='DELETE'),dict(id='id1',event='NONE')],
            [dict(id='id1',event='NONE'),dict(id='id1',event='NONE')],
            [dict(id='id1',event='UPDATE',text='conflict'),dict(id='id1',event='DELETE')],
            [dict(id='id1',event='DELETE'),dict(id='id1',event='UPDATE',text='conflict')],
            [dict(id='id1',event='UPDATE',text='conflict'),dict(id='id1',event='DELETE'),
             dict(event='ADD',text='accepted')],
            [dict(id='hidden',event='UPDATE',text='invalid')],
            [dict(id='id1',event='DELETE')],
            [dict(event='ADD',text=f'new{i}') for i in range(7)],
        ]
        turn = dict(turn_index=1,speaker='A',timestamp='today',facts=[dict(
            text='new fact', speaker='A', timestamp='today', key='fact-key',
            memory_type='UserMemory', tags=['tag'])])
        with patch('evaluation.manager_validation.top_k',lambda q,b,k:b):
            for decisions in cases:
                with self.subTest(decisions=decisions),tempfile.TemporaryDirectory() as tmp:
                    prompt = prompt_for_turn(turn,bank,lambda q,b,k:b)
                    self.assertEqual(prompt,build_manager_input(bank,turn['facts']))
                    old_input = prompt.rsplit('\n\nOld Memory:\n', 1)[1].split('\n\nRetrieved facts:\n', 1)[0]
                    self.assertEqual(json.loads(old_input), [dict(id='id1', text='original')])
                    facts_input = prompt.rsplit('\n\nRetrieved facts:\n', 1)[1].removesuffix('\n\nNew Memory:\n')
                    self.assertEqual(json.loads(facts_input), ['new fact'])
                    self.assertEqual(turn['facts'][0]['tags'], ['tag'])
                    raw = json.dumps({'memory':decisions})
                    expected, action = manager_transition(dict(turn,dialogue_id='test'),bank,lambda p:raw,5)
                    self.assertEqual(action['prompt'], prompt)
                    actual, actual_action = commit_turn(timeline,turn,bank,prompt,raw,None,tmp,0,1,0,0)
                    self.assertEqual(actual,expected)
                    self.assertEqual(actual_action['valid'],action['valid'])
                    self.assertEqual(actual_action['id_mappings'],action['id_mappings'])
                    if action.get('partial_applied'):
                        self.assertEqual(actual_action['rejected_ids'], ['id1'])
                        self.assertEqual(actual_action['actions'], 1)
                        self.assertEqual([m['text'] for m in actual], ['original', 'accepted'])
                    rows = [json.loads(l) for l in (Path(tmp)/'trajectories_rank_0.jsonl').read_text().splitlines()]
                    self.assertEqual(len(rows),1)
                    self.assertEqual(rows[0]['id_mappings'], action['id_mappings'])
                    self.assertTrue((Path(tmp)/rows[0]['memory_bank_ref']['path']).is_file())
                    if any(r['kind']=='update_over_delete' for r in action.get('action_repairs', [])):
                        self.assertTrue(action['valid'])
                        self.assertEqual(actual[0]['text'], 'conflict')
                        repair = json.loads((Path(tmp)/'manager_repairs_rank_0.jsonl').read_text())
                        self.assertEqual(repair['action_repairs'],action['action_repairs'])
                    if not action['valid']:
                        error = json.loads((Path(tmp)/'manager_errors_rank_0.jsonl').read_text())
                        self.assertEqual(error['raw_output'],raw)
                        self.assertEqual(error['id_mappings'],action['id_mappings'])
                        self.assertEqual(error['partial_applied'], action.get('partial_applied', False))
                        self.assertEqual(error['applied_actions'], action['actions'])

    def test_reward_matches_existing_population_std_then_qa_mean(self):
        rewards = [[0.,.3,1.],[.1,.3,.6],[.7,.3,.3],[.8,.3,0.]]
        matrix = torch.tensor(rewards,dtype=torch.float32)
        std = matrix.std(dim=0,correction=0)
        expected = torch.where(std>1e-12,(matrix-matrix.mean(0))/(std+1e-6),0).mean(1)
        torch.testing.assert_close(torch.tensor(per_qa_advantages(rewards)),expected)
        self.assertEqual(per_qa_advantages([[.3],[.3],[.3],[.3]]),[0.,0.,0.,0.])

    def test_weighted_verl_loss_and_gradients_equal_trajectory_objective(self):
        ids = np.array([0,0,1,1])
        binary = torch.tensor([[1,0,0],[1,1,1],[1,1,0],[1,0,0]],dtype=torch.float32)
        data = SimpleNamespace(batch={'response_mask':binary.clone()},non_tensor_batch={'trajectory_id':ids})
        values = torch.arange(12,dtype=torch.float32).reshape(4,3).requires_grad_()
        weight_loss_mask(data,2)
        # Actual verl seq-mean-token-sum reduction, also used for KL.
        loss = (values*data.batch['response_mask']).sum(-1).mean()
        expected = ((values[:2]*binary[:2]).sum()/4+(values[2:]*binary[2:]).sum()/3)/2
        torch.testing.assert_close(loss,expected)
        actual_grad = torch.autograd.grad(loss,values,retain_graph=True)[0]
        reference_grad = torch.autograd.grad(expected,values)[0]
        torch.testing.assert_close(actual_grad,reference_grad)

    def test_block_scores_keep_signed_bounded_credit_and_final_partial_block(self):
        values, records = block_advantages([dict(after_turn=0,phi=-2),dict(after_turn=4,phi=-1),
            dict(after_turn=5,phi=-1.25)],.2,5,4,.1,.5)
        self.assertAlmostEqual(values[0],.3)
        self.assertAlmostEqual(values[4],.15)
        self.assertEqual(len(records),2)

    def test_sf_uses_generated_f1_reuses_terminal_answers_and_keeps_question_intersection(self):
        from verl_memory.train import score_trajectories
        from memfactory.diagnostics import EvidenceStore
        args = SimpleNamespace(kind='sf', aux_lambda=.1, aux_tau=.5, credit_interval=1,
                               max_prompt_tokens=4096, max_new_tokens=2048)
        questions = [dict(question_id='q0', question='q0', answer='blue', sf_ready_after_turn=1),
                     dict(question_id='q1', question='q1', answer='green', sf_ready_after_turn=2),
                     dict(question_id='q2', question='q2', answer='gold', sf_ready_after_turn=None)]
        timeline = dict(dialogue_id='d', participants=['A'], questions=questions,
                        turns=[dict(turn_index=i) for i in range(3)])
        calls = []
        def generate(workers, requests, args, root, label, **kwargs):
            self.assertFalse(kwargs.get('gold', False))
            self.assertTrue(all('gold' not in r for r in requests))
            calls.append((label, requests))
            result = []
            for r in requests:
                question, bank = r['prompt'].split(':')
                prediction = {'q0': {'a':'blue extra', 'b':'blue', 'c':'red'},
                              'q1': {'b':'green', 'c':'green'}, 'q2': {'c':'gold'}}[question][bank]
                result.append(dict(prediction=prediction, raw_answer=prediction,
                                   hit_token_limit=False, output_tokens=2))
            return result
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(tmp)
            trajectories = [dict(trajectory_id=i, bank=['c'], bank_points=[
                dict(after_turn=j, memory_bank_ref=store.save(bank))
                for j,bank in enumerate([[], ['a'], ['b'], ['c']])]) for i in range(2)]
            with patch('verl_memory.train.score_requests', side_effect=generate), patch(
                    'verl_memory.train.answer_prompt',
                    side_effect=lambda q,b,*a: (q+':'+(b[0] if b else 'empty'), [])):
                advantages, credit = score_trajectories(trajectories,timeline,[],args,tmp,1,0)
            self.assertEqual([label for label,_ in calls], ['terminal_qa','sf_f1'])
            # Three unique intermediate prompts, shared across both trajectories;
            # the final snapshots reuse the terminal QA outputs.
            self.assertEqual(len(calls[1][1]), 3)
            self.assertEqual({r['prompt'] for r in calls[1][1]}, {'q0:a','q0:b','q1:b'})
            self.assertAlmostEqual(advantages[0][0], 0.)  # first eligible snapshot
            self.assertAlmostEqual(advantages[0][1], .1*(1-2/3)/.5)
            self.assertAlmostEqual(advantages[0][2], -.1)
            self.assertEqual(credit[1]['common_question_count'], 1)
            self.assertEqual(credit[2]['common_question_count'], 2)
            rows = [json.loads(line) for line in (Path(tmp)/'bank_scores.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['score_metric']=='generated_f1' and 'ell' not in r for r in rows))
            from verl_memory.protocol import load_bank
            evidence = load_bank(tmp, rows[0]['score_evidence_ref'])
            self.assertEqual(evidence['prediction'], 'blue extra')
            self.assertAlmostEqual(evidence['f1'], 2/3)

    def test_sf_zero_lambda_and_validation_do_not_score_snapshots(self):
        from verl_memory.train import score_trajectories
        timeline = dict(dialogue_id='d', participants=['A'], turns=[dict(turn_index=0)],
                        questions=[dict(question_id='q', question='q', answer='blue')])
        output = dict(prediction='blue', hit_token_limit=False, output_tokens=1)
        for kind,weight,training in [('sf',0.,True),('sa',.1,True),('sf',.1,False)]:
            with self.subTest(kind=kind,weight=weight,training=training), tempfile.TemporaryDirectory() as tmp:
                args = SimpleNamespace(kind=kind, aux_lambda=weight, max_prompt_tokens=4096,max_new_tokens=2048)
                trajectories = [dict(trajectory_id=i,bank=[]) for i in range(2)]
                with patch('verl_memory.train.score_requests', return_value=[output,output]) as score, patch(
                        'verl_memory.train.answer_prompt', return_value=('prompt',[])):
                    _,credit = score_trajectories(trajectories,timeline,[],args,tmp,1,0,training=training)
                score.assert_called_once()
                self.assertEqual(credit, [])

    def test_sf_all_snapshot_prompts_cached_closes_answer_phase_without_generation(self):
        import sys
        from verl_memory.train import score_trajectories
        from memfactory.diagnostics import EvidenceStore
        args = SimpleNamespace(kind='sf', aux_lambda=.1, aux_tau=.5, credit_interval=4,
                               max_prompt_tokens=4096,max_new_tokens=2048)
        timeline = dict(dialogue_id='d',participants=['A'],turns=[dict(turn_index=0)],
                        questions=[dict(question_id='q',question='q',answer='blue',sf_ready_after_turn=1)])
        output = dict(prediction='blue',hit_token_limit=False,output_tokens=1)
        worker = SimpleNamespace(sleep=SimpleNamespace(remote=Mock(return_value='asleep')))
        fake_ray = SimpleNamespace(get=Mock())
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(tmp)
            trajectory = dict(trajectory_id=0,bank=[],bank_points=[
                dict(after_turn=0,memory_bank_ref=store.save([])),
                dict(after_turn=1,memory_bank_ref=store.save([]))])
            with patch.dict(sys.modules, {'ray':fake_ray}), patch(
                    'verl_memory.train.score_requests',return_value=[output]) as score, patch(
                    'verl_memory.train.answer_prompt',return_value=('prompt',[])):
                _,credit = score_trajectories([trajectory],timeline,[worker],args,tmp,1,0)
            score.assert_called_once()
            worker.sleep.remote.assert_called_once_with()
            fake_ray.get.assert_called_once_with(['asleep'])
            self.assertEqual(credit[0]['delta'], 0.)

    def test_block_scores_compare_only_common_questions(self):
        points = [
            dict(after_turn=0, phi=-2, scores_by_question={'q0': -2}),
            dict(after_turn=4, phi=-6, scores_by_question={'q0': -1, 'q1': -10}),
        ]
        values, records = block_advantages(points, 0., 4, 4, .1, .5)
        self.assertAlmostEqual(records[0]['delta'], 1.)
        self.assertEqual(records[0]['common_question_count'], 1)
        self.assertAlmostEqual(records[0]['auxiliary_advantage'], .1)
        self.assertAlmostEqual(values[0], .1)

    def test_block_scores_with_no_common_questions_has_zero_auxiliary_credit(self):
        points = [
            dict(after_turn=0, scores_by_question={}),
            dict(after_turn=4, scores_by_question={'q1': -1}),
        ]
        values, records = block_advantages(points, .2, 4, 4, .1, .5)
        self.assertEqual(records[0]['common_question_count'], 0)
        self.assertEqual(records[0]['delta'], 0.)
        self.assertEqual(records[0]['auxiliary_advantage'], 0.)
        self.assertAlmostEqual(values[0], .2)

    def test_linked_question_eligibility_waits_for_all_evidence_turns(self):
        timeline = dict(dialogue_id='d', turns=[
            dict(turn_index=i, dia_id=f'D1:{i + 1}') for i in range(4)
        ], questions=[
            dict(question_id='single'), dict(question_id='multiple'),
            dict(question_id='unmapped'),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'linked.jsonl'
            rows = [
                dict(dialogue_id='d', linked_questions=[
                    dict(question_id='single', turn_index=1, evidence=['D1:02']),
                    dict(question_id='multiple', turn_index=1, evidence=['D1:1', 'D1:4']),
                    dict(question_id='unmapped', turn_index=0, evidence=['D9:99']),
                ])
            ]
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
            attach_linked_question_turns([timeline], path)

        questions = {q['question_id']: q for q in timeline['questions']}
        self.assertEqual(questions['single']['sf_ready_after_turn'], 2)
        self.assertEqual(questions['single']['evidence_turn_indices'], [1])
        self.assertEqual(questions['multiple']['sf_ready_after_turn'], 4)
        self.assertEqual(questions['multiple']['evidence_turn_indices'], [0, 3])
        self.assertIsNone(questions['unmapped']['sf_ready_after_turn'])
        self.assertEqual(questions['unmapped']['evidence_turn_indices'], [])

    def test_no_qa_or_gold_in_manager_prompt(self):
        turn = dict(facts=[dict(text='fact')],question='SECRET_QA',answer='SECRET_GOLD')
        prompt = prompt_for_turn(turn,[],lambda q,b,k:b)
        self.assertNotIn('SECRET',prompt)

    def test_nonchronological_data_rejected(self):
        self.assertEqual(validate_timelines([dict(dialogue_id='a',participants=['A'],
                turns=[dict(turn_index=3,speaker='A',facts=[])],
                questions=[dict(question_id=0,question='q',answer='a')])]), [])


if __name__ == '__main__':
    unittest.main()
