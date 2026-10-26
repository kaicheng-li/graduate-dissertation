import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from evaluation import score_judge as j
from scripts import diagnose_rl


def row(i=0):
    return dict(dialogue_id='a', question_id=str(i), question='What color?', gold='blue',
                prediction='It is blue', category=1, em=0., f1=.5, bleu1=.5)


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace(backend='local', max_workers=1, retries=1, retry_delay=0,
            save_every=1, gpus='0,1,2,3', local_model_path='gpt-oss', max_tokens=2048, max_prompt_tokens=4096)
        self.info = {'model':'gpt-oss', 'prompt':'fixture'}
        self.judge = Mock()
        self.judge.generate.return_value = '{"reason":"Same color", "label":"CORRECT"}'

    def test_ambiguous_labels_do_not_get_silently_accepted(self):
        self.assertIsNone(j.extract_label('could be CORRECT or WRONG'))
        self.assertIsNone(j.extract_label('{"label":"CORRECT"}{"label":"WRONG"}'))
        self.assertEqual(j.extract_label('{"reason":"Not WRONG", "label":"CORRECT"}'), 'CORRECT')

    def test_jsonl_scores_resume_and_rejudge_changed_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'test.jsonl'
            j.write_jsonl_atomic(path, [row()])
            path.with_suffix('.json').write_text('{"component":"answer", "qa":1}')
            self.assertTrue(j.score_file(path, self.judge, self.args, self.info))
            self.assertTrue(j.score_file(path, self.judge, self.args, self.info))
            self.assertEqual(self.judge.generate.call_count, 1)
            summary = json.loads(path.with_suffix('.json').read_text())
            self.assertEqual(summary['paper_metrics']['multi_hop']['llm_judge'], 1)
            self.assertEqual(summary['paper_metrics']['overall']['em'], 0)
            self.assertTrue(path.with_suffix('.judge_metrics.csv').is_file())
            records = j.read_jsonl(path)
            records[0]['prediction'] = 'red'
            j.write_jsonl_atomic(path, records)
            self.judge.generate.return_value = 'ambiguous'
            self.assertFalse(j.score_file(path, self.judge, self.args, self.info))
            summary = json.loads(path.with_suffix('.json').read_text())
            self.assertIsNone(summary['llm_judge'])
            self.assertEqual(summary['llm_judge_coverage'], 0)

    def test_validation_json_updates_both_modes_and_chart_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'evaluation').mkdir()
            path=root/'evaluation/epoch_0.json'
            document={mode:dict(summary={'f1':.5,'em':0.,'qa':1},records=[row()])
                      for mode in ('answer','roll_forward')}
            document['epoch']=0
            path.write_text(json.dumps(document))
            j.score_file(path,self.judge,self.args,self.info)
            report=diagnose_rl.analyze(root)
            self.assertEqual(report['epochs'][0]['llm_judge'],1)
            self.assertEqual(report['epochs'][1]['llm_judge_coverage'],1)

    def test_plot_accepts_skipped_group_without_reward(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'training_metrics.jsonl').write_text(json.dumps(dict(
                epoch=1,group=0,global_step=0,skipped_update=True,skip_reason='no_manager_actions'))+'\n')
            result=diagnose_rl.render(root)
            self.assertEqual(len(result['groups']),1)
            self.assertTrue((root/'plots/training_curves.png').exists())
            self.assertTrue((root/'plots/training_metrics.csv').exists())

    def test_four_gpu_shards_merge_without_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'test.jsonl'
            j.write_jsonl_atomic(path,[row(i) for i in range(9)])
            devices=[]
            def launch(command, **kwargs):
                devices.append(kwargs['env']['CUDA_VISIBLE_DEVICES'])
                shard=Path(command[3])
                j.score_file(shard,self.judge,self.args,self.info)
                process=Mock(); process.poll.return_value=0; process.returncode=0
                return process
            with patch.object(j.subprocess,'Popen',side_effect=launch):
                self.assertTrue(j.parallel_local(path,self.args,self.info))
            self.assertEqual(devices,['0','1','2','3'])
            records=j.read_jsonl(path)
            self.assertEqual([r['question_id'] for r in records],list(map(str,range(9))))
            self.assertTrue(all(r['llm_judge']==1 for r in records))
            self.assertTrue(all('_judge_index' not in r for r in records))


if __name__=='__main__':
    unittest.main()
