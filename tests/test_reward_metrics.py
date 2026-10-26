"""Deterministic tests for the shared LoCoMo reward and diagnostics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import f1_score, extract_answer
from memfactory.locomo_split import questions_with_raw_ids, select_locomo_split
from memfactory.manager_env import validate_and_apply


class TokenF1Tests(unittest.TestCase):
    def test_answer_extraction_excludes_explicit_explanations(self):
        for tail in ('\n\n**Explanation:** extra words', '\nExplanation: extra words',
                     '\n## Explanation\nextra words', ' **Explanation**: extra words',
                     '\n**Reasoning:** extra words'):
            for label in ('**Answer:**', '**Answer**:', 'Answer:', 'Final Answer:'):
                raw = f'{label} Paris{tail}'
                with self.subTest(raw=raw):
                    self.assertEqual(extract_answer(raw),'Paris')
                    self.assertEqual(f1_score(extract_answer(raw),'Paris'),1.)

    def test_answer_extraction_preserves_lists_multiline_and_plain_answers(self):
        answer = 'Paris\nLondon\nNew York'
        self.assertEqual(extract_answer('**Answer:** '+answer+'\nExplanation: details'), answer)
        self.assertEqual(extract_answer(answer),answer)
        self.assertEqual(extract_answer('<answer>Paris</answer>\nExplanation: details'),'Paris')
        self.assertEqual(extract_answer('Memories: "Answer: old"\n**Answer:** Paris'),'Paris')
        self.assertEqual(extract_answer(None),'')
        self.assertEqual(extract_answer('**Answer:**\n**Explanation:** missing answer'),'')
        # No arbitrary 6-word truncation or sentence splitting.
        answer='The answer has more than six words and a date: May 8, 2023.'
        self.assertEqual(extract_answer('**Answer:** '+answer),answer)

    def test_null_gold_never_becomes_a_literal_none_match(self):
        from evaluation.metrics import score_record
        with self.assertRaisesRegex(ValueError,'gold'):
            score_record(dict(prediction='None',gold=None))
        with self.assertRaises(ValueError): f1_score('None',None)
        record=dict(prediction=None,gold='Paris')
        score_record(record)
        self.assertEqual(record['f1'],0.)
    def test_exact_match_is_one(self) -> None:
        self.assertEqual(f1_score("blue bicycle", "blue bicycle"), 1.0)

    def test_partial_overlap_is_fractional(self) -> None:
        self.assertAlmostEqual(f1_score("blue bicycle", "blue car"), 0.5)

    def test_articles_case_and_punctuation_are_normalized(self) -> None:
        self.assertEqual(f1_score("The Blue, bicycle!", "blue bicycle"), 1.0)

    def test_no_overlap_is_zero(self) -> None:
        self.assertEqual(f1_score("red car", "blue bicycle"), 0.0)


class DataIdentityTests(unittest.TestCase):
    def test_raw_question_id_survives_category_filtering(self) -> None:
        sample = {
            "sample_id": "conv-30",
            "qa": [
                {"question": "first", "category": 1},
                {"question": "excluded", "category": 5},
                {"question": "third", "category": 4},
            ],
        }
        self.assertEqual(
            [row["question_id"] for row in questions_with_raw_ids(sample)],
            ["0", "1", "2"],
        )
        selected = select_locomo_split([sample], "validation")
        self.assertEqual(
            [row["question_id"] for row in selected[0]["qa"]], ["0", "2"]
        )


class ManagerEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bank = [{
            "id": "conv-1:m0", "text": "old", "speaker": "A",
            "timestamp": "t0", "source_turn": 0,
        }]

    def test_illegal_hidden_id_is_rejected_atomically(self) -> None:
        result = validate_and_apply(
            self.bank, [], [{"event": "DELETE", "id": "conv-1:m0"}],
            dialogue_id="conv-1", source_turn=1, speaker="A", timestamp="t1",
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.memory_bank, self.bank)

    def test_original_none_event_preserves_memory_and_noop_is_not_an_event(self):
        from memfactory.modules.memory_updater import UPDATE_MEMORY_PROMPT, build_manager_input
        prompt = build_manager_input(self.bank, ['old'])
        self.assertTrue(prompt.startswith(UPDATE_MEMORY_PROMPT))
        self.assertIn('"event": "NONE"', prompt)
        self.assertNotIn('Do not return NONE', prompt)
        self.assertNotIn('NOOP', prompt)
        for event, valid in [('NONE', True), ('NOOP', False)]:
            result = validate_and_apply(self.bank, self.bank,
                [dict(event=event, id='conv-1:m0')], dialogue_id='conv-1',
                source_turn=1, speaker='A', timestamp='t1')
            self.assertEqual(result.valid, valid)
            self.assertEqual(result.applied_actions, 0)
            self.assertEqual(result.memory_bank, self.bank)

    def test_valid_add_gets_provenance(self) -> None:
        result = validate_and_apply(
            self.bank, self.bank, [{"event": "ADD", "text": "new"}],
            dialogue_id="conv-1", source_turn=1, speaker="B", timestamp="t1",
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.memory_bank[-1]["id"], "0")
        self.assertEqual(result.memory_bank[-1]["speaker"], "B")
        self.assertEqual(result.memory_bank[-1]["source_turn"], 1)

    def test_repeated_mutations_preserve_conflicting_target(self) -> None:
        for events in (
            ("UPDATE", "UPDATE"), ("DELETE", "DELETE"),
            ("UPDATE", "DELETE", "UPDATE"),
        ):
            with self.subTest(events=events):
                decisions = [dict(event=event, id="conv-1:m0", text=f"text-{i}")
                             for i, event in enumerate(events)]
                result = validate_and_apply(
                    self.bank, self.bank, decisions,
                    dialogue_id="conv-1", source_turn=1, speaker="A", timestamp="t1",
                )
                self.assertFalse(result.valid)
                self.assertIn("duplicate target id", result.error)
                self.assertEqual(result.applied_actions, 0)
                self.assertEqual(result.memory_bank, self.bank)

    def test_update_over_delete_in_both_orders_preserves_other_operations(self):
        bank = self.bank + [dict(id='other', text='keep')]
        pair = [dict(event='UPDATE', id='conv-1:m0', text='updated'),
                dict(event='DELETE', id='conv-1:m0')]
        for ordered in (pair, list(reversed(pair))):
            actions = ordered + [dict(event='NONE',id='conv-1:m0'),
                                 dict(event='DELETE',id='other'),dict(event='ADD',text='new')]
            result = validate_and_apply(bank, bank, actions, dialogue_id='d',
                                        source_turn=1, speaker='A', timestamp='now')
            self.assertTrue(result.valid)
            self.assertEqual(result.applied_actions, 3)
            self.assertEqual({m['id']:m['text'] for m in result.memory_bank},
                             {'conv-1:m0':'updated','0':'new'})
            repair = next(r for r in result.repairs if r['kind']=='update_over_delete')
            self.assertEqual(actions[repair['selected_index']]['event'], 'UPDATE')
            self.assertEqual([actions[i]['event'] for i in repair['ignored_indices']], ['DELETE'])
            self.assertEqual(bank[0]['text'], 'old')

    def test_none_is_compatible_with_one_mutation_in_either_order(self) -> None:
        for events in (("NONE", "UPDATE"), ("UPDATE", "NONE"),
                       ("NONE", "DELETE"), ("DELETE", "NONE"),
                       ("NONE", "NONE")):
            with self.subTest(events=events):
                result = validate_and_apply(
                    self.bank, self.bank,
                    [dict(event=e, id="conv-1:m0", text="updated" if e == "UPDATE" else "ignored")
                     for e in events],
                    dialogue_id="conv-1", source_turn=1, speaker="A", timestamp="t1")
                self.assertTrue(result.valid, result.error)
                self.assertEqual(result.applied_actions, int(any(e != "NONE" for e in events)))
                if "DELETE" in events:
                    self.assertEqual(result.memory_bank, [])
                elif "UPDATE" in events:
                    self.assertEqual(result.memory_bank[0]["text"], "updated")
                else:
                    self.assertEqual(result.memory_bank, self.bank)
                self.assertEqual(self.bank[0]["text"], "old")

    def test_conflicting_id_does_not_discard_other_ids_or_adds(self) -> None:
        bank = self.bank + [dict(self.bank[0], id="other"), dict(self.bank[0], id="deleted")]
        decisions = [dict(event="UPDATE", id="conv-1:m0", text="conflict"),
                     dict(event="UPDATE", id="other", text="accepted"),
                     dict(event="DELETE", id="deleted"),
                     dict(event="NONE", id="other"),
                     dict(event="UPDATE", id="conv-1:m0", text="other conflict"),
                     dict(event="ADD", text="new")]
        for ordered in (decisions, list(reversed(decisions))):
            result = validate_and_apply(bank, bank, ordered, dialogue_id="conv-1",
                                        source_turn=1, speaker="A", timestamp="t1")
            self.assertFalse(result.valid)  # Errors remain visible, even with partial execution.
            self.assertEqual(result.rejected_ids, ("conv-1:m0",))
            self.assertEqual(result.applied_actions, 3)
            self.assertEqual({m['id']: m['text'] for m in result.memory_bank},
                             {'conv-1:m0': 'old', 'other': 'accepted', '0': 'new'})
            self.assertEqual([m['text'] for m in bank], ['old', 'old', 'old'])

    def test_more_than_five_distinct_changes_are_accepted(self) -> None:
        bank = [dict(self.bank[0], id=f"conv-1:m{i}") for i in range(8)]
        decisions = [dict(event="UPDATE", id=m["id"], text=f"updated {i}")
                     for i, m in enumerate(bank)]
        decisions += [dict(event="ADD", id="ignored", text=f"new {i}") for i in range(7)]
        decisions = ([dict(event="NONE", id=m["id"]) for m in bank]
                     + decisions + [dict(event="NONE", id=m["id"]) for m in bank])
        result = validate_and_apply(bank, bank, decisions,
            dialogue_id="conv-1", source_turn=1, speaker="A", timestamp="t1")
        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.applied_actions, 15)
        self.assertEqual(len(result.memory_bank), 15)
        self.assertEqual(len({m["id"] for m in result.memory_bank}), 15)
        self.assertEqual([m["text"] for m in bank], ["old"] * 8)

    def test_bad_action_cannot_block_valid_same_id_in_either_order(self) -> None:
        for bad in (dict(event='UNKNOWN', id='conv-1:m0'),
                    dict(event='UPDATE', id='conv-1:m0', text=None),
                    dict(event='ADD', id='conv-1:m0', text='')):
            for event in ('UPDATE', 'DELETE'):
                good = dict(event=event, id='conv-1:m0', text='accepted')
                for decisions in ([bad, good], [good, bad]):
                    with self.subTest(decisions=decisions):
                        result = validate_and_apply(self.bank, self.bank, decisions,
                            dialogue_id='conv-1', source_turn=1, speaker='A', timestamp='t1')
                        self.assertFalse(result.valid)
                        self.assertEqual(result.applied_actions, 1)
                        self.assertEqual(len(result.rejected_actions), 1)
                        self.assertEqual([m['text'] for m in result.memory_bank],
                                         ['accepted'] if event == 'UPDATE' else [])

    def test_unknown_id_never_resolves_by_text_regardless_of_order(self) -> None:
        bank = [dict(self.bank[0], id='first', text='first old'),
                dict(self.bank[0], id='second', text='second old')]
        actions = [dict(event='UPDATE', id='first', text='second old'),
                   dict(event='UPDATE', id='placeholder', old_memory='second old', text='accepted')]
        for decisions in (actions, list(reversed(actions))):
            result = validate_and_apply(bank, bank, decisions,
                dialogue_id='conv-1', source_turn=1, speaker='A', timestamp='t1')
            self.assertFalse(result.valid)
            self.assertEqual(result.rejected_ids, ('placeholder',))
            self.assertEqual(result.applied_actions, 1)
            self.assertEqual({m['id']:m['text'] for m in result.memory_bank},
                             {'first':'second old', 'second':'second old'})

    def test_same_turn_add_alias_resolves_in_either_order(self) -> None:
        actions = [dict(event='ADD', id='conv-1:t1:m0', text='new'),
                   dict(event='UPDATE', id='conv-1:t1:m0', text='changed')]
        for decisions in (actions, list(reversed(actions))):
            result = validate_and_apply(self.bank, self.bank, decisions,
                dialogue_id='conv-1', source_turn=1, speaker='A', timestamp='t1')
            self.assertTrue(result.valid)
            self.assertEqual(result.applied_actions, 1)
            self.assertEqual([m['text'] for m in result.memory_bank], ['old', 'changed'])
            self.assertEqual({m['bank_id'] for m in result.id_mappings}, {'0'})

    def test_ids_use_full_bank_max_and_aliases_do_not_overwrite_existing_ids(self):
        bank = [dict(id='2', text='visible'), dict(id='19', text='hidden'),
                dict(id='legacy:t1:m4', text='legacy')]
        actions = [dict(event='ADD', id='2', text='added'),
                   dict(event='UPDATE', id='2', text='existing updated'),
                   dict(event='ADD', id='temporary', text='second'),
                   dict(event='NONE', id='temporary')]
        result = validate_and_apply(bank, bank[:1], actions, dialogue_id='d',
                                    source_turn=2, speaker='A', timestamp='now')
        self.assertTrue(result.valid)
        self.assertEqual({m['id']:m['text'] for m in result.memory_bank},
                         {'2':'existing updated','19':'hidden','legacy:t1:m4':'legacy',
                          '20':'added','21':'second'})
        self.assertEqual([m['bank_id'] for m in result.id_mappings], ['20','2','21','21'])
        self.assertEqual(bank[0]['text'], 'visible')

    def test_duplicate_add_alias_rejects_only_ambiguous_reference(self):
        actions = [dict(event='ADD',id='new',text='one'),
                   dict(event='ADD',id='new',text='two'),
                   dict(event='DELETE',id='new'),
                   dict(event='UPDATE',id='conv-1:m0',text='accepted')]
        for rows in (actions, list(reversed(actions))):
            result = validate_and_apply(self.bank,self.bank,rows,dialogue_id='d',
                                        source_turn=1,speaker='A',timestamp='now')
            self.assertFalse(result.valid)
            self.assertEqual(len(result.rejected_actions),1)
            self.assertEqual(result.applied_actions,3)
            self.assertEqual({m['text'] for m in result.memory_bank},{'accepted','one','two'})

    def test_new_alias_delete_and_missing_ids(self):
        for event in ('DELETE','NONE'):
            actions = [dict(event=event,id='placeholder'),
                       dict(event='ADD',id='placeholder',text='one'),
                       dict(event='ADD',id=None,text='two'),
                       dict(event='ADD',id=True,text='three')]
            result = validate_and_apply([],[],actions,dialogue_id='d',source_turn=1,
                                        speaker='A',timestamp='now')
            self.assertTrue(result.valid)
            self.assertEqual([m['text'] for m in result.memory_bank],
                             ['two','three'] if event == 'DELETE' else ['one','two','three'])
            self.assertEqual(result.applied_actions,len(result.memory_bank))

    def test_unknown_delete_never_uses_visible_or_hidden_text(self):
        bank=[dict(id='0',text='one'),dict(id='1',text='hidden')]
        result=validate_and_apply(bank,bank[:1],[dict(event='DELETE',id='wrong',text='one'),
            dict(event='DELETE',id='wrong-hidden',text='hidden'),dict(event='ADD',text='new')],
            dialogue_id='d',source_turn=1,speaker='A',timestamp='now')
        self.assertFalse(result.valid)
        self.assertEqual({m['id']:m['text'] for m in result.memory_bank},
                         {'0':'one','1':'hidden','2':'new'})
        self.assertEqual(len(result.rejected_actions),2)
        self.assertEqual(result.applied_actions,1)

    def test_cross_turn_updates_and_new_ids_follow_current_bank(self) -> None:
        bank = self.bank
        for turn in (1, 2):
            result = validate_and_apply(bank, bank,
                [dict(event="UPDATE", id="conv-1:m0", text=f"version {turn}")],
                dialogue_id="conv-1", source_turn=turn, speaker="A", timestamp=str(turn))
            self.assertTrue(result.valid, result.error)
            bank = result.memory_bank
        self.assertEqual(bank[0]["text"], "version 2")
        created = validate_and_apply(bank, bank,
            [dict(event="ADD", text=str(i)) for i in range(12)],
            dialogue_id="conv-1", source_turn=3, speaker="A", timestamp="3")
        old_ids = {m["id"] for m in created.memory_bank[1:]}
        deleted = validate_and_apply(created.memory_bank, created.memory_bank,
            [dict(event="DELETE", id=i) for i in sorted(old_ids)],
            dialogue_id="conv-1", source_turn=4, speaker="A", timestamp="4")
        self.assertTrue(deleted.valid, deleted.error)
        added = validate_and_apply(deleted.memory_bank, deleted.memory_bank,
            [dict(event="ADD", text="later")],
            dialogue_id="conv-1", source_turn=5, speaker="A", timestamp="5")
        self.assertTrue(added.valid, added.error)
        self.assertEqual(added.memory_bank[-1]["id"], "0")  # Current bank has no numeric IDs left.

    def test_bad_action_after_valid_duplicate_still_rejects_atomically(self) -> None:
        result = validate_and_apply(
            self.bank, self.bank,
            [dict(event="UPDATE", id="conv-1:m0", text="a"),
             dict(event="UPDATE", id="conv-1:m0", text="b"),
             dict(event="DELETE", id="hidden")],
            dialogue_id="conv-1", source_turn=1, speaker="A", timestamp="t1",
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.memory_bank, self.bank)


if __name__ == "__main__":
    unittest.main()
