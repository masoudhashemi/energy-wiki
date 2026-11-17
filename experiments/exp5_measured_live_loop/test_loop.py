import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("conditional_loop", Path(__file__).with_name("run_exp5.py"))
loop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loop)


def evaluation(value=1):
    return {**{key: value for key in loop.LIMITS},
            "evidence": {key: "Specific missing explanation" for key in loop.LIMITS},
            "critic_summary": "Revise the explanation", "grad_x_directives": ["Clarify the document"]}


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def call(self, purpose, system, payload):
        self.requests.append((purpose, system, copy.deepcopy(payload)))
        return copy.deepcopy(next(self.responses))


class ConditionalSynthesisTests(unittest.TestCase):
    def setUp(self):
        self.sources = {"s1": {"title": "Energy source", "content": "Energy scores states."},
                        "s2": {"title": "Editing source", "content": "Critiques suggest edits."},
                        "noise": {"title": "Unselected source", "content": "Other topic."}}
        self.masks = {"a1": {"s1": 1, "s2": 0, "noise": 0}, "a2": {"s1": 0, "s2": 1, "noise": 0}}
        self.problem = loop.FixedProblem(self.sources, loop.CONCEPTS, self.masks)

    def test_conditions_are_immutable_and_requests_get_copies(self):
        before = self.problem.fingerprint
        self.sources["s1"]["content"] = "changed input"
        self.masks["a1"]["s1"] = 0
        payload = self.problem.payload()
        payload["W"]["w1"] = "new concept"
        payload["A"]["a1"]["s1"] = 0
        payload["D"]["s1"]["content"] = "new source"
        self.assertEqual(loop.digest(self.problem.payload()), before)
        with self.assertRaises(AttributeError):
            self.problem._json = "changed"

    def test_document_schema_rejects_concept_or_mask_updates(self):
        for key in ["W", "A", "concepts", "membership", "sources", "objective"]:
            document = {**loop.SEED, key: {"changed": True}}
            with self.assertRaises(ValueError):
                loop.validate_document(document, self.problem)

    def test_citations_cannot_expand_fixed_selection(self):
        for title in ["Unknown source", "Unselected source"]:
            with self.assertRaises(ValueError):
                loop.validate_document({"title": "Draft", "content": f"Claim [[{title}]]"}, self.problem)
        loop.validate_document({"title": "Draft", "content": "[[Energy source]] and [[Editing source]]"}, self.problem)

    def test_invalid_scores_do_not_enter_energy(self):
        for bad in [float("nan"), float("inf"), True, -1, 4]:
            ev = evaluation()
            ev["concept_deficit"] = bad
            with self.assertRaises(ValueError):
                loop.validate_evaluation(ev)
        ev = evaluation()
        ev["evidence"] = {}
        with self.assertRaises(ValueError):
            loop.validate_evaluation(ev)

    def test_composition_sums_separately_evaluated_concept_energies(self):
        energies = loop.energy({"w1": evaluation(1), "w2": evaluation(0.5)})
        self.assertEqual(energies["E1"], 4)
        self.assertEqual(energies["E2"], 2)
        self.assertAlmostEqual(energies["total"], 6)
        self.assertAlmostEqual(energies["E1"] + energies["E2"], energies["total"])

    def test_live_loop_changes_only_document_and_retains_score_increases(self):
        first = {"title": "Bridge", "content": "Scoring differs from editing. [[Energy source]]"}
        second = {"title": "Revised bridge", "content": "Link scoring and editing. [[Energy source]] [[Editing source]]"}
        client = FakeClient([evaluation(2), evaluation(2), first, evaluation(0.5), evaluation(0.5), second, evaluation(1), evaluation(1)])
        with tempfile.TemporaryDirectory() as directory:
            records = loop.run(client, self.problem, 2, Path(directory))
            self.assertEqual([r["iteration"] for r in records], [0, 1, 2])
            self.assertEqual([r["energy"]["total"] for r in records], [16, 4, 8])
            self.assertTrue(records[-1]["accepted"])
            self.assertEqual(records[-1]["document"], second)
            self.assertEqual(len(client.requests), 8)
            self.assertTrue((Path(directory)/"REPORT.md").exists())
            for purpose, _, payload in client.requests:
                self.assertEqual(set(payload["x_new"]), {"title", "content"})
                if purpose.startswith("evaluate"):
                    self.assertEqual(set(payload), {"fixed_conditions", "x_new"})
                    i = int(purpose[-1])
                    self.assertEqual(payload["fixed_conditions"], self.problem.condition(i))
                    self.assertEqual(set(payload["fixed_conditions"]), {"D", "w", "a"})
                else:
                    self.assertEqual(loop.digest(payload["fixed_conditions"]), self.problem.fingerprint)
            critic_prompts = [system for purpose,system,_ in client.requests if purpose.startswith("evaluate")]
            self.assertEqual(len(set(critic_prompts)), 1)
            checkpoint = json.loads((Path(directory)/"trajectory.json").read_text())
            self.assertEqual({r["fixed_conditions_sha256"] for r in checkpoint}, {self.problem.fingerprint})
            # Resume consumes no further calls for already completed revisions.
            replay = FakeClient([])
            self.assertEqual(loop.run(replay, self.problem, 2, Path(directory)), records)
            self.assertEqual(replay.requests, [])

    def test_forbidden_update_consumes_slot_without_editing_conditions(self):
        bad = {**loop.SEED, "membership": {"s1": 0}}
        valid = {"title": "Bridge", "content": "Revised explanation."}
        client = FakeClient([evaluation(), evaluation(), bad, valid, evaluation(0.5), evaluation(0.5)])
        with tempfile.TemporaryDirectory() as directory:
            records = loop.run(client, self.problem, 2, Path(directory))
        self.assertFalse(records[1]["accepted"])
        self.assertIsNone(records[1]["candidate_energy"])
        self.assertEqual(records[1]["document"], loop.SEED)
        self.assertEqual(client.requests[3][2]["x_new"], loop.SEED)
        self.assertEqual([r[0] for r in client.requests], ["evaluate seed/w1", "evaluate seed/w2", "edit 1", "edit 2", "evaluate candidate 2/w1", "evaluate candidate 2/w2"])

    def test_manifest_blocks_changed_conditions_and_legacy_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/"run"
            args = (out, self.problem, {}, "test-model", 2, 100, 1000)
            manifest = loop.prepare(*args)
            self.assertEqual(manifest["optimized_variable"], "x_new")
            self.assertEqual(manifest["max_calls"], 8)
            self.assertEqual(loop.prepare(*args, resume=True), manifest)
            manifest["protocol"] = "legacy-joint-state"
            loop.dump(out/"manifest.json", manifest)
            with self.assertRaises(ValueError):
                loop.prepare(*args, resume=True)

    def test_corpus_masks_match_exp4_and_source_files_remain_unchanged(self):
        problem, files = loop.load_problem()
        p = problem.payload()
        self.assertEqual(len(p["D"]), 18)
        self.assertEqual({k for k,v in p["A"]["a1"].items() if v}, set(loop.CLUSTER1))
        self.assertEqual({k for k,v in p["A"]["a2"].items() if v}, set(loop.CLUSTER2))
        _, after = loop.load_problem()
        self.assertEqual(files, after)


if __name__ == "__main__":
    unittest.main()
