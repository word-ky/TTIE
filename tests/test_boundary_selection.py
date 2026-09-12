import copy
import tempfile
from pathlib import Path
import unittest
import torch
from ttie.energy_model import EnergyHead, features
from ttie.semantic_ttt import FixedObjective
from ttie.soft_basis.renderer import FixedCorners
from ttie.soft_basis.selection import BOUNDARIES, HARD_INDICES, choose, score_episode
from ttie.soft_basis.score import score_inputs, write, tensor_sha
from ttie.stop_receipt import sha
from ttie.soft_basis.evaluate import evaluate, spearman
from test_semantic_ttt import scorer, RECEIPT
from test_projected_ttt import pixels


class SelectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.image = pixels(); self.scorer = scorer()
        self.head = EnergyHead().eval().requires_grad_(False)
        self.objective = FixedObjective(self.scorer, self.image, RECEIPT)
        self.gate = {k: getattr(self.objective, k).tolist() for k in ('active', 'winner', 'evidence')}
        self.corners = torch.tensor([[[[.3, -.2], [0., .4]], [[.9, 1.1], [1., .8]]]])

    def run_score(self):
        return score_episode(self.image, self.corners, self.gate,
                             RECEIPT['calibration'], self.scorer, self.head)

    def test_exact_order_ties_and_frozen_feature_equivalence(self):
        self.assertEqual(HARD_INDICES, tuple(range(0, 27, 3)))
        self.assertEqual(BOUNDARIES[4], (.5, .5, 0.))
        self.assertEqual(choose([2.]*9), dict(selected_index=0, margin=0., minimum_ties=9))
        original = self.corners.clone(); gate = copy.deepcopy(self.gate)
        result = self.run_score()
        for i, candidate in enumerate(BOUNDARIES):
            current = self.scorer(FixedCorners(self.corners, candidate)(self.image))
            expected = features(self.objective, current, self.corners)
            self.assertTrue(torch.equal(torch.tensor(result['features'][i]), expected))
            self.assertEqual(result['energies'][i], float(self.head(expected)))
        self.assertTrue(torch.equal(original, self.corners)); self.assertEqual(gate, self.gate)
        self.assertTrue(all(p.grad is None for p in self.head.parameters()))

    def test_reference_metadata_cannot_enter_scoring(self):
        before = self.run_score()
        for key in ('clean', 'condition', 'image_id', 'reference_mse', 'mask', 'gain'):
            with self.assertRaises(TypeError):
                score_episode(self.image, self.corners, self.gate, RECEIPT['calibration'],
                              self.scorer, self.head, **{key: torch.rand(3)})
        self.assertEqual(before, self.run_score())

    def test_replacing_reference_files_cannot_change_scores_or_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); inputs = root/'label_free'; inputs.mkdir()
            torch.save(dict(image=self.image, corners=self.corners, gate=self.gate), inputs/'001.pt')
            write(inputs/'inputs.json', dict(episodes=[dict(episode='episodes/001', file='001.pt', sha256=sha(inputs/'001.pt'))]))
            reference = root/'candidate_metrics.json'
            write(reference, dict(clean='old', condition='left_right', image_id=1, mse=[0]*27))
            before = score_inputs(inputs, self.scorer, self.head, RECEIPT['calibration'], 'cpu')
            write(reference, dict(clean='changed', condition='quadrants', image_id=999, mse=[100]*27))
            after = score_inputs(inputs, self.scorer, self.head, RECEIPT['calibration'], 'cpu')
            self.assertEqual(before, after)

    def test_evaluation_clauses_rank_ties_and_selection_immutability(self):
        selections, references = [], []
        for i, condition in enumerate(('left_right', 'quadrants', 'offset_left_right_40')):
            mse = [1.]*27; mse[0] = .9
            selections.append(dict(episode=str(i), selected_index=0, corners_sha256='same',
                                   energies=[0., 1., 2., 3., 4., 5., 6., 7., 8.], margin=1., minimum_ties=1))
            references.append(dict(source_directory=str(i), image_id=i, condition=condition,
                corners_sha256='same', candidate_mse=mse, region2_mse=1., t015_oracle_mse=.99))
        selection = dict(episodes=selections); before = copy.deepcopy(selection)
        report, rows = evaluate(selection, references)
        self.assertEqual(report['passed'], 5); self.assertEqual(report['verdict'], 'label_free_boundary_selection_evidence')
        self.assertEqual(report['groups']['spatial_pool']['ratios']['selected_over_hard_oracle'], 1.)
        self.assertEqual(selection, before)
        for r in references: r['candidate_mse'][0] = 2.
        report, _ = evaluate(selection, references)
        self.assertEqual(report['passed'], 0); self.assertEqual(report['verdict'], 'negative')
        for r in references:
            r.update(candidate_mse=[0.]*27, region2_mse=0., t015_oracle_mse=0.)
        report, _ = evaluate(selection, references)
        self.assertIsNone(report['groups']['spatial_pool']['ratios']['selected_over_region2'])
        self.assertEqual(report['groups']['spatial_pool']['oracle_minimum_tied_episodes'], 3)
        self.assertEqual(spearman([1, 2, 2], [4, 3, 3]), -1.)
        self.assertIsNone(spearman([1]*9, list(range(9))))


if __name__ == '__main__': unittest.main()
