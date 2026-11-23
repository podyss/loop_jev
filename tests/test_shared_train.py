import json
import random
import unittest

import torch

from loop_jev.data.shared_train import digest, prepare_sample, shared_rows, validate_teacher
from loop_jev.evaluation.block_shortcut import choose_train_rows
from loop_jev.evaluation.option_loop_residual import shared_mix, train_adapter


class SharedTrainTests(unittest.TestCase):
    def test_prepare_matches_original_teacher_sampling(self):
        pool = [{'id': i, 'split': 'train'} for i in range(1300)]
        source = b''.join(json.dumps(row).encode() + b'\n' for row in pool)
        sample = prepare_sample(source)
        rows = [json.loads(line) for line in sample.splitlines()]
        self.assertEqual(rows, [pool[i] for i in random.Random(42).sample(range(len(pool)), 1024)])
        self.assertIs(choose_train_rows(rows, 1024, 'fixed'), rows)
        with self.assertRaises(ValueError):
            shared_rows(rows[:-1] + rows[:1])

        metadata = dict(status='complete', examples=1024)
        validate_teacher(dict(metadata, train_sha256=digest(sample)), source, sample)
        legacy = dict(metadata, train_sha256=digest(source), train_sample_seed=42, max_train_examples=1024)
        validate_teacher(legacy, source, sample)
        with self.assertRaises(ValueError):
            validate_teacher(dict(legacy, train_sample_seed=43), source, sample)

    def test_calibration_checks_train_hash_and_model(self):
        report = dict(model='teacher', train_sha256='hash', results={'label-compact-noslots': {'mix': 1.5}})
        self.assertEqual(shared_mix(report, 'teacher', 'hash'), 1.5)
        for model, fingerprint in (('teacher2', 'hash'), ('teacher', 'other')):
            with self.subTest(model=model, fingerprint=fingerprint), self.assertRaises(ValueError):
                shared_mix(report, model, fingerprint)

    def test_all_rows_fit_without_validation(self):
        scores = torch.zeros(4, 4)
        features = torch.eye(4).expand(4, -1, -1)
        gold = torch.zeros(4, dtype=torch.long)
        head, epoch = train_adapter(scores, features, None, gold, None, distill=False)
        self.assertEqual(epoch, 12)
        self.assertTrue((head(scores, features).argmax(-1) == gold).all())
