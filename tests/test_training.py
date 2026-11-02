import contextlib
import importlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.model.dataset import SentimentDataset, split_reviews
from app.model import model as model_module
from app.model import train as training


torch.set_num_threads(1)


def reviews_fixture():
    return pd.DataFrame({
        'comment_cleaned': [f'review {i}' for i in range(120)],
        'rate': [1] * 100 + [5] * 20,
        'rate_filtered': [0] * 100 + [1] * 20,
    })


class SplitTests(unittest.TestCase):
    def test_duplicates_never_cross_splits_and_only_training_is_balanced(self):
        reviews = reviews_fixture()
        duplicated = pd.concat([reviews, reviews.iloc[:10]], ignore_index=True)
        train, valid, test = split_reviews(duplicated, valid_size=20, test_size=20)
        sets = [set(part['comment_cleaned']) for part in (train, valid, test)]
        self.assertFalse(sets[0] & sets[1])
        self.assertFalse(sets[0] & sets[2])
        self.assertFalse(sets[1] & sets[2])
        self.assertEqual(set.union(*sets), set(reviews['comment_cleaned']))
        self.assertEqual(len(valid), 20)
        self.assertEqual(len(test), 20)
        self.assertTrue(valid['comment_cleaned'].is_unique)
        self.assertTrue(test['comment_cleaned'].is_unique)
        self.assertEqual(train['rate_filtered'].value_counts().nunique(), 1)
        self.assertGreater(len(train), len(sets[0]))
        # Holdouts retain the imbalanced population rather than being resampled.
        self.assertGreater((valid['rate_filtered'] == 0).sum(), 10)
        self.assertGreater((test['rate_filtered'] == 0).sum(), 10)
        self.assertEqual(len(duplicated), 130)

    def test_fixed_seed_reproduces_the_partitions(self):
        first = split_reviews(reviews_fixture(), valid_size=20, test_size=20)
        second = split_reviews(reviews_fixture(), valid_size=20, test_size=20)
        for left, right in zip(first, second):
            pd.testing.assert_frame_equal(left, right)
        changed = split_reviews(
            reviews_fixture(), valid_size=20, test_size=20, random_state=7,
        )
        self.assertNotEqual(
            set(first[2]['comment_cleaned']), set(changed[2]['comment_cleaned']),
        )

    def test_prepared_dataset_does_not_reload_the_csv(self):
        with patch('pandas.read_csv', side_effect=AssertionError('unexpected I/O')):
            dataset = SentimentDataset(reviews=reviews_fixture())
            self.assertEqual(dataset[0], ('review 0', 0))
            self.assertEqual(len(dataset), 120)

    def test_model_import_does_not_load_training_data_or_a_tokenizer(self):
        with patch('pandas.read_csv', side_effect=AssertionError('unexpected I/O')):
            reloaded = importlib.reload(model_module)
        self.assertEqual(reloaded.get_tokenizer.cache_info().currsize, 0)


class EvaluationTests(unittest.TestCase):
    def test_metrics_weight_examples_and_accept_a_final_singleton_batch(self):
        probabilities = torch.tensor([0.9, 0.6, 0.8, 0.2, 0.1])
        labels = torch.tensor([1, 0, 1, 0, 1])
        loader = DataLoader(
            [(p.reshape(1), label, torch.tensor(1))
             for p, label in zip(probabilities, labels)],
            batch_size=2,
        )

        class FixedModel(nn.Module):
            def forward(self, text, lengths):
                return text

        accuracy, loss = training.evaluate(FixedModel(), loader, nn.BCELoss())
        self.assertAlmostEqual(accuracy, 3 / 5)
        expected = nn.BCELoss()(probabilities, labels.float()).item()
        self.assertAlmostEqual(loss, expected, places=6)

    def test_two_epoch_run_validates_before_scheduler_and_loads_only_new_checkpoint(self):
        partitions = split_reviews(reviews_fixture(), valid_size=20, test_size=20)
        events = []
        real_evaluate = training.evaluate
        real_scheduler = training.ReduceLROnPlateau
        real_load = torch.load

        class Tokenizer:
            def encode(self, text):
                return [1, 2, int(text.rsplit(' ', 1)[-1]) % 20 + 3]

        def evaluate(*args):
            result = real_evaluate(*args)
            events.append(('validation_or_test', result[1]))
            return result

        def scheduler(*args, **kwargs):
            instance = real_scheduler(*args, **kwargs)
            real_step = instance.step

            def step(loss):
                events.append(('scheduler', loss))
                real_step(loss)

            instance.step = step
            return instance

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'new-checkpoint.pth'
            with (
                patch.object(training, 'load_reviews', return_value=reviews_fixture()),
                patch.object(training, 'split_reviews', return_value=partitions),
                patch.object(training, 'RNN',
                             side_effect=lambda *args: model_module.RNN(32, 8, 4, 8)),
                patch.object(model_module, 'get_tokenizer', return_value=Tokenizer()),
                patch.object(training, 'evaluate', side_effect=evaluate),
                patch.object(training, 'ReduceLROnPlateau', side_effect=scheduler),
                patch.object(training.torch, 'load', wraps=real_load) as checkpoint_load,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                training.main(['--epochs', '2', '--batch-size', '4',
                               '--output', str(output)])
            self.assertTrue(output.is_file())
            self.assertEqual(checkpoint_load.call_count, 1)
            self.assertEqual(checkpoint_load.call_args.args[0], output)

        self.assertEqual([event[0] for event in events], [
            'validation_or_test', 'scheduler',
            'validation_or_test', 'scheduler', 'validation_or_test',
        ])
        self.assertEqual(events[0][1], events[1][1])
        self.assertEqual(events[2][1], events[3][1])


if __name__ == '__main__':
    unittest.main()

