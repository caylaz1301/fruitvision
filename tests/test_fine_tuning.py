import unittest

from src.train import fine_tuning_settings


class FineTuningTests(unittest.TestCase):
    def test_baseline_optimizer_settings_are_preserved(self):
        self.assertEqual(fine_tuning_settings(None, 50), {})

    def test_fine_tuning_does_not_inherit_high_bias_warmup(self):
        settings = fine_tuning_settings(.0001, 6)
        self.assertEqual(settings['optimizer'], 'AdamW')
        self.assertEqual(settings['lr0'], .0001)
        self.assertEqual(settings['warmup_bias_lr'], 0)
        self.assertEqual(settings['close_mosaic'], 6)

    def test_invalid_rates_and_epochs(self):
        for rate in (0, -1, float('nan'), float('inf'), 2):
            with self.assertRaises(ValueError):
                fine_tuning_settings(rate, 6)
        with self.assertRaises(ValueError):
            fine_tuning_settings(.0001, 0)
