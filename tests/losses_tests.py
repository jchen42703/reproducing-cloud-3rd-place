import unittest
import torch
import numpy as np
from clouds.losses import ComboLoss, ComboLossOnlyPos


class LossesTests(unittest.TestCase):
    """Testing losses.

    This mainly consists of exploration tests to verify assumptions of what
    goes on under the hood.
    """
    def setUp(self):
        """Initializing the parameters:
        """
        self.batch_size = 16
        self.channels = 4
        self.out_size = (self.batch_size, self.channels, 384, 576)
        self.targets = torch.from_numpy(np.random.choice([0, 1],
                                                         size=self.out_size))
        self.logits = torch.from_numpy(np.random.uniform(0, 1,
                                                         size=self.out_size))

    def test_fc_tensor_shapes(self):
        """Tests logit/target assumptions for the combo losses.
        """
        # TARGETS
        # shape: (batch_size, channel)
        summed = self.targets.view(self.batch_size, self.channels, -1).sum(-1)
        self.assertEqual(summed.shape, (self.batch_size, self.channels))
        # gathers all of the slices that actually have an ROI
        targets_fc = (summed > 0).float()
        self.assertEqual(targets_fc.shape, summed.shape)

        # LOGITS
        logits_fc = self.logits.view(self.batch_size, self.channels, -1)
        # gathers all of the slices that have been predicted to have an ROI
        logits_fc = torch.max(logits_fc, -1)[0]
        self.assertEqual(logits_fc.shape, summed.shape)

    def test_ComboLoss(self):
        """Tests that ComboLoss runs.
        """
        combo_loss = ComboLoss(weights=[0, 0, 1], activation=None)
        self.assertTrue(isinstance(combo_loss, torch.nn.Module))
        loss = combo_loss(self.logits, self.targets)
        self.assertTrue(isinstance(loss, torch.Tensor))

    def test_ComboLossOnlyPos(self):
        """Tests that ComboLossOnlyPos runs.
        """
        combo_loss = ComboLossOnlyPos(weights=[0, 1, 0], activation=None)
        self.assertTrue(isinstance(combo_loss, torch.nn.Module))
        loss = combo_loss(self.logits, self.targets)
        self.assertTrue(isinstance(loss, torch.Tensor))


unittest.main(argv=[''], verbosity=2, exit=False)
