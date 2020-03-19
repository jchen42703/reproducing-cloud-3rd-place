import unittest
import torch
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

from clouds.custom_lr_schedulers import WarmRestartsCustomScheduler


class LRSchedTests(unittest.TestCase):
    """Testing learning rate scheduelrs.

    This mainly consists of exploration tests to verify assumptions of what
    goes on under the hood.
    """
    def setUp(self):
        """Initializing the parameters:
        """
        pass

    @unittest.skip("Exploration test. Non-essential.")
    def test_last_epoch(self):
        """Tests the assumption that last_epoch is the internal epoch counter.

        Should just be range(start_epoch, last_epoch)
        """
        start_epoch = 0
        num_epochs = 20
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        lr_scheduler_1 = lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max=num_epochs,
                                                        eta_min=0.1)
        steps = []
        for i in range(start_epoch, num_epochs):
            steps.append(lr_scheduler_1.last_epoch)
            lr_scheduler_1.step()
        self.assertEqual(steps, list(range(start_epoch, num_epochs)))

        lr_scheduler_2 = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                  T_0=10,
                                                                  T_mult=2,
                                                                  eta_min=1e-6)
        steps = []
        for i in range(start_epoch, num_epochs):
            steps.append(lr_scheduler_2.last_epoch)
            lr_scheduler_2.step()
        self.assertEqual(steps, list(range(start_epoch, num_epochs)))

    def test_WarmRestartsCustomScheduler(self):
        """Tests that WarmRestartsCustomScheduler creates the right schedule.

        This is a visual check. To resume other tests, close the plot that pops
        up.
        """
        start_epoch = 0
        num_epochs = 35
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

        # Using the default paramters:
        # num_wr_epochs=28, mid_const_lr_epochs_range=[29, 32],
        # constant_lrs=[1e-5, 5e-6]
        lr_scheduler_1 = WarmRestartsCustomScheduler(optimizer,
                                                     T_0=10, T_mult=2,
                                                     eta_min=1e-6)
        lrs, steps = [], []
        for i in range(start_epoch, num_epochs):
            steps.append(lr_scheduler_1.last_epoch)
            lr_scheduler_1.step()
            lrs.append(
                optimizer.param_groups[0]["lr"]
            )
        # Quick checks
        self.assertEqual(steps, list(range(start_epoch+1, num_epochs+1)))
        # 29th epoch and onwards
        self.assertEqual(lrs[28:],
                         [1e-5, 1e-5, 1e-5, 5e-6, 5e-6, 5e-6, 5e-6])
        # Visual Check
        plt.plot(lrs)
        plt.show()


unittest.main(argv=[''], verbosity=2, exit=False)
