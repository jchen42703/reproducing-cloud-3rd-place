from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, _LRScheduler


class WarmRestartsCustomScheduler(_LRScheduler):
    """Custom Learning Rate Scheduler based on the 3rd Place Solution.

    This is for setting the learning rate schedule:
    Warm Restarts for epochs (1-28)
    LR=1e-5 (29-32), LR=1e-6 (33-35)

    The general version looks like this:
    # from:
    # https://github.com/naivelamb/kaggle-cloud-organization/blob/master/main_seg.py
    if epoch < start_epoch + n_epochs - 1:
        if epoch != 0:
            scheduler.step()
            scheduler=warm_restart(scheduler, T_mult=2)
    elif (epoch < start_epoch + n_epochs + 2 and
          epoch >= start_epoch + n_epochs - 1):
        optimizer.param_groups[0]['lr'] = 1e-5
    else:
        optimizer.param_groups[0]['lr'] = 5e-6

    """
    def __init__(self, optimizer, T_0, T_mult=2, eta_min=0, num_wr_epochs=28,
                 mid_const_lr_epochs_range=[29, 32], constant_lrs=[1e-5, 5e-6],
                 last_epoch=-1):
        """
        Args:
            optimizer (torch.optim.Optimizer):
            T_0:
            T_mult:
            eta_min:
            num_wr_epochs (int): The number of warm restart epochs to do
            mid_const_lr_epochs_range (list-like[int]): [min, max] where max
                is not included. This is the epoch interval where the first
                lr of constant_lr is used
            constant_lrs (list-like[float]): the learning rates to use for the
                mid and end intervals after warm restarts ends.
        """
        self.num_wr_epochs = num_wr_epochs
        assert len(mid_const_lr_epochs_range) == 2, \
            "`constant_lrs` must be a list-like with length 2."
        self.mid_const_lr_epochs_range = mid_const_lr_epochs_range
        assert len(constant_lrs) == 2, \
            "`constant_lrs` must be a list-like with length 2."
        self.constant_lrs = constant_lrs

        self.optimizer = optimizer
        self.warm_restarts = CosineAnnealingWarmRestarts(self.optimizer, T_0,
                                                         T_mult, eta_min)
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        """No calculation done here.
        """
        return self.get_last_lr()

    def step(self, epoch=None):
        """Computes a step for the learning rate scheduler.

        Here, a step is an epoch. This is where the learning rates are set
        and the last_epoch counter is updated.
        """
        # warm restarts
        if self.last_epoch < self.num_wr_epochs + 1:
            self.warm_restarts.step()
            self.last_epoch = self.warm_restarts.last_epoch
            self._last_lr = self.warm_restarts.get_last_lr()
        # constant LR (first round)
        elif (self.last_epoch >= self.mid_const_lr_epochs_range[0] and
              self.last_epoch < self.mid_const_lr_epochs_range[1]):
            self.last_epoch += 1
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.constant_lrs[0]
            self._last_lr = [group['lr']
                             for group in self.optimizer.param_groups]
        # constant LR (second round)
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.constant_lrs[1]
            self.last_epoch += 1
            self._last_lr = [group['lr']
                             for group in self.optimizer.param_groups]
