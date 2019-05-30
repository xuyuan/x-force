from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR


class FindLR(_LRScheduler):
    """
    inspired by fast.ai @https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    """
    def __init__(self, optimizer, max_steps, max_lr=10):
        self.max_steps = max_steps
        self.max_lr = max_lr
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * ((self.max_lr / base_lr) ** (self.last_epoch / (self.max_steps - 1)))
                for base_lr in self.base_lrs]


class NoamLR(_LRScheduler):
    """
    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first ``warmup_steps`` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.
    Parameters
    ----------
    warmup_steps: ``int``, required.
        The number of steps to linearly increase the learning rate.
    """
    def __init__(self, optimizer, warmup_steps):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.warmup_steps ** 0.5 * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]


class WarmUpLR(_LRScheduler):
    def __init__(self, lr_scheduler, warmup_steps, eta_min=0):
        self.lr_scheduler = lr_scheduler
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min
        super().__init__(lr_scheduler.optimizer, lr_scheduler.last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [self.eta_min + (base_lr - self.eta_min) * (self.last_epoch / self.warmup_steps)
                    for base_lr in self.base_lrs]
        return self.lr_scheduler.get_lr()

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        if epoch < self.warmup_steps:
            super().step(epoch)
        else:
            self.last_epoch = epoch
            self.lr_scheduler.step(epoch - self.warmup_steps)

    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'lr_scheduler')}
        state_dict['lr_scheduler'] = self.lr_scheduler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        lr_scheduler = state_dict.pop('lr_scheduler')
        self.__dict__.update(state_dict)
        self.lr_scheduler.load_state_dict(lr_scheduler)