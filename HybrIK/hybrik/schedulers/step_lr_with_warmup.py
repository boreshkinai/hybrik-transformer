from torch.optim.lr_scheduler import _LRScheduler
from bisect import bisect_right
from collections import Counter


class StepLrWithWarmup(_LRScheduler):
    """
    Decay the LR based on the step schedule with warmup
    """

    def __init__(self, optimizer, warmup_updates: int = 50, warmup_init_lr: float = -1,
                 warmup_end_lr: float = 0.0005, gamma: float = 0.5, step_size: int = 200,
                 last_epoch: int = -1, verbose: bool = False):
        
        if warmup_init_lr < 0:
            self.warmup_init_lr = 0 if warmup_updates > 0 else warmup_end_lr
        else:
            self.warmup_init_lr = warmup_init_lr
        self.warmup_end_lr = warmup_end_lr
        
        self.warmup_updates = warmup_updates
        # linearly warmup for the first cfg.warmup_updates
        self.lr_step = (self.warmup_end_lr - self.warmup_init_lr) / self.warmup_updates
        # then, apply decay rate
        self.gamma = gamma
        # initial learning rate
        self.lr = warmup_init_lr   
        self.step_size = step_size
        
        super(StepLrWithWarmup, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """Update the learning rate after each update."""
        if self.last_epoch < self.warmup_updates:
            self.lr = self.warmup_init_lr + self.last_epoch * self.lr_step
        else:
            self.lr = self.warmup_end_lr * self.gamma ** ((self.last_epoch - self.warmup_updates) // self.step_size)
            
        return [self.lr]
    
    
class MultiStepWithWarmup(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma once the
    number of epoch reaches one of the milestones. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside
    this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, milestones, warmup_updates: int = 50, warmup_init_lr: float = -1,
                 warmup_end_lr: float = 0.0005, gamma=0.1, last_epoch=-1, verbose=False):
        self.milestones = Counter(milestones)
        if warmup_init_lr < 0:
            self.warmup_init_lr = 0 if warmup_updates > 0 else warmup_end_lr
        else:
            self.warmup_init_lr = warmup_init_lr
        self.warmup_end_lr = warmup_end_lr
        
        self.warmup_updates = warmup_updates
        # linearly warmup for the first cfg.warmup_updates
        self.lr_step = (self.warmup_end_lr - self.warmup_init_lr) / self.warmup_updates
        # then, apply decay rate
        self.gamma = gamma
        # initial learning rate
        self.lr = warmup_init_lr   
        super(MultiStepWithWarmup, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
            
        if self.last_epoch < self.warmup_updates:
            self.lr = self.warmup_init_lr + self.last_epoch * self.lr_step
        else:
            milestones = list(sorted(self.milestones.elements()))
            self.lr = self.warmup_end_lr * self.gamma ** bisect_right(milestones, self.last_epoch)
            
        return [self.lr]

    