from .inverse_square_root import InverseSquareRoot
from .step_lr_with_warmup import MultiStepWithWarmup

__all__ = ['InverseSquareRoot', 'MultiStepWithWarmup', 'schedule_builder']


def schedule_builder(optimizer, cfg):
    if cfg['TYPE'] == 'MultiStepWithWarmup':
        return MultiStepWithWarmup(optimizer,
                                   warmup_updates = cfg['WARMUP_UPDATES'],
                                   warmup_end_lr = cfg['WARMUP_END_LR'],
                                   milestones = cfg['MILESTONES'],
                                   gamma = cfg['GAMMA'])
    elif  cfg['TYPE'] == 'InverseSquareRoot':
        return InverseSquareRoot(optimizer, 
                                 warmup_updates = cfg['WARMUP_UPDATES'],
                                 warmup_end_lr = cfg['WARMUP_END_LR'])
    else:
        raise ValueError(f"Scheduler {cfg['TYPE']} is not implemented")
    
    
    
    
