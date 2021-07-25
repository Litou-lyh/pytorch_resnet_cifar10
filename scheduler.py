import torch
import torch.optim.lr_scheduler as lr_scheduler
class MultiStageScheduler():
    def __init__(self,optimizer,args=None) -> None:
        self.optimizer = optimizer
        self.policy = args.policy if args else None
        self.schedulers = []
        self.steps = 0
        self.stage = 0
        self.milestones = args.milestones if args and args.policy[0] != 'step' else None
        self.decay = 0.1
            
        for i, p in enumerate(self.policy):
            if p == 'cyclic':
                self.schedulers.append(lr_scheduler.CyclicLR(self.optimizer, base_lr=args.lr * pow(self.decay,i), max_lr=args.max_lr * pow(self.decay,i),step_size_up=5000 * pow(self.decay * 5,i)))
            elif p == 'step':
                self.schedulers.append(lr_scheduler.MultiStepLR(self.optimizer, milestones=args.milestones))
            elif p == 'rop':
                self.schedulers.append(lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max'))
            elif p == 'sgdr':
                self.schedulers.append(lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,T_0=10, T_mult=2))
            elif p == 'constant':
                self.schedulers.append(None)
            elif p == 'cos':
                self.schedulers.append(lr_scheduler.CosineAnnealingLR(self.optimizer, args.epochs))
        self.current_scheduler = self.schedulers[0] if self.schedulers else None
        
    def check_switch(self):
        if self.stage >= len(self.milestones):
            return
        if self.steps > self.milestones[self.stage]:
            self.stage += 1
            if self.stage >= len(self.schedulers):
                raise Exception('error')
            self.current_scheduler = self.schedulers[self.stage]
    
    def step(self, arg=None):
        if self.milestones:
            self.check_switch()
        if not self.current_scheduler: return
        if arg:
            self.current_scheduler.step(arg)
        else:
            self.current_scheduler.step()
        self.steps += 1
