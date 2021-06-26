import torch
from torch.optim.optimizer import Optimizer, required


class TSGD(Optimizer):
    r"""Implements: Scaling transition from SGDM to plain SGD.
    'https://arxiv.org/abs/2106.06753'
    
    base on: https://github.com/pytorch/pytorch/blob/v1.4.0/torch/optim/sgd.py

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        iters(int, required): iterations
            iters = (testSampleSize / batchSize) * epoch      
        lr (float): learning rate 
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        moment(float, optional): transition moment, moment = transition_iters / iters
        up_lr(float, optional): upper learning rate
        low_lr(float, optional): lower learning rate         
        
    """

    def __init__(self, params, lr=required, iters=required, momentum=0.9, 
                 dampening=0, weight_decay=0, nesterov=False, moment=3/8, up_lr=0.1, low_lr=0.005):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 1.0 <= iters:
            raise ValueError("Invalid iters: {}".format(iters))             
        if momentum <= 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 < moment < 1:
            raise ValueError("Invalid gamma: {}".format(moment))                    
        if not 0.0 <= up_lr:
            raise ValueError("Invalid up learning rate: {}".format(up_lr))
        if not 0.0 <= low_lr:
            raise ValueError("Invalid low learning rate: {}".format(low_lr))            
        if not low_lr <= up_lr:
            raise ValueError("required up_lr  >= low_lr, but (up_lr = {}, low_lr = {})".format(up_lr, low_lr))  
            
        defaults = dict(lr=lr, iters=iters, momentum=momentum, dampening=dampening, weight_decay=weight_decay,
                        nesterov=nesterov, moment=moment, up_lr=up_lr, low_lr=low_lr)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
            
        super(TSGD, self).__init__(params, defaults)

        
    def __setstate__(self, state):
        super(TSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self,closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:

        
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            iters = group['iters']
            moment  = group['moment']
            gamma = 10 ** (-2 / (iters * moment))

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data,  alpha=weight_decay)
                param_state = self.state[p]

                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                if 'step' not in param_state:
                    param_state['step'] = 0
                else:
                    param_state['step'] += 1
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    
                #Scaling the momentum
                d_p = (buf - d_p) * (gamma ** param_state['step']) + d_p
                #Decreasing the learning rate
                dlr = (group['up_lr'] - group['low_lr']) * (1 - 1/iters * param_state['step']) + group['low_lr']    
                
                p.data.add_(d_p, alpha=-dlr)
                
        return loss
