import torch
import math
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
            iters = math.ceil(trainSampleSize / batchSize) * epochs
        lr (float): learning rate 
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        coeff(float, optional): scaling coefficient
       
    """

    def __init__(self, params, lr=required, iters=required, momentum=0.9, 
                 dampening=0, weight_decay=0, nesterov=False, coeff=1e-2):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 1 <= iters:
            raise ValueError("Invalid iters: {}".format(iters))             
        if momentum <= 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= coeff <= 1.0:
            raise ValueError("Invalid coeff: {}".format(coeff))                    
            
        defaults = dict(lr=lr, iters=iters, momentum=momentum, dampening=dampening, weight_decay=weight_decay,
                        nesterov=nesterov, coeff=coeff)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
            
        super(TSGD, self).__init__(params, defaults)

        
    def __setstate__(self, state):
        super(TSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, epoch, closure=None):
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
            coeff = group['coeff']
            iters = group['iters']
            rho = 10 ** (math.log(coeff, 10) / iters)

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
                d_p = (buf - d_p) * (rho ** param_state['step']) + d_p   
                
                p.data.add_(d_p, alpha=-group['lr'])
                
        return loss
