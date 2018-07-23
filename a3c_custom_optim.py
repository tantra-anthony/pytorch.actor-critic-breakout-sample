# create custom optimizer for LSTM as normal optimizer is not enough

# import relevant modules
import math
import torch
import torch.optim as optim

# inherit Adam optim from torch library
class SharedAdam(optim.Adam):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        # import all the basic params in Adam class
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        # param_groups contains all the params passed into the __init__ that we
        # have to optimize
        for group in self.param_groups:
            # scoopt the params in param_groups
            for p in group['params']:
                # the update made by adam optimizer is based on exponential moving
                # average of tha gradient of order 1, also based on exponential moving
                # average of the gradient of order 2
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                # to accelerate the computations similar to tensor.cuda
                # sent to the GPU for parallelization
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self): # use super(SharedAdam, self).step
        # refer to paper for algorithm in adam
        loss = None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step'][0]
                bias_correction2 = 1 - beta2 ** state['step'][0]
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                p.data.addcdiv_(-step_size, exp_avg, denom)
        return loss

