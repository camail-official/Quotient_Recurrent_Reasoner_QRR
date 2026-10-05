import copy
import torch
import torch.nn as nn

class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        # foreach batching: one fused launch set instead of ~100 per-tensor kernels/step.
        # shadow = (1-mu)*param + mu*shadow, evaluated as two products then a sum.
        pairs = [(n, p) for n, p in module.named_parameters() if p.requires_grad]
        t = torch._foreach_mul([p.data for _, p in pairs], 1. - self.mu)
        sh = [self.shadow[n].data for n, _ in pairs]
        torch._foreach_mul_(sh, self.mu)
        torch._foreach_add_(sh, t)

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        # Apply shadow to every param PRESENT IN SHADOW, not just requires_grad
        # ones: a param frozen AFTER registration (e.g. a frozen backbone during
        # fine-tuning) must still get its shadow value, not the raw live weight.
        for name, param in module.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        # MERGE rather than replace: params registered on the current model
        # but absent from a loaded (older) shadow keep their registered
        # values, so resuming with newly added params does not KeyError in
        # update()/ema().  With identical keys this is a plain replace.
        self.shadow = {**self.shadow, **state_dict}

