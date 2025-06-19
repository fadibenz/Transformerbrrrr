import torch.optim as optim
from typing import Callable, Optional


class AdamW(optim.Optimizer):
    def __init__(self, params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps = 1e-8,
                 weight_decay = 0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate {lr}")
        if betas[0] < 0:
            raise ValueError(f"Invalid Beta1 {betas[0]}")
        if betas[1] < 0:
            raise ValueError(f"Invalid Beta2 {betas[1]}")

        defaults = {"lr": lr,
                    "Beta1": betas[0],
                    "Beta2": betas[1],
                    "eps": eps,
                    "lambda": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            Beta1 = group["Beta1"]
            Beta2 = group["Beta2"]
            eps = group["eps"]
            lambd = group["lambda"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", 0)
                v = state.get("v", 0)

                grad = p.grad

                m = Beta1 * m + (1 - Beta1) * grad
                v = Beta2 * v + (1 - Beta2) * (grad ** 2)

                adjusted_alpha = lr * ((1 - (Beta2 ** t)) ** 0.5
                                       / (1 - (Beta1 ** t)))
                p.data = p.data - (adjusted_alpha * (m / (v ** 0.5 + eps)))
                p.data = p.data - (lambd * lr * p.data)

                state["m"] = m
                state["v"] = v
                state["t"] = t + 1

        return loss
