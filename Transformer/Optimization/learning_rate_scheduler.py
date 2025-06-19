import math

def learning_rate_scheduler(t: int, T_w: int, T_c: int,
                            max_lr: float, min_lr: float) -> float:
    new_lr = min_lr

    if t < T_w:
        new_lr = (t/T_w) * max_lr
    elif (t >= T_w) and (t <= T_c):
        frequency = ((t - T_w) / (T_c - T_w)) * math.pi
        new_lr = min_lr + 0.5 * ( 1 + math.cos(frequency)) * (max_lr - min_lr)

    return new_lr
