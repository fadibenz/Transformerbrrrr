import torch
import numpy.typing as npt
import numpy as np

def data_loading(dataset: npt.NDArray,
                 batch_size: int,
                 context_length: int,
                 device: torch.device
                 ) -> tuple[torch.Tensor, torch.Tensor]:

    n = len(dataset)
    if n < context_length + 1:
        raise ValueError(f"Dataset size is too short for the specified context length {context_length}")

    if batch_size > (n - context_length) // 2:
        print(f"[WARNING] batch size {batch_size} might be too big for dataset size {n}, "
              "this might lead to overfitting")

    indices = np.random.randint(0, n - context_length, size=batch_size)
    sampled_input = dataset[indices[:, None] + np.arange(context_length)]
    targets = dataset[indices[:, None] + np.arange(context_length) + 1]

    sampled_input = torch.tensor(sampled_input, device=device, dtype=torch.int64)
    targets = torch.tensor(targets, device=device, dtype=torch.int64)

    return sampled_input, targets