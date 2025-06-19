import torch
import os
import typing

def save_checkpoint(model:torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    iteration: int,
                    out:str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):

    to_save = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }

    torch.save(to_save, out)


def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes] ,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer):
    to_load = torch.load(src)
    model.load_state_dict(to_load["model"])
    optimizer.load_state_dict(to_load["optimizer"])

    return to_load["iteration"]
