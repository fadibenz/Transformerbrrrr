import torch

def cross_entropy(predicted_logits: torch.Tensor,
                  targets: torch.Tensor,
                  )-> torch.Tensor:

    vocab_size = predicted_logits.size(-1)
    if torch.any(targets > vocab_size):
        raise ValueError("Some target indices are out of bounds of vocab size.")

    size = predicted_logits.numel() // vocab_size
    maximum = torch.max(predicted_logits, -1, keepdim=True).values
    predicted_logits = predicted_logits - maximum

    denominator = torch.sum(torch.exp(predicted_logits), -1)

    selected_logits = torch.gather(predicted_logits, -1, targets.unsqueeze(-1)).squeeze(-1)
    loss = - torch.sum((selected_logits - torch.log(denominator))) / torch.tensor(size, dtype=torch.float32, device=predicted_logits.device)


    return loss