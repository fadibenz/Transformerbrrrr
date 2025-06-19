import numpy as np
import torch
import numpy.typing as npt

from data.Utils.checkpointing import save_checkpoint
from data.Utils.data_loading import data_loading

from Transformer.Transformer import Transformer
from Transformer.Optimization.AdamW import AdamW
from Transformer.Optimization.cross_entropy import cross_entropy
from Transformer.Optimization.learning_rate_scheduler import learning_rate_scheduler
from Transformer.Utils.gradient_clipping import gradient_clipping

import yaml
import argparse
import wandb
from pathlib import Path
import pathlib
import typing

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def initialize_model_optimizer(run_config, device):

    # Model Hyperparams
    vocab_size = run_config.training["model"]["vocab_size"]
    context_length = run_config.training["model"]["context_length"]
    d_model = run_config.training["model"]["d_model"]
    d_ff = run_config.training["model"]["d_ff"]
    num_layers = run_config.training["model"]["num_layers"]
    num_heads = run_config.training["model"]["num_heads"]
    d_k = run_config.training["model"]["d_k"]
    d_v = run_config.training["model"]["d_v"]
    theta = run_config.training["model"]["RoPE_theta"]
    use_rope = False

    # Optimizer Hyperparams
    beta_1 = run_config.training["optimizer"]["beta_1"]
    beta_2 = run_config.training["optimizer"]["beta_2"]
    weight_decay = run_config.training["optimizer"]["weight_decay"]

    if theta is not None:
        use_rope = True

    # Model initialization
    model = Transformer(vocab_size, context_length, num_layers,
                        d_model, num_heads, d_ff, d_k, d_v,
                        theta, use_rope, device)

    # Optimizer Initialization
    optimizer = AdamW(model.parameters(),
                      betas=(beta_1, beta_2),
                      weight_decay=weight_decay
                      )

    return model, optimizer

def load_data(train_path, valid_path):

    training_data = np.load(train_path, mmap_mode="r")
    validation_data = np.load(valid_path, mmap_mode="r")
    print(f"Loaded training data with {len(training_data):,} tokens.")
    print(f"Loaded validation data with {len(validation_data):,} tokens.")
    return training_data, validation_data

def compute_batch_loss(model: torch.nn.Module,
                        data: npt.NDArray,
                        batch_size: int,
                        context_length:int,
                        device: torch.device) -> torch.Tensor:
    training_batch, targets = data_loading(data, batch_size, context_length, device=device)
    predicted_logits = model(training_batch)
    loss = cross_entropy(predicted_logits, targets)
    return loss

def evaluate_on_dataset(model: torch.nn.Module,
                        data: npt.NDArray,
                        batch_size: int,
                        context_length: int,
                        device: torch.device) -> tuple[float, float]:
    """
    Computes loss and perplexity over an entire dataset deterministically.
    """
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(data) - (context_length * batch_size), context_length * batch_size):
            batch_indices = range(i, i + context_length * batch_size, context_length)

            # Ensure the batch doesn't go out of bounds
            if batch_indices[-1] + context_length + 1 > len(data):
                break

            inputs_list = [data[start: start + context_length] for start in batch_indices]
            targets_list = [data[start + 1: start + 1 + context_length] for start in batch_indices]

            inputs = torch.from_numpy(np.array(inputs_list).astype(np.int64)).to(device)
            targets = torch.from_numpy(np.array(targets_list).astype(np.int64)).to(device)

            predicted_logits = model(inputs)
            loss = cross_entropy(predicted_logits, targets)
            losses.append(loss.item())

    if not losses:
        return float('inf'), float('inf')

    avg_loss = np.mean(losses)
    perplexity = np.exp(avg_loss)

    return avg_loss, perplexity


def train_model(config: dict,
                output_path: str | pathlib.PurePath | typing.BinaryIO | typing.IO[bytes]):
    seed = 2025
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    wandb.init(project="transformer_experiments_tiny_stories", config=config,
             name=config.get("wandb_run_name", None))

    run_config = wandb.config
    print(f"\nRunning with config:\n {run_config}")

    # Training Hyperparams
    min_learning_rate = run_config.training["min_lr"]
    max_learning_rate = run_config.training["max_lr"]
    warmup_steps = run_config.training["warmup_steps"]
    annealing_steps = run_config.training["annealing_steps"]
    batch_size = run_config.training["batch_size"]
    epochs = run_config.training["epochs"]
    max_l2_norm = run_config.training["max_l2_norm"]
    context_length = run_config.training["model"]["context_length"]
    logging_freq = run_config.training["logging_freq"]
    train_data_path = run_config.training["train_data_path"]
    valid_data_path = run_config.training["valid_data_path"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, optimizer = initialize_model_optimizer(run_config, device)

    training_data, validation_data = load_data(train_data_path, valid_data_path)

    number_iterations = len(training_data) // (batch_size * context_length)
    global_step = 0
    cumulated_loss = 0

    for i in range(epochs):
        for j in range(number_iterations):
            model.train()
            optimizer.param_groups[0]['lr'] = learning_rate_scheduler(global_step, warmup_steps, annealing_steps,
                                                    max_learning_rate, min_learning_rate)
            optimizer.zero_grad()

            loss = compute_batch_loss(model, training_data, batch_size, context_length, device)
            loss.backward()
            # Gradient Clipping
            gradient_clipping(model.parameters(),max_l2_norm)
            optimizer.step()
            global_step += 1
            cumulated_loss += loss.item()

            if (global_step % logging_freq) == 0:
                avg_train_loss = cumulated_loss / logging_freq
                print(f"Step {global_step}: Running evaluation...")
                val_loss, val_perplexity = evaluate_on_dataset(model, validation_data, batch_size, context_length,
                                                               device)
                print(f"Step {global_step}: Val Perplexity: {val_perplexity:.4f}")
                wandb.log({
                    "training_loss": avg_train_loss,
                    "training_perplexity": np.exp(avg_train_loss),
                    "validation_loss": val_loss,
                    "validation_perplexity": val_perplexity,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "global_step": global_step,
                })
                out_path = output_path / f"checkpoint_{global_step}.pt"
                save_checkpoint(model, optimizer, global_step, out_path)
                cumulated_loss = 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Transformer Training Script")
    parser.add_argument("--config_path", required=True, type=str, help="Path to base configuration file")
    parser.add_argument("--output_path", required=True, type=str, help="Path to output directory")

    args = parser.parse_args()
    base_config = load_config(Path(args.config_path))
    output_path = Path(args.output_path)
    Path.mkdir(output_path, parents=True, exist_ok=True)

    train_model(base_config, output_path)
