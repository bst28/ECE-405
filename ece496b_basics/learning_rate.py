from __future__ import annotations

# These are libraries we need for:
# - reading command line settings
# - saving CSV
# - math functions
# - randomness
# - timing
# - file paths
import argparse
import csv
import math
import random
import time
from pathlib import Path

# Plotting + arrays
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# PyTorch for deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Small helper functions (utils)
# ----------------------------

def set_seed(seed: int = 0) -> None:
    """
    Make randomness repeatable so results are consistent.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def open_memmap_tokens(path: str | Path, dtype: str = "int32") -> np.memmap:
    """
    Open a large token file without fully loading it into RAM.
    """
    return np.memmap(path, mode="r", dtype=dtype)


def parse_lr_list(text: str) -> list[float]:
    """
    Convert something like "1e-4,3e-4,1e-3" into a list of floats.
    """
    vals: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if item:
            vals.append(float(item))
    if not vals:
        raise ValueError("No valid learning rates found in --lrs")
    return vals


def maybe_load_tokens(path: str | None, dtype: str, vocab_size: int, fallback_size: int) -> np.ndarray:
    """
    If a real token file exists, use it.
    Otherwise, create fake random tokens (just so the script can run).
    """
    if path is not None and Path(path).exists():
        return open_memmap_tokens(path, dtype=dtype)

    print(f"[warn] No token file at {path}. Using synthetic tokens (won't hit 1.45 target).")
    rng = np.random.default_rng(0)
    return rng.integers(0, vocab_size, size=(fallback_size,), dtype=np.int32)


def get_batch_from_tokens(
    tokens: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Take random chunks of tokens to make:
      xb = input tokens
      yb = the same tokens shifted by 1 (the “next token” targets)
    """
    n = int(tokens.shape[0])
    m = int(context_length)
    if n < m + 1:
        raise ValueError(f"Token buffer too short: len={n}, need >= {m+1}")

    # Pick random starting positions for each example in the batch
    starts = np.random.randint(0, n - m, size=(batch_size,))

    # Build input sequences of length m
    xb = np.stack([tokens[s : s + m] for s in starts], axis=0)

    # Build target sequences: same but shifted by 1 token
    yb = np.stack([tokens[s + 1 : s + 1 + m] for s in starts], axis=0)

    # Convert numpy -> torch tensors, move to device
    return (
        torch.from_numpy(xb).long().to(device, non_blocking=True),
        torch.from_numpy(yb).long().to(device, non_blocking=True),
    )


# ----------------------------
# Tiny model (a small language model)
# ----------------------------

class TinyLM(nn.Module):
    """
    A very small language model:
      tokens -> embedding -> layernorm -> MLP -> vocab logits
    """
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)    # turn token IDs into vectors
        self.ln = nn.LayerNorm(d_model)                 # normalize for stability
        self.fc1 = nn.Linear(d_model, 4 * d_model)      # expand features
        self.fc2 = nn.Linear(4 * d_model, vocab_size)   # map back to vocab logits

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx shape: (batch, seq_len)
        x = self.emb(idx)               # -> (batch, seq_len, d_model)
        x = self.ln(x)                  # normalize
        x = F.silu(self.fc1(x))         # nonlinearity
        return self.fc2(x)              # -> (batch, seq_len, vocab_size)


def forward_and_loss(model: nn.Module, xb: torch.Tensor, yb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the model forward and compute cross-entropy loss for next-token prediction.
    """
    logits = model(xb)  # (B, T, V)

    # cross_entropy expects 2D (N, V) and targets (N,), so we flatten B*T into N
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
    return logits, loss


@torch.no_grad()
def eval_val_loss(
    model: nn.Module,
    val_tokens: np.ndarray,
    val_steps: int,
    batch_size: int,
    context_length: int,
    device: str,
) -> float:
    """
    Evaluate average validation loss by sampling a few batches.
    """
    model.eval()
    losses: list[float] = []

    for _ in range(val_steps):
        xb, yb = get_batch_from_tokens(val_tokens, batch_size, context_length, device)
        _, loss = forward_and_loss(model, xb, yb)
        losses.append(float(loss.item()))

    model.train()
    return float(np.mean(losses))


def train_one_lr(
    lr: float,
    train_tokens: np.ndarray,
    val_tokens: np.ndarray,
    vocab_size: int,
    d_model: int,
    max_steps: int,
    eval_every: int,
    val_steps: int,
    warmup_steps: int,
    batch_size: int,
    context_length: int,
    device: str,
    diverge_threshold: float,
) -> dict:
    """
    Train the model using ONE learning rate (lr).
    Track validation loss over time.
    """
    set_seed(0)

    # Create model and move it to CPU or GPU
    model = TinyLM(vocab_size=vocab_size, d_model=d_model).to(device)
    model.train()

    # Create optimizer (AdamW) with this learning rate
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    def lr_schedule(step: int) -> float:
        """
        Learning rate schedule:
          - warmup for first warmup_steps
          - then cosine decay until max_steps
        """
        if warmup_steps > 0 and step < warmup_steps:
            return lr * (step + 1) / warmup_steps

        t = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
        return lr * 0.5 * (1.0 + math.cos(math.pi * t))

    # Lists for plotting learning curves
    val_steps_list: list[int] = []
    val_losses: list[float] = []
    diverged = False

    start = time.time()

    for step in range(max_steps):
        # Update the optimizer LR based on schedule
        for pg in opt.param_groups:
            pg["lr"] = lr_schedule(step)

        # Get a training batch
        xb, yb = get_batch_from_tokens(train_tokens, batch_size, context_length, device)

        # Clear old gradients
        opt.zero_grad(set_to_none=True)

        # Forward pass + loss
        _, loss = forward_and_loss(model, xb, yb)
        loss_value = float(loss.item())

        # Stop early if training "blows up"
        if (not np.isfinite(loss_value)) or (loss_value > diverge_threshold):
            diverged = True
            break

        # Backprop: compute gradients
        loss.backward()

        # Clip gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimizer step: update weights
        opt.step()

        # Every so often, measure validation loss and print progress
        if (step % eval_every == 0) or (step + 1 == max_steps):
            vloss = eval_val_loss(model, val_tokens, val_steps, batch_size, context_length, device)
            val_steps_list.append(step)
            val_losses.append(vloss)
            print(f"lr={lr:.2e} step={step:5d} train={loss_value:.4f} val={vloss:.4f}")

    total_time = time.time() - start
    final_val = None if len(val_losses) == 0 else val_losses[-1]

    # Return everything we need for plotting/reporting
    return {
        "lr": lr,
        "diverged": diverged,
        "val_steps": val_steps_list,
        "val_losses": val_losses,
        "final_val": final_val,
        "time_sec": total_time,
        "model_state": model.state_dict(),
    }


def plot_runs(runs: list[dict], out_path: str | Path) -> None:
    """
    Plot validation loss vs step for each learning rate and save the figure.
    """
    plt.figure()
    for r in runs:
        label = f"lr={r['lr']:.0e}" + (" (diverged)" if r["diverged"] else "")
        if r["val_steps"]:
            plt.plot(r["val_steps"], r["val_losses"], label=label)

    plt.xlabel("step")
    plt.ylabel("val loss (per-token)")
    plt.title("Learning Curves (LR Sweep)")
    plt.legend()
    plt.tight_layout()

    out_path = Path(out_path)
    plt.savefig(out_path, dpi=160)

    # If we're in an interactive backend, show the plot
    backend = matplotlib.get_backend().lower()
    if backend not in {"agg", "pdf", "ps", "svg", "cairo"}:
        plt.show(block=True)

    print(f"[plot] saved to {out_path.resolve()}")


def save_report_csv(runs: list[dict], out_csv: str | Path) -> None:
    """
    Save a small summary CSV (lr, diverged?, final val loss, time).
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lr", "diverged", "final_val_loss", "time_sec"])

        for r in runs:
            w.writerow([
                f"{r['lr']:.8g}",
                r["diverged"],
                "" if r["final_val"] is None else f"{r['final_val']:.6f}",
                f"{r['time_sec']:.2f}",
            ])

    print(f"[report] saved to {out_csv.resolve()}")


def main() -> None:
    """
    This is the entry point:
      - read command-line args
      - load tokens
      - run LR sweep
      - print report
      - save plot + CSV
      - save best model
    """
    p = argparse.ArgumentParser()

    # File paths for token data
    p.add_argument("--train_path", type=str, default=None)
    p.add_argument("--val_path", type=str, default=None)

    # Token details
    p.add_argument("--dtype", type=str, default="int32")
    p.add_argument("--vocab_size", type=int, default=50257)

    # Model + training setup
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--context_length", type=int, default=128)

    # How long to train and how often to evaluate
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--val_steps", type=int, default=20)
    p.add_argument("--warmup_steps", type=int, default=200)

    # Learning rates to try (a sweep)
    p.add_argument("--lrs", type=str, default="1e-5,3e-5,1e-4,3e-4,1e-3,3e-3")

    # If no real token file exists, create random tokens of this size
    p.add_argument("--synthetic_tokens", type=int, default=500_000)

    # Output files
    p.add_argument("--plot_out", type=str, default="learning_curves.png")
    p.add_argument("--report_csv", type=str, default="lr_sweep_report.csv")
    p.add_argument("--best_out", type=str, default="best_lr_model.pt")

    # Device override (cpu/cuda)
    p.add_argument("--device", type=str, default=None)

    # When to call it “diverged”
    p.add_argument("--diverge_threshold", type=float, default=50.0)

    args = p.parse_args()

    # Decide device if not given
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Parse the learning rates string into a list of floats
    lrs = parse_lr_list(args.lrs)

    # Load train/val tokens (or fake tokens if paths are missing)
    train_tokens = maybe_load_tokens(args.train_path, args.dtype, args.vocab_size, args.synthetic_tokens)
    val_tokens = maybe_load_tokens(args.val_path, args.dtype, args.vocab_size, max(10_000, args.synthetic_tokens // 5))

    # Run training for each lr and store results
    runs: list[dict] = []
    for lr in lrs:
        print("\n" + "=" * 60)
        print(f"RUN lr={lr:.2e}")

        run = train_one_lr(
            lr=lr,
            train_tokens=train_tokens,
            val_tokens=val_tokens,
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            max_steps=args.max_steps,
            eval_every=args.eval_every,
            val_steps=args.val_steps,
            warmup_steps=args.warmup_steps,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=device,
            diverge_threshold=args.diverge_threshold,
        )
        runs.append(run)

    # Print final summary (useful for your writeup)
    print("\nFINAL REPORT (paste this into Part A)")
    for r in runs:
        if r["diverged"] or r["final_val"] is None:
            print(f"lr={r['lr']:.2e} -> DIVERGED")
        else:
            print(f"lr={r['lr']:.2e} -> final val={r['final_val']:.4f} ({r['time_sec']:.1f}s)")

    # Save learning curve plot + CSV report
    plot_runs(runs, out_path=args.plot_out)
    save_report_csv(runs, out_csv=args.report_csv)

    # Save the best model (lowest final validation loss)
    ok = [r for r in runs if (not r["diverged"]) and (r["final_val"] is not None)]
    if ok:
        best = min(ok, key=lambda x: x["final_val"])
        print(f"\nBEST lr={best['lr']:.2e} final val={best['final_val']:.4f}")
        torch.save(best["model_state"], args.best_out)
        print(f"[best] saved model_state to {Path(args.best_out).resolve()}")


# Only run main() if this file is executed directly
if __name__ == "__main__":
    main()