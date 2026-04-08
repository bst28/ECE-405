from __future__ import annotations

import json
import importlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


# This is a simple “record” (container) that stores one logging event.
# Each event = one line in the log file.
@dataclass
class LogEvent:
    step: int                 # training step number
    wall_time_s: float        # how many seconds since the run started
    split: str                # which dataset split: "train", "val", or "test"
    metrics: Dict[str, Any]   # values we want to log, like loss, lr, accuracy, etc.


class ExperimentLogger:
    """
    This class tracks experiment results while training.
    It writes metrics to a JSONL file (one JSON object per line),
    and can also optionally log to Weights & Biases (wandb).
    """

    def __init__(
        self,
        out_dir: str | Path,                    # folder where logs will be saved
        config: Optional[Dict[str, Any]] = None,# run configuration settings
        use_wandb: bool = False,                # whether to also log to wandb
        wandb_project: str = "cs336-assignment1",
        wandb_name: Optional[str] = None,
    ) -> None:

        # Make sure output folder exists
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Remember when the run started (so we can compute elapsed time)
        self.start_time = time.time()

        # Path for the metrics log file (JSONL)
        self.jsonl_path = self.out_dir / "metrics.jsonl"

        # Save config settings to a file so the run is reproducible
        self.config = config or {}
        (self.out_dir / "config.json").write_text(
            json.dumps(self.config, indent=2),
            encoding="utf-8"
        )

        # Optional: set up Weights & Biases if requested
        self._wandb = None
        if use_wandb:
            try:
                # Import wandb only if needed
                wandb = importlib.import_module("wandb")
                self._wandb = wandb

                # Start a wandb run
                self._wandb.init(
                    project=wandb_project,
                    name=wandb_name,
                    config=self.config
                )
            except Exception as e:
                # If wandb fails, we keep going without it
                print(f"[warn] W&B init failed; continuing without wandb. Error: {e}")
                self._wandb = None

        # Write a "header" line into the JSONL file describing the run
        header = {
            "type": "run_start",
            "wall_time_s": 0.0,
            "config_path": str(self.out_dir / "config.json"),
        }
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(header) + "\n")


    # This returns how many seconds have passed since the run started
    def wall_time_s(self) -> float:
        return float(time.time() - self.start_time)


    # This writes one “event” (like train loss at a step) into the JSONL file
    def log(self, step: int, split: str, metrics: Dict[str, Any]) -> None:

        # Create a LogEvent object with the info for this step
        ev = LogEvent(
            step=int(step),
            wall_time_s=self.wall_time_s(),
            split=str(split),
            metrics=dict(metrics),
        )

        # Append this event as one JSON line into metrics.jsonl
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(ev)) + "\n")

        # If wandb is enabled, also send the metrics there
        if self._wandb is not None:
            # Flatten keys like "train/loss", "val/loss", etc.
            flat = {f"{split}/{k}": v for k, v in metrics.items()}
            flat[f"{split}/wall_time_s"] = ev.wall_time_s

            # Log to wandb at this step
            self._wandb.log(flat, step=ev.step)


    # This closes out wandb logging nicely at the end
    def close(self) -> None:
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                # Ignore errors when closing
                pass