#!/usr/bin/env python3
"""CLI entrypoint for Weeks 6-9: Train Compact 1D CNN and Residual 1D CNN under matched budgets."""
import argparse
from pathlib import Path
import yaml
from ecg_xval.utils.reproducibility import lock_random_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train deep learning ECG baselines.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/locked_protocol.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    lock_random_seed(args.seed)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"Initialized baseline training for experiment '{cfg['experiment_name']}' with seed={args.seed}")


if __name__ == "__main__":
    main()
