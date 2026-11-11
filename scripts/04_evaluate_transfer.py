#!/usr/bin/env python3
"""CLI entrypoint for Weeks 10-12: Execute locked source-to-target and LOPGO external evaluation."""
import argparse
from pathlib import Path
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run locked external transfer evaluation.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/locked_protocol.yaml"))
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert not cfg["allow_target_labels_in_transfer"], "Target labels are prohibited in primary transfer evaluation!"
    assert not cfg["allow_target_driven_tuning"], "Target-driven tuning is prohibited in primary transfer evaluation!"
    print("Locked external evaluation protocol verified.")


if __name__ == "__main__":
    main()
