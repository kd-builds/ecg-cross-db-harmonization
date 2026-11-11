#!/usr/bin/env python3
"""CLI entrypoint for Weeks 1-3: Audit candidate ECG databases for provenance and label coverage."""
import argparse
from pathlib import Path
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit candidate 12-lead ECG databases.")
    parser.add_argument("--config", type=Path, default=Path("configs/datasets/cohorts.yaml"))
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print(f"Loaded audit configuration for {len(cfg.get('cohorts', {}))} candidate cohorts:")
    for key, meta in cfg.get("cohorts", {}).items():
        print(f"  - [{key}] {meta['name']} (provenance_group={meta['provenance_group']})")


if __name__ == "__main__":
    main()
