#!/usr/bin/env python3
"""CLI entrypoint for Weeks 4-5: Build versioned label crosswalk masks and patient-disjoint partitions."""
import argparse
from pathlib import Path
from ecg_xval.harmonization.crosswalk import LabelCrosswalk


def main() -> None:
    parser = argparse.ArgumentParser(description="Harmonize labels and generate patient-disjoint splits.")
    parser.add_argument("--crosswalk", type=Path, default=Path("configs/crosswalk/label_crosswalk_v1.csv"))
    args = parser.parse_args()

    cw = LabelCrosswalk(args.crosswalk, mapping_tier="primary")
    print(f"Loaded primary crosswalk with canonical shared classes: {cw.canonical_classes}")


if __name__ == "__main__":
    main()
