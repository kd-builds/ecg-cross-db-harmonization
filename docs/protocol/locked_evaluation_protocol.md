# Locked External Evaluation Protocol

## 1. Pre-Test Freeze Checklist
Before running any external evaluation (`scripts/04_evaluate_transfer.py`), verify and freeze:
- [ ] Candidate dataset audit and provenance group assignments (`data/manifests/`)
- [ ] Primary label crosswalk (`configs/crosswalk/label_crosswalk_v1.csv`)
- [ ] Per-class eligibility and masking rules (`configs/crosswalk/eligibility_rules.yaml`)
- [ ] Preprocessing parameters fitted strictly on source training partitions
- [ ] Model checkpoints, decision thresholds, and calibration maps selected strictly on source validation sets

## 2. Non-Negotiable Leakage Controls
1. **Patient & Provenance Disjointness:** Records from the same patient or overlapping provenance group must never cross train/validation/test boundaries.
2. **No Silent Negative Conversion:** If a database did not annotate a candidate class, records from that database must be masked for that class unless the annotation protocol explicitly supports interpreting absence as a confirmed negative.
3. **No Target-Driven Tuning:** External transfer evaluation (pairwise source-to-target and leave-one-provenance-group-out) must use zero target labels and zero target-validation tuning.
