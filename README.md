# Cross-Database Evaluation of Deep Learning for ECG Arrhythmia Classification Using Harmonized Labels

**Author:** Riddhi Bhagat  
**Institution:** Sikkim Manipal Institute of Technology  
**Proposal Date:** 5 October 2026  

---

## 1. Project Title, Author & Overview

**Title:** Cross-Database Evaluation of Deep Learning for ECG Arrhythmia Classification Using Harmonized Labels

**Author:** Riddhi Bhagat — Sikkim Manipal Institute of Technology

### Overview

This repository implements a **locked, reproducible benchmark** for answering one central research question:

> *How well do deep learning 12-lead ECG arrhythmia classifiers generalize across independently collected hospital databases?*

The answer is tested by harmonizing labels from three very different sources — **PTB-XL**, **CPSC 2018**, and **Chapman-Shaoxing** — into a single shared taxonomy of five canonical rhythm classes:

| Canonical Class | SNOMED-CT Concept | Role |
| :--- | :--- | :--- |
| `NSR` | 426783006 | Sinus rhythm — the **comparator** (not an arrhythmia) |
| `AF` | 164889003 | Atrial fibrillation |
| `STACH` | 427084000 | Sinus tachycardia |
| `SBRAD` | 426177001 | Sinus bradycardia |
| `IAVB` | 270492004 | First-degree atrioventricular block |

### System Architecture Overview

```mermaid
flowchart LR
    subgraph Inputs["Hospital ECG Databases"]
        PTB["PTB-XL"]
        CPSC["CPSC 2018"]
        CHAP["Chapman-Shaoxing"]
    end

    subgraph Core["Core Research Pipeline"]
        AUDIT["Dataset & Provenance Audit<br/>SHA-256 hashing, deduplication, source grouping"]
        HARM["Label Harmonization<br/>SNOMED-CT → NSR / AF / STACH / SBRAD / IAVB<br/>with eval_mask logic"]
        PREP["Leakage-Safe Standardization<br/>12 leads @ 500 Hz, age/sex-aware preprocessing, patient-disjoint splits"]
        MODEL["Baseline Models & Training<br/>CompactCNN1D / ResNet1D"]
        EVAL["Locked External Transfer Evaluation<br/>Macro-F1, AUPRC, AUROC, bootstrap confidence intervals"]
        ART["Artifacts<br/>processed waveforms, splits, manifests, checkpoints, results"]
    end

    subgraph Deck["AETHER Telemetry Deck"]
        BRIDGE["FastAPI + WebSocket Bridge<br/>python scripts/05_launch_aether_deck.py"]
        UI["dashboard/aether_deck.html<br/>Live ECG UI + diagnosis overlays"]
    end

    PTB --> AUDIT
    CPSC --> AUDIT
    CHAP --> AUDIT
    AUDIT --> HARM --> PREP --> MODEL --> EVAL
    EVAL --> ART
    HARM --> BRIDGE
    MODEL --> BRIDGE
    BRIDGE --> UI
    UI --> LIVE["Real PyTorch inference<br/>Grad-CAM saliency + posterior probabilities"]
```

### Plain-English Summary

Hospitals record heart rhythms with their own conventions, and some never record a given rhythm at all. The pipeline therefore **harmonizes** raw SNOMED-CT codes into the shared classes above, and when a diagnosis is missing or uncertain it applies an explicit **`eval_mask = 0`** — i.e., the class is *masked out* of the evaluation — **rather than silently treating a missing diagnosis as a "confirmed negative."** This keeps the benchmark scientifically honest and auditable.

The same harmonized data pipeline powers two surfaces:

1. **Core Research Pipeline** (`src/ecg_xval/`, `scripts/01`–`04`) — auditing, harmonization, modeling, and locked transfer evaluation.
2. **AETHER Telemetry Deck** (`dashboard/aether_deck.html`, `scripts/05_launch_aether_deck.py`) — the interactive real-time visualization and live-inference layer built on top of the same harmonized concepts.

---

## 2. Key Functionalities (Simple & Structured Breakdown)

The repository has **two complementary halves**.

### A. Core Research Pipeline (`src/ecg_xval/` & `scripts/01`–`04`)

| # | Stage | Module(s) | Purpose |
| :- | :--- | :--- | :--- |
| 1 | **Dataset & Provenance Auditing** | `src/ecg_xval/audit/`, `scripts/01_run_dataset_audit.py` | Detects duplicate waveforms (SHA-256 hashing) and groups records by hospital origin. |
| 2 | **Versioned Label Harmonization** | `configs/crosswalk/`, `src/ecg_xval/harmonization/` | Maps raw SNOMED-CT codes → 5 canonical classes with explicit per-class eligibility / `eval_mask` rules. |
| 3 | **Leakage-Safe Standardization & Splitting** | `src/ecg_xval/preprocessing/`, `src/ecg_xval/partitioning/` | Standardizes to **12 leads @ 500 Hz** (10 s windows, mV units), fits normalizers on training data only, enforces patient-disjoint splits. |
| 4 | **Locked External Transfer & Bootstrap Evaluation** | `src/ecg_xval/evaluation/`, `scripts/04_evaluate_transfer.py` | Computes Macro-F1, AUPRC, AUROC (absent classes marked `"not estimable"`), plus 95% patient-cluster bootstrap CIs. |

### B. AETHER Real-Time Telemetry & Visual Studio (`dashboard/aether_deck.html` & `scripts/05_launch_aether_deck.py`)

**Two modes:**

- **Standalone Demo Mode** — the deck runs directly in any modern browser from `file://` with **zero server setup**: 60 FPS clinical oscilloscope, rhythm synthesis, noise/drift injection, posterior panel, onion simulator, and machine probe.
- **Live Bridge Mode** — `python scripts/05_launch_aether_deck.py` launches a FastAPI + WebSocket bridge on `http://127.0.0.1:8765` serving the dashboard and streaming **real PyTorch inference (Grad-CAM bands, posterior probabilities)**, plus **Windows host telemetry**.

**Three interactive tabs in the deck:**

1. **Tab 1 — 12-Lead ECG Synthesizer & Grad-CAM Lab** — 60 FPS clinical grid + waveform oscilloscope, rhythm switcher, baseline-drift/EMG noise injector, Grad-CAM saliency overlay, and live crosswalk posterior bars with `MASKED` badges.
2. **Tab 2 — 6-Hop v3 Onion & Channel Topology Simulator** — interactive cryptographic relay visualizer with packet animation.
3. **Tab 3 — Local Machine Probe & PowerShell Audit** — real-time CPU/GPU hardware detection, floating-point stress test, one-click read-only PowerShell audit command.

---

## 3. Complete Repository Directory Structure

```text
ecg-cross-db-harmonization/
├── .vscode/                        # VS Code settings, launch configs, recommended extensions
├── configs/
│   ├── aether/                     # AETHER deck runtime configuration (deck.yaml)
│   ├── crosswalk/                   # Versioned label crosswalk CSVs + eligibility rules
│   ├── datasets/                    # Source cohort metadata (CPSC, PTB-XL, Chapman-Shaoxing)
│   ├── experiments/                 # Locked within-db / source-to-target / LOPGO protocols
│   └── models/                      # Hyperparameter budgets (CompactCNN1D, ResNet1D)
├── dashboard/
│   └── aether_deck.html            # AETHER Telemetry Deck — live/standalone ECG UI
├── data/
│   ├── interim/                     # Audited record inventories & duplicate hashes
│   ├── manifests/                   # Dataset coverage & exclusion tables
│   ├── processed/                   # Standardized 12-lead waveforms
│   ├── raw/                         # Source ECG databases (cpsc_2018/, ptb_xl/, chapman_shaoxing/)
│   └── splits/                      # Patient-disjoint partitions
├── docs/
├── models/                          # Trained checkpoint placeholders / artifact dir
├── notebooks/                       # Audit, calibration, error-analysis notebooks
├── results/
│   ├── aether_snapshots/           # Live-deck audit snapshots + .gitkeep
│   ├── calibration/ …               # Evaluation outputs, bootstrap CIs, performance matrices
├── scripts/
│   ├── 01_run_dataset_audit.py
│   ├── 02_harmonize_and_split.py
│   ├── 03_train_baselines.py
│   ├── 04_evaluate_transfer.py
│   └── 05_launch_aether_deck.py   # FastAPI + WebSocket bridge for the AETHER deck
├── src/ecg_xval/                    # Core Python package
│   ├── audit/ …
│   ├── harmonization/ …
│   ├── preprocessing/ …
│   ├── partitioning/ …
│   ├── models/ …
│   ├── training/ …
│   ├── evaluation/ …
│   ├── analysis/ …
│   └── utils/ …
├── tests/
│   ├── integration/                 # AETHER deck regression tests (JS parse, wiring, backend)
│   └── unit/                        # Harmonization & splits tests
├── pyproject.toml
└── requirements.txt
```

---

## 4. Quick Start Guide (Windows PowerShell & VS Code)

### 4.1 Python environment

```powershell
cd ecg-cross-db-harmonization
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 4.2 Run the research pipeline

```powershell
python scripts\01_run_dataset_audit.py
python scripts\02_harmonize_and_split.py
python scripts\03_train_baselines.py
python scripts\04_evaluate_transfer.py
```

### 4.3 Launch the AETHER Live Bridge

```powershell
python scripts\05_launch_aether_deck.py
# or via VS Code: press F5 and select "5. Launch AETHER Telemetry Deck"
# → serves dashboard/aether_deck.html at http://127.0.0.1:8765
```

### 4.4 Run the test suite

```powershell
pytest
```

---

## 5. 16-Week Research Milestone Table

| Weeks | Planned Activity | Directory / Module |
| :--- | :--- | :--- |
| **Weeks 1–3** | Literature & dataset/label audit | `src/ecg_xval/audit/`, `docs/audit_reports/` |
| **Weeks 4–5** | Harmonization & leakage checks | `configs/crosswalk/`, `src/ecg_xval/harmonization/`, `src/ecg_xval/partitioning/` |
| **Weeks 6–9** | Preprocessing & baselines | `src/ecg_xval/preprocessing/`, `src/ecg_xval/models/`, `src/ecg_xval/training/` |
| **Weeks 10–12** | Locked external evaluation | `src/ecg_xval/evaluation/`, `results/source_to_target/`, `results/leave_one_group_out/` |
| **Weeks 13–14** | Sensitivity / error analysis | `src/ecg_xval/analysis/`, `results/sensitivity_analysis/` |
| **Weeks 15–16** | Writing & reproducibility package | `docs/manuscript/`, `docs/protocol/reporting_checklist.md` |

---

## License & Citation

*Locked protocol version: `configs/crosswalk/eligibility_rules.yaml` → `protocol_version: "1.0.0-locked"`.
*Audit snapshots from the live deck are persisted under `results/aether_snapshots/`.
