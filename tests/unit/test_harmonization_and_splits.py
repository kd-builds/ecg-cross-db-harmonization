"""Unit tests verifying label masking rules, patient-disjoint splitting, and model forward passes."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from ecg_xval.harmonization.crosswalk import LabelCrosswalk
from ecg_xval.partitioning.patient_disjoint_split import create_patient_disjoint_splits
from ecg_xval.models.compact_cnn1d import CompactCNN1D
from ecg_xval.models.resnet1d import ResNet1D
from ecg_xval.evaluation.metrics import evaluate_predictions, NOT_ESTIMABLE


def test_crosswalk_masks_unannotated_classes(tmp_path: Path) -> None:
    csv_path = Path("configs/crosswalk/label_crosswalk_v1.csv")
    cw = LabelCrosswalk(csv_path, mapping_tier="primary")
    # Suppose source database only annotated AF and NSR, not STACH
    y_true, eval_mask = cw.encode_record_labels(
        source_database="ptb_xl",
        raw_codes=["164889003"],
        annotated_universe={"AF", "NSR"},
    )
    af_idx = cw.class_to_idx["AF"]
    stach_idx = cw.class_to_idx["STACH"]
    assert y_true[af_idx] == 1.0
    assert eval_mask[af_idx] == 1.0
    assert eval_mask[stach_idx] == 0.0  # Masked rather than silently converted to negative


def test_patient_disjoint_splits_have_zero_overlap() -> None:
    df = pd.DataFrame({
        "record_id": [f"rec_{i}" for i in range(100)],
        "patient_id": [f"pat_{i // 2}" for i in range(100)],
    })
    train_df, val_df, test_df = create_patient_disjoint_splits(df, group_col="patient_id")
    assert set(train_df["patient_id"]).isdisjoint(set(val_df["patient_id"]))
    assert set(train_df["patient_id"]).isdisjoint(set(test_df["patient_id"]))
    assert set(val_df["patient_id"]).isdisjoint(set(test_df["patient_id"]))


def test_models_forward_shape() -> None:
    x = torch.randn(4, 12, 2500)
    compact = CompactCNN1D(in_channels=12, num_classes=5)
    resnet = ResNet1D(in_channels=12, num_classes=5)
    assert compact(x).shape == (4, 5)
    assert resnet(x).shape == (4, 5)


def test_metrics_report_not_estimable_when_class_absent() -> None:
    y_true = np.array([[1, 0], [0, 0], [1, 0]], dtype=np.float32)  # Class 1 has 0 positives
    y_prob = np.array([[0.9, 0.2], [0.1, 0.3], [0.8, 0.1]], dtype=np.float32)
    eval_mask = np.ones_like(y_true)
    thresholds = np.array([0.5, 0.5])
    res = evaluate_predictions(y_true, y_prob, eval_mask, thresholds, ["AF", "RARE_CLASS"])
    assert res["per_class"]["RARE_CLASS"]["auroc"] == NOT_ESTIMABLE
    assert res["macro_f1_shared_set"] == NOT_ESTIMABLE
