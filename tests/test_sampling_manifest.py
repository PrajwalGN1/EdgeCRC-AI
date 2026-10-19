import json
import hashlib
from pathlib import Path
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "reports" / "mobile_deployment" / "protocol1_patch_selection_manifest.json"
HOLDOUT_PATH = REPO_ROOT / "results" / "actual_validation_keys.json"
TENSORS_DIR = REPO_ROOT / "data" / "clean_tensors"


def test_manifest_structure_and_completeness():
    assert MANIFEST_PATH.exists(), "Manifest file must exist"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(HOLDOUT_PATH, "r", encoding="utf-8") as f:
        holdout_keys = json.load(f)

    assert len(manifest) == len(holdout_keys) == 66
    for pid in holdout_keys:
        assert pid in manifest
        entry = manifest[pid]
        assert "deidentified_id" in entry
        assert "ground_truth_label" in entry
        assert "total_patches" in entry
        assert "is_capped" in entry
        assert "protocol1_vision_tensor_fingerprint" in entry
        assert "protocol2_vision_tensor_fingerprint" in entry
        assert "text_tensor_fingerprint" in entry


def test_capped_cases_exact_constraints():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    capped_count = 0
    for pid, entry in manifest.items():
        if entry["total_patches"] > 1000:
            capped_count += 1
            assert entry["is_capped"] is True
            assert entry["selected_patch_count"] == 1000
            indices = entry["protocol1_selected_indices"]
            assert len(indices) == 1000
            assert len(set(indices)) == 1000  # All unique
            assert max(indices) < entry["total_patches"]
            assert min(indices) >= 0
        else:
            assert entry["is_capped"] is False
            assert entry["selected_patch_count"] == entry["total_patches"]

    assert capped_count == 8, f"Expected exactly 8 cases with N > 1000, got {capped_count}"


def test_manifest_order_independence_and_fingerprints():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(HOLDOUT_PATH, "r", encoding="utf-8") as f:
        holdout_keys = json.load(f)

    # Test reverse order loading
    for pid in reversed(holdout_keys):
        entry = manifest[pid]
        vp = TENSORS_DIR / f"{pid}_vision.pt"
        vt = torch.load(vp, weights_only=True)
        if entry["is_capped"]:
            indices = torch.tensor(entry["protocol1_selected_indices"])
            vt_sub = vt[indices]
        else:
            vt_sub = vt

        fp = hashlib.sha256(vt_sub.contiguous().numpy().tobytes()).hexdigest()
        assert fp == entry["protocol1_vision_tensor_fingerprint"], f"Fingerprint mismatch for {entry['deidentified_id']}"
