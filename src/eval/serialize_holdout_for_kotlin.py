#!/usr/bin/env python3
"""
serialize_holdout_for_kotlin.py

Serializes the frozen 66-case holdout dataset into binary format for
Kotlin/Android CrcMultimodalClassifier numerical parity verification.

Creates:
  - android/app/build/holdout_data/holdout_protocol1.bin
  - android/app/build/holdout_data/holdout_protocol2.bin
"""

import sys
import os
import json
import struct
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
TENSORS_DIR = REPO_ROOT / "data" / "clean_tensors"
HOLDOUT_PATH = REPO_ROOT / "results" / "actual_validation_keys.json"
MANIFEST_PATH = REPO_ROOT / "reports" / "mobile_deployment" / "protocol1_patch_selection_manifest.json"
GOLDEN_PATH = REPO_ROOT / "reports" / "mobile_deployment" / "phase1_fp32_benchmark_data.json"
OUTPUT_DIR = REPO_ROOT / "android" / "app" / "build" / "holdout_data"


def serialize_protocol(protocol_name: str, output_path: Path):
    holdout_keys = json.load(open(HOLDOUT_PATH, "r"))
    manifest = json.load(open(MANIFEST_PATH, "r"))
    golden = json.load(open(GOLDEN_PATH, "r"))

    if protocol_name == "protocol_1":
        ref_list = golden["per_patient_reference_data_protocol_1_capped_seed42"]
    else:
        ref_list = golden["per_patient_reference_data_protocol_2_uncapped"]

    ref_by_pid = {r["patient_key"]: r for r in ref_list}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Serializing {len(holdout_keys)} cases for {protocol_name} to {output_path}...")

    with open(output_path, "wb") as f:
        # Write number of cases
        f.write(struct.pack("<i", len(holdout_keys)))

        for idx, pid in enumerate(holdout_keys):
            man_entry = manifest[pid]
            ref_entry = ref_by_pid[pid]

            vp = TENSORS_DIR / f"{pid}_vision.pt"
            tp = TENSORS_DIR / f"{pid}_text.pt"

            vt = torch.load(vp, weights_only=True)
            tt = torch.load(tp, weights_only=True)

            if protocol_name == "protocol_1":
                if man_entry["is_capped"]:
                    indices = torch.tensor(man_entry["protocol1_selected_indices"])
                    vt_eval = vt[indices]
                else:
                    vt_eval = vt
            else:
                vt_eval = vt

            num_patches = vt_eval.shape[0]
            gt_label = int(ref_entry["ground_truth_label"])
            ref_logit = float(ref_entry["fp32_logit"])
            ref_prob = float(ref_entry["fp32_probability"])
            ref_call = int(ref_entry["predicted_class_id"])

            key_bytes = pid.encode("utf-8")
            f.write(struct.pack("<i", len(key_bytes)))
            f.write(key_bytes)
            f.write(struct.pack("<i", num_patches))
            f.write(struct.pack("<i", gt_label))
            f.write(struct.pack("<f", ref_logit))
            f.write(struct.pack("<f", ref_prob))
            f.write(struct.pack("<i", ref_call))

            # Text: (768,) float32
            tt_np = tt.numpy().astype(np.float32).flatten()
            assert len(tt_np) == 768, f"Expected 768 text features, got {len(tt_np)}"
            f.write(tt_np.tobytes())

            # Vision: (num_patches, 1024) float32
            vt_np = vt_eval.numpy().astype(np.float32).flatten()
            assert len(vt_np) == num_patches * 1024, f"Expected {num_patches * 1024} vision features, got {len(vt_np)}"
            f.write(vt_np.tobytes())

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {protocol_name} binary ({size_mb:.2f} MB)")


def main():
    serialize_protocol("protocol_1", OUTPUT_DIR / "holdout_protocol1.bin")
    serialize_protocol("protocol_2", OUTPUT_DIR / "holdout_protocol2.bin")
    print("Serialization completed successfully.")


if __name__ == "__main__":
    main()
