#!/usr/bin/env python3
"""
export_reference_labels.py
Exports authoritative ground-truth reference labels for the 66-case holdout cohort
to JSON and CSV formats for Android workstation cohort evaluation.
"""

import json
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parents[1]
    keys_path = repo_root / "results" / "actual_validation_keys.json"
    registry_path = repo_root / "data" / "patient_registry.json"

    with open(keys_path, "r", encoding="utf-8") as f:
        keys = json.load(f)
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    labels = []
    for k in keys:
        entry = registry[k]
        lbl = int(entry["label"])
        cls_name = "MSI" if lbl == 1 else "MSS"
        labels.append({
            "specimen_id": k,
            "ground_truth_label": lbl,
            "ground_truth_class": cls_name,
            "source": "CAP pathology report / TCGA validation cohort"
        })

    payload = {
        "schema_version": "1.0",
        "cohort_id": "TCGA_HOLDOUT_66",
        "description": "Independently established reference histopathology ground-truth labels for the 66-case holdout cohort",
        "positive_class": "MSI (1)",
        "negative_class": "MSS (0)",
        "total_cases": len(labels),
        "class_distribution": {
            "MSS_0": sum(1 for l in labels if l["ground_truth_label"] == 0),
            "MSI_1": sum(1 for l in labels if l["ground_truth_label"] == 1)
        },
        "labels": labels
    }

    out_dir = repo_root / "android" / "app" / "src" / "main" / "assets" / "reference_labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "tcga_holdout_reference_labels.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Also write a CSV export in data/test_bundles/
    csv_dir = repo_root / "data" / "reference_labels"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_file = csv_dir / "tcga_holdout_reference_labels.csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("specimen_id,ground_truth_label,ground_truth_class,source\n")
        for l in labels:
            f.write(f"{l['specimen_id']},{l['ground_truth_label']},{l['ground_truth_class']},{l['source']}\n")

    print(f"Exported {len(labels)} labels to {out_file}")
    print(f"Exported CSV to {csv_file}")
    print(f"Distribution: MSS_0={payload['class_distribution']['MSS_0']}, MSI_1={payload['class_distribution']['MSI_1']}")

if __name__ == "__main__":
    main()
