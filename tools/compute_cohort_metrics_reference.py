#!/usr/bin/env python3
"""
compute_cohort_metrics_reference.py
Computes exact reference evaluation metrics across the 66-case holdout dataset
(Protocol 1 and Protocol 2) using scikit-learn and manual step integration,
and exports them to JSON for Android JVM unit test verification.
"""

import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, average_precision_score, f1_score
)

def compute_manual_roc_and_pr_auc(y_true, y_score):
    desc_score_indices = np.argsort(y_score, kind='mergesort')[::-1]
    y_score_sorted = y_score[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]
    
    distinct_value_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_sorted.size - 1]
    
    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    
    # ROC AUC
    tps_roc = np.r_[0, tps]
    fps_roc = np.r_[0, fps]
    fpr = fps_roc / fps_roc[-1]
    tpr = tps_roc / tps_roc[-1]
    roc_auc = float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2.0))
    
    # PR AUC
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    r_diff = np.diff(np.r_[0, recall])
    pr_auc = float(np.sum(r_diff * precision))
    
    roc_points = [{"fpr": float(fpr[i]), "tpr": float(tpr[i])} for i in range(len(fpr))]
    pr_points = [{"recall": float(recall[i]), "precision": float(precision[i])} for i in range(len(recall))]
    
    return roc_auc, pr_auc, roc_points, pr_points

def main():
    repo_root = Path(__file__).resolve().parents[1]
    benchmark_path = repo_root / "reports" / "mobile_deployment" / "phase1_fp32_benchmark_data.json"
    
    with open(benchmark_path, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    results = {}
    
    for proto_key, label_name in [
        ("per_patient_reference_data_protocol_1_capped_seed42", "protocol_1"),
        ("per_patient_reference_data_protocol_2_uncapped", "protocol_2")
    ]:
        pdata = bench_data[proto_key]
        specimen_records = []
        for d in pdata:
            specimen_records.append({
                "specimen_id": d["patient_key"],
                "ground_truth_label": int(d["ground_truth_label"]),
                "ground_truth_class": "MSI" if d["ground_truth_label"] == 1 else "MSS",
                "predicted_probability": float(d["fp32_probability"]),
                "predicted_class": int(d["predicted_class_id"]),
                "predicted_label": d["predicted_class_name"],
                "decision_threshold": 0.83
            })
            
        y_true = np.array([r["ground_truth_label"] for r in specimen_records])
        y_prob = np.array([r["predicted_probability"] for r in specimen_records])
        y_pred = (y_prob >= 0.83).astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]
        
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred))
        rec = float(recall_score(y_true, y_pred))
        spec = float(tn / (tn + fp))
        bacc = float((rec + spec) / 2.0)
        npv = float(tn / (tn + fn))
        f1 = float(f1_score(y_true, y_pred))
        
        sk_roc = float(roc_auc_score(y_true, y_prob))
        sk_pr = float(average_precision_score(y_true, y_prob))
        
        man_roc, man_pr, roc_pts, pr_pts = compute_manual_roc_and_pr_auc(y_true, y_prob)
        
        results[label_name] = {
            "total_cases": len(specimen_records),
            "decision_threshold": 0.83,
            "confusion_matrix": {
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn
            },
            "metrics": {
                "accuracy": acc,
                "precision": prec,
                "recall_sensitivity": rec,
                "specificity": spec,
                "balanced_accuracy": bacc,
                "npv": npv,
                "f1_score": f1,
                "roc_auc_sklearn": sk_roc,
                "roc_auc_manual": man_roc,
                "pr_auc_sklearn": sk_pr,
                "pr_auc_manual": man_pr
            },
            "roc_curve_points_count": len(roc_pts),
            "pr_curve_points_count": len(pr_pts),
            "specimens": specimen_records
        }
        
    out_path = repo_root / "reports" / "mobile_deployment" / "cohort_evaluation_reference_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"Computed reference metrics and saved to {out_path}")
    for p in ["protocol_1", "protocol_2"]:
        m = results[p]["metrics"]
        cm = results[p]["confusion_matrix"]
        print(f"[{p}] TP={cm['tp']}, TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}")
        print(f"      Acc={m['accuracy']:.6f}, Prec={m['precision']:.6f}, Rec={m['recall_sensitivity']:.6f}, Spec={m['specificity']:.6f}")
        print(f"      F1={m['f1_score']:.6f}, ROC-AUC={m['roc_auc_sklearn']:.6f}, PR-AUC={m['pr_auc_sklearn']:.6f}")

if __name__ == "__main__":
    main()
