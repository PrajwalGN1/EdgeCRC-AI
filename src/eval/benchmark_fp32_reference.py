#!/usr/bin/env python3
"""
benchmark_fp32_reference.py

Phase 1 — FP32 Reference Benchmark & Golden Standard Baseline
Generates reproducible reference outputs (logits, probabilities, classifications, latencies, memory)
for the trained SummaryToPatchCrossAttention model across the 66 holdout validation patients.
Outputs a structured JSON database to reports/mobile_deployment/phase1_fp32_benchmark_data.json.
"""

import os
import sys
import time
import json
import psutil
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from architectures.cross_attention_fusion import SummaryToPatchCrossAttention

# Configuration
CHECKPOINT_PATH = REPO_ROOT / "models" / "fusion_model_v1_1.pth"
REGISTRY_PATH = REPO_ROOT / "data" / "patient_registry.json"
HOLDOUT_PATH = REPO_ROOT / "results" / "actual_validation_keys.json"
TENSORS_DIR = REPO_ROOT / "data" / "clean_tensors"
OUTPUT_JSON_PATH = REPO_ROOT / "reports" / "mobile_deployment" / "phase1_fp32_benchmark_data.json"

DECISION_THRESHOLD = 0.83
WARMUP_REPS = 2
LATENCY_REPS = 5


def get_process_memory_mb():
    """Returns current process Resident Set Size (RSS) in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def compute_logit_and_prob(model, vision_seq, text_summary):
    """
    Executes cross-attention and classification layers to extract both
    pre-sigmoid logit and post-sigmoid probability.
    """
    if len(text_summary.shape) == 2:
        text_summary = text_summary.unsqueeze(1)
    if len(vision_seq.shape) == 2:
        vision_seq = vision_seq.unsqueeze(0)

    # 1. Text projection
    text_query = model.text_projection(text_summary)

    # 2. Multihead cross-attention
    attn_output, attn_weights = model.cross_attn(
        query=text_query,
        key=vision_seq,
        value=vision_seq
    )

    # 3. Residual & LayerNorm
    fused_features = model.layer_norm(text_query + attn_output)
    pooled_features = fused_features.squeeze(1)

    # 4. Classifier layers
    h = model.classifier[0](pooled_features)  # Linear(1024, 256)
    h = model.classifier[1](h)                # ReLU()
    h = model.classifier[2](h)                # Dropout (identity in eval mode)
    logit = model.classifier[3](h)            # Linear(256, 1) -> pre-sigmoid logit
    prob = torch.sigmoid(logit)               # Sigmoid() -> probability

    return logit.item(), prob.item(), attn_weights.squeeze(0).squeeze(0).cpu().numpy()


def run_benchmark():
    print("=" * 80)
    print("PHASE 1: FP32 REFERENCE BENCHMARK EXECUTION")
    print("=" * 80)

    # Environment
    device = torch.device("cpu")
    num_threads = torch.get_num_threads()
    print(f"Device: {device}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"CPU Threads: {num_threads}")
    print(f"Decision Threshold: {DECISION_THRESHOLD}")

    # Load Patient Registry & Holdout Keys
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)
    with open(HOLDOUT_PATH, "r", encoding="utf-8") as f:
        holdout_keys = json.load(f)

    print(f"Registry Total: {len(registry)} patients")
    print(f"Holdout Cohort: {len(holdout_keys)} patients")

    # Load Model Checkpoint strictly
    model = SummaryToPatchCrossAttention(vision_dim=1024, text_dim=768, num_heads=8)
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    load_res = model.load_state_dict(state_dict, strict=True)
    print(f"Loaded weights strictly from {CHECKPOINT_PATH.name}: {load_res}")
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    weight_memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"Model Parameters: {total_params:,} ({weight_memory_mb:.2f} MB FP32)")

    # Global warmup
    print(f"\nExecuting {WARMUP_REPS} initial system warmup passes...")
    dummy_v = torch.randn(1, 100, 1024, dtype=torch.float32)
    dummy_t = torch.randn(1, 768, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(WARMUP_REPS):
            _ = model(dummy_v, dummy_t)

    base_mem_mb = get_process_memory_mb()
    print(f"Baseline Process Memory: {base_mem_mb:.2f} MB")

    print("\nPreloading 66 holdout tensors into memory...")
    cached_tensors = {}
    for pid in holdout_keys:
        v_p = TENSORS_DIR / f"{pid}_vision.pt"
        t_p = TENSORS_DIR / f"{pid}_text.pt"
        cached_tensors[pid] = (
            torch.load(v_p, map_location="cpu", weights_only=True),
            torch.load(t_p, map_location="cpu", weights_only=True)
        )
    print(f"Preloaded {len(cached_tensors)} holdout tensor pairs.")

    # =========================================================================
    # PROTOCOL A: FULL UNCAPPED SEQUENCE (GOLDEN STANDARD REFERENCE)
    # =========================================================================
    print("\n" + "-" * 80)
    print("EXECUTING PROTOCOL A: FULL UNCAPPED SEQUENCE (ALL TISSUE PATCHES PRESERVED)")
    print("-" * 80)

    records_protocol_a = []
    latencies_all = []

    for idx, pid in enumerate(holdout_keys, start=1):
        vt, tt = cached_tensors[pid]

        num_patches = vt.shape[0]
        gt_label = registry[pid]["label"]
        gt_class = "MSI" if gt_label == 1 else "MSS"

        # Warmup per patient
        with torch.no_grad():
            for _ in range(WARMUP_REPS):
                _ = model(vt, tt)

            # Benchmark inference latency
            times_ms = []
            for _ in range(LATENCY_REPS):
                t0 = time.perf_counter()
                logit_val, prob_val, attn_vec = compute_logit_and_prob(model, vt, tt)
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)

        latencies_all.extend(times_ms)
        mean_lat = float(np.mean(times_ms))
        median_lat = float(np.median(times_ms))
        std_lat = float(np.std(times_ms))
        min_lat = float(np.min(times_ms))
        max_lat = float(np.max(times_ms))

        # Classification decision
        pred_class_id = 1 if prob_val >= DECISION_THRESHOLD else 0
        pred_class_name = "MSI" if pred_class_id == 1 else "MSS"
        is_correct = (pred_class_id == gt_label)

        # Classification outcome
        if gt_label == 1 and pred_class_id == 1:
            outcome = "TP"
        elif gt_label == 0 and pred_class_id == 0:
            outcome = "TN"
        elif gt_label == 0 and pred_class_id == 1:
            outcome = "FP"
        else:
            outcome = "FN"

        # Confidence percentage relative to call
        if prob_val >= DECISION_THRESHOLD:
            confidence_pct = prob_val * 100.0
        else:
            confidence_pct = (1.0 - prob_val) * 100.0

        # Attention distribution metrics
        # Clip small values for stable entropy calculation
        attn_safe = np.clip(attn_vec, 1e-12, 1.0)
        entropy = float(-np.sum(attn_safe * np.log(attn_safe)))
        max_attn = float(np.max(attn_vec))
        min_attn = float(np.min(attn_vec))
        top5_indices = [int(i) for i in np.argsort(attn_vec)[::-1][:5]]

        records_protocol_a.append({
            "patient_index": idx,
            "patient_key": pid,
            "num_patches": num_patches,
            "ground_truth_label": gt_label,
            "ground_truth_class": gt_class,
            "fp32_logit": round(logit_val, 6),
            "fp32_probability": round(prob_val, 6),
            "decision_threshold": DECISION_THRESHOLD,
            "predicted_class_id": pred_class_id,
            "predicted_class_name": pred_class_name,
            "classification_outcome": outcome,
            "is_correct": is_correct,
            "confidence_pct": round(confidence_pct, 2),
            "latency_ms": {
                "mean": round(mean_lat, 2),
                "median": round(median_lat, 2),
                "std": round(std_lat, 2),
                "min": round(min_lat, 2),
                "max": round(max_lat, 2)
            },
            "attention_stats": {
                "entropy": round(entropy, 4),
                "max_attention": round(max_attn, 6),
                "min_attention": round(min_attn, 8),
                "top5_patch_indices": top5_indices
            }
        })

    peak_mem_mb = get_process_memory_mb()

    # Aggregate Metrics for Protocol A
    y_true = [r["ground_truth_label"] for r in records_protocol_a]
    y_prob = [r["fp32_probability"] for r in records_protocol_a]
    y_pred = [r["predicted_class_id"] for r in records_protocol_a]

    acc_a = accuracy_score(y_true, y_pred)
    bacc_a = balanced_accuracy_score(y_true, y_pred)
    auc_a = roc_auc_score(y_true, y_prob)
    pr_auc_a = average_precision_score(y_true, y_prob)
    cm_a = confusion_matrix(y_true, y_pred).tolist()
    rep_a = classification_report(y_true, y_pred, target_names=["MSS", "MSI"], output_dict=True)

    print(f"Protocol A Overall Accuracy: {acc_a:.4f} ({acc_a*100:.2f}%)")
    print(f"Protocol A Balanced Accuracy: {bacc_a:.4f}")
    print(f"Protocol A ROC-AUC Score:    {auc_a:.4f}")
    print(f"Protocol A PR-AUC Score:     {pr_auc_a:.4f}")
    print(f"Protocol A Confusion Matrix: {cm_a} (TN: {cm_a[0][0]}, FP: {cm_a[0][1]}, FN: {cm_a[1][0]}, TP: {cm_a[1][1]})")
    print(f"Protocol A MSI Metrics: Precision: {rep_a['MSI']['precision']:.4f}, Recall: {rep_a['MSI']['recall']:.4f}, F1: {rep_a['MSI']['f1-score']:.4f}")
    print(f"Protocol A MSS Metrics: Precision: {rep_a['MSS']['precision']:.4f}, Recall: {rep_a['MSS']['recall']:.4f}, F1: {rep_a['MSS']['f1-score']:.4f}")
    print(f"Mean Slide Latency: {np.mean(latencies_all):.2f} ms (Median: {np.median(latencies_all):.2f} ms)")

    # =========================================================================
    # PROTOCOL B: CAPPED AT 1000 PATCHES WITH EXPLICIT SEED (seed=42)
    # =========================================================================
    print("\n" + "-" * 80)
    print("EXECUTING PROTOCOL B: CAPPED AT 1000 PATCHES (torch.manual_seed(42))")
    print("-" * 80)

    records_protocol_b = []
    torch.manual_seed(42)

    for idx, pid in enumerate(holdout_keys, start=1):
        vt, tt = cached_tensors[pid]

        original_n = vt.shape[0]
        if original_n > 1000:
            perm = torch.randperm(original_n)[:1000]
            vt_eval = vt[perm]
        else:
            vt_eval = vt

        gt_label = registry[pid]["label"]
        with torch.no_grad():
            logit_b, prob_b, _ = compute_logit_and_prob(model, vt_eval, tt)

        pred_b = 1 if prob_b >= DECISION_THRESHOLD else 0
        is_corr_b = (pred_b == gt_label)
        if gt_label == 1 and pred_b == 1:
            out_b = "TP"
        elif gt_label == 0 and pred_b == 0:
            out_b = "TN"
        elif gt_label == 0 and pred_b == 1:
            out_b = "FP"
        else:
            out_b = "FN"

        conf_b = prob_b * 100.0 if prob_b >= DECISION_THRESHOLD else (1.0 - prob_b) * 100.0

        records_protocol_b.append({
            "patient_index": idx,
            "patient_key": pid,
            "original_n": original_n,
            "evaluated_n": vt_eval.shape[0],
            "ground_truth_label": gt_label,
            "ground_truth_class": "MSI" if gt_label == 1 else "MSS",
            "fp32_logit": round(logit_b, 6),
            "fp32_probability": round(prob_b, 6),
            "decision_threshold": DECISION_THRESHOLD,
            "predicted_class_id": pred_b,
            "predicted_class_name": "MSI" if pred_b == 1 else "MSS",
            "classification_outcome": out_b,
            "is_correct": is_corr_b,
            "confidence_pct": round(conf_b, 2)
        })

    y_true_b = [r["ground_truth_label"] for r in records_protocol_b]
    y_prob_b = [r["fp32_probability"] for r in records_protocol_b]
    y_pred_b = [r["predicted_class_id"] for r in records_protocol_b]

    acc_b = accuracy_score(y_true_b, y_pred_b)
    auc_b = roc_auc_score(y_true_b, y_prob_b)
    cm_b = confusion_matrix(y_true_b, y_pred_b).tolist()
    rep_b = classification_report(y_true_b, y_pred_b, target_names=["MSS", "MSI"], output_dict=True)

    print(f"Protocol B Accuracy: {acc_b:.4f} ({acc_b*100:.2f}%)")
    print(f"Protocol B ROC-AUC:  {auc_b:.4f}")
    print(f"Protocol B Confusion Matrix: {cm_b}")
    print(f"Protocol B MSI F1:   {rep_b['MSI']['f1-score']:.4f}, MSS F1: {rep_b['MSS']['f1-score']:.4f}")

    # =========================================================================
    # PROTOCOL C: UNSEEDED JITTER ANALYSIS (MONTE CARLO OVER 10 RUNS)
    # =========================================================================
    print("\n" + "-" * 80)
    print("EXECUTING PROTOCOL C: STOCHASTIC JITTER ANALYSIS (10 UNSEEDED RUNS)")
    print("-" * 80)

    trial_accs = []
    trial_aucs = []
    patient_prob_variance = {pid: [] for pid in holdout_keys}

    for trial in range(10):
        t_probs = []
        t_preds = []
        for pid in holdout_keys:
            vt, tt = cached_tensors[pid]
            if vt.shape[0] > 1000:
                perm = torch.randperm(vt.shape[0])[:1000]
                vt_eval = vt[perm]
            else:
                vt_eval = vt
            with torch.no_grad():
                _, p, _ = compute_logit_and_prob(model, vt_eval, tt)
            t_probs.append(p)
            t_preds.append(1 if p >= DECISION_THRESHOLD else 0)
            patient_prob_variance[pid].append(p)

        trial_acc = accuracy_score(y_true, t_preds)
        trial_auc = roc_auc_score(y_true, t_probs)
        trial_accs.append(trial_acc)
        trial_aucs.append(trial_auc)

    print(f"10-Trial Accuracy Range: [{min(trial_accs):.4f}, {max(trial_accs):.4f}]")
    print(f"10-Trial ROC-AUC Range:  [{min(trial_aucs):.4f}, {max(trial_aucs):.4f}]")

    # Patients that fluctuate
    fluctuating_patients = []
    for pid, p_list in patient_prob_variance.items():
        min_p = min(p_list)
        max_p = max(p_list)
        # Check if boundary 0.83 is crossed
        crosses = (min_p < DECISION_THRESHOLD <= max_p)
        if crosses or (max_p - min_p > 0.02):
            fluctuating_patients.append({
                "patient_key": pid,
                "num_patches": registry[pid]["patch_count"],
                "gt_label": registry[pid]["label"],
                "min_prob": round(min_p, 4),
                "max_prob": round(max_p, 4),
                "spread": round(max_p - min_p, 4),
                "crosses_threshold": crosses
            })

    print(f"Patients with spread > 0.02 or crossing tau=0.83: {len(fluctuating_patients)}")
    for fp in fluctuating_patients:
        print(f"  {fp['patient_key']}: N={fp['num_patches']}, GT={fp['gt_label']}, Spread=[{fp['min_prob']} - {fp['max_prob']}], Crosses={fp['crosses_threshold']}")

    # =========================================================================
    # COMPOSE MACHINE-READABLE BENCHMARK JSON
    # =========================================================================
    benchmark_database = {
        "metadata": {
            "benchmark_phase": "Phase 1 - FP32 Reference Benchmark",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "environment": {
                "os": "Windows 11 (AMD64)",
                "python_version": sys.version.split()[0],
                "pytorch_version": torch.__version__,
                "execution_device": str(device),
                "cpu_threads": num_threads
            },
            "model": {
                "architecture": "SummaryToPatchCrossAttention",
                "checkpoint_file": CHECKPOINT_PATH.name,
                "checkpoint_size_bytes": CHECKPOINT_PATH.stat().st_size,
                "total_parameters": total_params,
                "weight_memory_mb": round(weight_memory_mb, 2)
            },
            "evaluation_cohort": {
                "manifest_file": HOLDOUT_PATH.name,
                "total_patients": len(holdout_keys),
                "class_counts": {"MSS_0": y_true.count(0), "MSI_1": y_true.count(1)},
                "decision_threshold": DECISION_THRESHOLD
            }
        },
        "protocol_provenance_and_relationship": {
            "training_configuration": {
                "script": "src/cross_attentiion_train.py (line 64)",
                "max_patches_configured": 1000,
                "sampling_mechanism": "torch.randperm(num_patches)[:1000] in MultimodalTCGADataset.__getitem__",
                "finding": "The checkpoint fusion_model_v1_1.pth was strictly trained on sequences capped at 1,000 patches."
            },
            "original_evaluation_configuration": {
                "script": "src/cross_attention_eval.py (line 73)",
                "max_patches_configured": 1000,
                "reported_metrics": {"accuracy": 0.8788, "roc_auc": 0.9250},
                "finding": "The published report was generated under unseeded capped execution (an instance of Protocol 3)."
            },
            "protocol_roles": {
                "protocol_1_training_aligned_capped_seed42": "Faithful operational counterpart to original training/eval configuration. Strictly capped at 1,000 patches with fixed seed=42.",
                "protocol_2_full_uncapped_candidate": "Exploratory edge deployment candidate. Leverages model's computational capacity for flexible N to evaluate complete slide evidence without tile dropping.",
                "recommendation": "Preserve Protocol 1 as the training-aligned baseline while benchmarking Protocol 2 as a deployment candidate. Do not silently replace the original protocol."
            }
        },
        "protocols_summary": {
            "protocol_1_training_aligned_capped_seed42": {
                "description": "Capped at 1000 patches using deterministic torch.manual_seed(42), matching training pipeline.",
                "overall_accuracy": round(acc_b, 4),
                "roc_auc": round(auc_b, 4),
                "confusion_matrix": cm_b,
                "classification_report": rep_b
            },
            "protocol_2_full_uncapped_candidate": {
                "description": "Full sequence evaluated without capping or tile dropping (N in [10, 3914]). Candidate deployment mode.",
                "overall_accuracy": round(acc_a, 4),
                "balanced_accuracy": round(bacc_a, 4),
                "roc_auc": round(auc_a, 4),
                "pr_auc": round(pr_auc_a, 4),
                "confusion_matrix": cm_a,
                "classification_report": rep_a,
                "mean_slide_latency_ms": round(float(np.mean(latencies_all)), 2),
                "median_slide_latency_ms": round(float(np.median(latencies_all)), 2)
            },
            "protocol_3_legacy_unseeded_monte_carlo": {
                "description": "10 unseeded stochastic trials matching legacy training DataLoader subsampling.",
                "accuracy_range": [round(min(trial_accs), 4), round(max(trial_accs), 4)],
                "roc_auc_range": [round(min(trial_aucs), 4), round(max(trial_aucs), 4)],
                "fluctuating_patients_count": len(fluctuating_patients)
            }
        },
        "numerical_tolerances_and_determinism_guidelines": {
            "guideline": "Bit-exact determinism cannot be assumed across different runtimes (PyTorch vs ONNX Runtime vs NNAPI). Floating-point non-associativity, SIMD vectorization, and compiler optimizations cause minor numerical divergence.",
            "target_tolerances_for_phase2": {
                "max_absolute_logit_diff": 1e-4,
                "max_absolute_prob_diff": 1e-4,
                "max_relative_error": 1e-3,
                "non_borderline_classification_agreement_pct": 100.0
            },
            "boundary_sensitivity_zone": {
                "threshold": DECISION_THRESHOLD,
                "critical_window": "[0.78, 0.88]",
                "borderline_patients_count": len([r for r in records_protocol_a if 0.78 <= r["fp32_probability"] <= 0.88]),
                "borderline_patients": [
                    {
                        "patient_key": r["patient_key"],
                        "patch_count": r["num_patches"],
                        "ground_truth": r["ground_truth_class"],
                        "probability_uncapped": r["fp32_probability"],
                        "distance_to_threshold": round(abs(r["fp32_probability"] - DECISION_THRESHOLD), 4),
                        "call": r["predicted_class_name"]
                    }
                    for r in records_protocol_a if 0.78 <= r["fp32_probability"] <= 0.88
                ]
            }
        },
        "resource_profiling_framing": {
            "host_development_profile": {
                "context": "x86_64 host developer machine (Windows 11, Intel CPU, 4 PyTorch threads)",
                "measured_mean_latency_ms": round(float(np.mean(latencies_all)), 2),
                "measured_median_latency_ms": round(float(np.median(latencies_all)), 2),
                "baseline_rss_mb": round(base_mem_mb, 2),
                "peak_rss_mb": round(peak_mem_mb, 2),
                "note": "Peak RSS reflects batch preloading of all 66 holdout tensor pairs in memory for multi-trial evaluation."
            },
            "mobile_runtime_target_expectation": {
                "execution_model": "Strictly single-patient sequential inference",
                "model_storage_mb": round(weight_memory_mb, 2),
                "single_patient_input_tensor_mb": "2.05 MB (typical N=500) to 4.10 MB (capped N=1000)",
                "peak_activation_memory_mb": "<25 MB per patient",
                "note": "Host development latency and memory must NOT be directly extrapolated to Android phone performance. Physical on-device profiling is required in later phases."
            }
        },
        "clinical_scope_caveat": {
            "holdout_size": len(holdout_keys),
            "statement": "The 66-case holdout cohort is an isolated academic test partition suitable for technical verification and conversion benchmarking. It is not sufficiently powered by itself to assert generalized clinical diagnostic efficacy."
        },
        "per_patient_reference_data_protocol_1_capped_seed42": records_protocol_b,
        "per_patient_reference_data_protocol_2_uncapped": records_protocol_a
    }

    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark_database, f, indent=2)

    print(f"\n[OK] Successfully saved complete FP32 reference benchmark database to:")
    print(f"   {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    run_benchmark()
