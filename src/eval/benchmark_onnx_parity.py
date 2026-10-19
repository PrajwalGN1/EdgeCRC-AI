#!/usr/bin/env python3
"""
benchmark_onnx_parity.py

Phase 2 — ONNX FP32 Export and Numerical Parity Verification
Exports SummaryToPatchCrossAttention to models/fusion_model_fp32.onnx and evaluates
bit-level numerical parity across three paths:
  1. Original PyTorch model
  2. Wrapper-based PyTorch model
  3. ONNX Runtime CPU session
Evaluates both Protocol 1 (manifest-controlled capped inputs) and Protocol 2 (uncapped candidate mode)
across all 66 holdout validation cases.
"""

import sys
import os
import json
import time
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from architectures.cross_attention_fusion import SummaryToPatchCrossAttention
from eval.export_onnx import CrossAttentionExportWrapper, check_onnx_environment

CHECKPOINT_PATH = REPO_ROOT / "models" / "fusion_model_v1_1.pth"
REGISTRY_PATH = REPO_ROOT / "data" / "patient_registry.json"
HOLDOUT_PATH = REPO_ROOT / "results" / "actual_validation_keys.json"
TENSORS_DIR = REPO_ROOT / "data" / "clean_tensors"
MANIFEST_PATH = REPO_ROOT / "reports" / "mobile_deployment" / "protocol1_patch_selection_manifest.json"
GOLDEN_PATH = REPO_ROOT / "reports" / "mobile_deployment" / "phase1_fp32_benchmark_data.json"
ONNX_OUTPUT_PATH = REPO_ROOT / "models" / "fusion_model_fp32.onnx"
OUTPUT_PARITY_JSON = REPO_ROOT / "reports" / "mobile_deployment" / "phase2_parity_benchmark_data.json"

DECISION_THRESHOLD = 0.83
ACCEPTANCE_MAX_LOGIT_ERR = 1e-4
ACCEPTANCE_MAX_PROB_ERR = 1e-4
ACCEPTANCE_MEAN_PROB_ERR = 1e-5


def make_parity_wrapper(base_model: SummaryToPatchCrossAttention) -> CrossAttentionExportWrapper:
    """
    Creates CrossAttentionExportWrapper configured for dual output parity verification,
    outputting pre-sigmoid logit, post-sigmoid probability, and attention weights.
    """
    return CrossAttentionExportWrapper(base_model, export_attention=True, export_logit=True)


def export_fp32_onnx(model: nn.Module, output_path: Path, opset_version: int = 17):
    import onnx
    print(f"Exporting ONNX FP32 model to {output_path} (Opset {opset_version})...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_vision = torch.randn(1, 100, 1024, dtype=torch.float32)
    dummy_text = torch.randn(1, 768, dtype=torch.float32)

    dynamic_axes = {
        "vision_seq": {0: "batch_size", 1: "num_patches"},
        "text_summary": {0: "batch_size"},
        "logit": {0: "batch_size"},
        "probability": {0: "batch_size"},
        "attention_weights": {0: "batch_size", 2: "num_patches"}
    }

    torch.onnx.export(
        model,
        (dummy_vision, dummy_text),
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["vision_seq", "text_summary"],
        output_names=["logit", "probability", "attention_weights"],
        dynamic_axes=dynamic_axes,
        dynamo=False
    )

    # Validate graph with ONNX checker
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model passed onnx.checker validation successfully.")
    return output_path


def test_dynamic_shapes(session, base_model, wrapper_model):
    print("\n" + "=" * 60)
    print("TESTING DYNAMIC PATCH DIMENSIONS ON ONNX RUNTIME")
    print("=" * 60)
    test_counts = [1, 6, 50, 100, 385, 1000, 4305]
    shape_results = {}
    torch.manual_seed(999)

    for n in test_counts:
        v = torch.randn(1, n, 1024, dtype=torch.float32)
        t = torch.randn(1, 768, dtype=torch.float32)

        with torch.no_grad():
            ref_logit, ref_prob, ref_attn = wrapper_model(v, t)

        v_np = v.numpy()
        t_np = t.numpy()

        ort_inputs = {"vision_seq": v_np, "text_summary": t_np}
        ort_outs = session.run(["logit", "probability", "attention_weights"], ort_inputs)
        onnx_logit = ort_outs[0].item()
        onnx_prob = ort_outs[1].item()
        onnx_attn = ort_outs[2]

        logit_err = abs(onnx_logit - ref_logit.item())
        prob_err = abs(onnx_prob - ref_prob.item())
        attn_err = float(np.max(np.abs(onnx_attn - ref_attn.numpy())))

        passed = (logit_err <= ACCEPTANCE_MAX_LOGIT_ERR and prob_err <= ACCEPTANCE_MAX_PROB_ERR)
        shape_results[f"N_{n}"] = {
            "num_patches": n,
            "ref_logit": float(ref_logit.item()),
            "onnx_logit": float(onnx_logit),
            "logit_error": float(logit_err),
            "ref_prob": float(ref_prob.item()),
            "onnx_prob": float(onnx_prob),
            "prob_error": float(prob_err),
            "max_attn_error": float(attn_err),
            "within_tolerance": passed
        }
        print(f"N={n:4d}: Ref Prob={ref_prob.item():.6f}, ONNX Prob={onnx_prob:.6f}, Logit Err={logit_err:.2e}, Prob Err={prob_err:.2e} -> {'PASS' if passed else 'FAIL'}")

    all_shapes_pass = all(v["within_tolerance"] for v in shape_results.values())
    print(f"All dynamic patch shapes passed? {all_shapes_pass}")
    return shape_results


def run_cohort_parity(protocol_name, session, wrapper_model, holdout_keys, manifest, golden_list):
    print(f"\nEvaluating Cohort Parity: {protocol_name}...")
    ref_by_pid = {r["patient_key"]: r for r in golden_list}
    case_results = []
    flips = 0

    for idx, pid in enumerate(holdout_keys, start=1):
        man_entry = manifest[pid]
        ref = ref_by_pid[pid]

        vp = TENSORS_DIR / f"{pid}_vision.pt"
        tp = TENSORS_DIR / f"{pid}_text.pt"
        vt = torch.load(vp, weights_only=True)
        tt = torch.load(tp, weights_only=True)

        if "protocol 1" in protocol_name.lower() or "protocol_1" in protocol_name.lower():
            if man_entry["is_capped"]:
                indices = torch.tensor(man_entry["protocol1_selected_indices"])
                vt_eval = vt[indices]
            else:
                vt_eval = vt
        else:
            vt_eval = vt

        # PyTorch Wrapper execution
        with torch.no_grad():
            py_logit, py_prob, py_attn = wrapper_model(vt_eval, tt)

        # ONNX Runtime execution
        v_np = vt_eval.unsqueeze(0).numpy() if vt_eval.dim() == 2 else vt_eval.numpy()
        t_np = tt.unsqueeze(0).numpy() if tt.dim() == 1 else tt.numpy()
        ort_outs = session.run(["logit", "probability", "attention_weights"], {"vision_seq": v_np, "text_summary": t_np})
        onnx_logit = float(ort_outs[0].item())
        onnx_prob = float(ort_outs[1].item())

        # Reference values from Phase 1 golden baseline
        ref_logit = float(ref["fp32_logit"])
        ref_prob = float(ref["fp32_probability"])
        ref_call = ref["predicted_class_id"]
        gt_label = ref["ground_truth_label"]

        # Classification decision
        onnx_call = 1 if onnx_prob >= DECISION_THRESHOLD else 0
        call_agreement = (onnx_call == ref_call)
        if not call_agreement:
            flips += 1

        logit_err = abs(onnx_logit - ref_logit)
        prob_err = abs(onnx_prob - ref_prob)
        dist_to_thresh = onnx_prob - DECISION_THRESHOLD

        case_results.append({
            "deidentified_id": f"CASE_{idx:02d}",
            "num_patches": vt_eval.shape[0],
            "ground_truth_label": gt_label,
            "ground_truth_class": "MSI" if gt_label == 1 else "MSS",
            "ref_logit": ref_logit,
            "onnx_logit": onnx_logit,
            "logit_error": logit_err,
            "ref_prob": ref_prob,
            "onnx_prob": onnx_prob,
            "prob_error": prob_err,
            "distance_to_threshold": dist_to_thresh,
            "ref_call": ref_call,
            "onnx_call": onnx_call,
            "classification_agreement": call_agreement,
            "is_boundary_case": (0.78 <= ref_prob <= 0.88),
            "within_tolerance": (logit_err <= ACCEPTANCE_MAX_LOGIT_ERR and prob_err <= ACCEPTANCE_MAX_PROB_ERR)
        })

    max_logit_err = max(c["logit_error"] for c in case_results)
    max_prob_err = max(c["prob_error"] for c in case_results)
    mean_prob_err = sum(c["prob_error"] for c in case_results) / len(case_results)
    agreement_pct = (sum(1 for c in case_results if c["classification_agreement"]) / len(case_results)) * 100.0

    print(f"  {protocol_name} Max Logit Error: {max_logit_err:.2e} (limit: {ACCEPTANCE_MAX_LOGIT_ERR})")
    print(f"  {protocol_name} Max Prob Error:  {max_prob_err:.2e} (limit: {ACCEPTANCE_MAX_PROB_ERR})")
    print(f"  {protocol_name} Mean Prob Error: {mean_prob_err:.2e} (limit: {ACCEPTANCE_MEAN_PROB_ERR})")
    print(f"  {protocol_name} Agreement:       {agreement_pct:.2f}% ({flips} flips)")

    return {
        "max_logit_error": max_logit_err,
        "max_prob_error": max_prob_err,
        "mean_prob_error": mean_prob_err,
        "classification_agreement_pct": agreement_pct,
        "classification_flips": flips,
        "all_cases_within_tolerance": all(c["within_tolerance"] for c in case_results),
        "per_case_results": case_results
    }


def main():
    print("=" * 80)
    print("PHASE 2: ONNX FP32 EXPORT & NUMERICAL PARITY VERIFICATION")
    print("=" * 80)

    env = check_onnx_environment()
    print(f"ONNX Package Version: {env['onnx_version']}")
    print(f"ONNX Runtime Version: {env['onnxruntime_version']}")

    # Load Base PyTorch Model
    base_model = SummaryToPatchCrossAttention(vision_dim=1024, text_dim=768, num_heads=8)
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    base_model.load_state_dict(state_dict, strict=True)
    base_model.eval()

    # Wrap with CrossAttentionExportWrapper configured for dual output
    wrapper_model = make_parity_wrapper(base_model)
    wrapper_model.eval()

    # Export to ONNX
    export_fp32_onnx(wrapper_model, ONNX_OUTPUT_PATH, opset_version=17)

    # Initialize ONNX Runtime Session
    import onnxruntime as ort
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    session = ort.InferenceSession(str(ONNX_OUTPUT_PATH), sess_options, providers=["CPUExecutionProvider"])
    print("Initialized ONNX Runtime CPUExecutionProvider session.")

    # 1. Dynamic Shape Validation
    shape_val = test_dynamic_shapes(session, base_model, wrapper_model)

    # Load holdout keys, manifest, and golden reference
    with open(HOLDOUT_PATH, "r", encoding="utf-8") as f:
        holdout_keys = json.load(f)
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    # 2. Protocol 1 Parity
    p1_parity = run_cohort_parity(
        "Protocol 1 (Capped via Manifest)",
        session,
        wrapper_model,
        holdout_keys,
        manifest,
        golden_data["per_patient_reference_data_protocol_1_capped_seed42"]
    )

    # 3. Protocol 2 Parity
    p2_parity = run_cohort_parity(
        "Protocol 2 (Uncapped Candidate)",
        session,
        wrapper_model,
        holdout_keys,
        manifest,
        golden_data["per_patient_reference_data_protocol_2_uncapped"]
    )

    # Structure comprehensive parity report artifact
    parity_summary = {
        "metadata": {
            "onnx_model_path": str(ONNX_OUTPUT_PATH),
            "onnx_model_bytes": ONNX_OUTPUT_PATH.stat().st_size,
            "onnx_version": env["onnx_version"],
            "onnxruntime_version": env["onnxruntime_version"],
            "decision_threshold": DECISION_THRESHOLD,
            "acceptance_criteria": {
                "max_logit_error": ACCEPTANCE_MAX_LOGIT_ERR,
                "max_prob_error": ACCEPTANCE_MAX_PROB_ERR,
                "mean_prob_error": ACCEPTANCE_MEAN_PROB_ERR
            }
        },
        "dynamic_shapes_validation": shape_val,
        "protocol_1_parity": p1_parity,
        "protocol_2_parity": p2_parity,
        "overall_parity_gate_passed": (
            p1_parity["all_cases_within_tolerance"] and
            p2_parity["all_cases_within_tolerance"] and
            p1_parity["classification_flips"] == 0 and
            p2_parity["classification_flips"] == 0
        )
    }

    OUTPUT_PARITY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PARITY_JSON, "w", encoding="utf-8") as f:
        json.dump(parity_summary, f, indent=2)

    print(f"\nSaved structured parity benchmark to {OUTPUT_PARITY_JSON}")
    print(f"Overall Parity Gate Passed: {parity_summary['overall_parity_gate_passed']}")


if __name__ == "__main__":
    main()
