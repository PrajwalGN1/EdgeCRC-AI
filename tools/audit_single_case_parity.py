"""
tools/audit_single_case_parity.py
Phase 5.1C: Full-Precision Single-Case Numerical Parity Audit for TCGA-A6-3807.
Compares:
  Path A: Original PyTorch Reference (fusion_model_v1_1.pth)
  Path B: Frozen ONNX Runtime CPU (models/fusion_model_fp32.onnx)
  Path C: Android Ingestion & Packaged ONNX Session (app/src/main/assets/models/fusion_model_fp32.onnx)
"""

import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

REPO_ROOT = Path("D:/CRC Edge deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from architectures.cross_attention_fusion import SummaryToPatchCrossAttention
from eval.export_onnx import CrossAttentionExportWrapper
from tools.export_specimen_bundle import load_upstream_tensors

AUDIT_DIR = REPO_ROOT / "reports" / "mobile_deployment" / "phase5_1_audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.83
ACCEPTANCE_TOLERANCE_MAX = 1.0e-4


def classify(prob: float) -> str:
    return "MSI" if prob >= THRESHOLD else "MSS"


def main():
    case_id = "TCGA-A6-3807"
    tensors_dir = REPO_ROOT / "data" / "clean_tensors"

    # 1. Load authentic source tensors
    v_np, t_np = load_upstream_tensors(case_id, tensors_dir)
    num_patches = v_np.shape[0]

    v_tensor = torch.from_numpy(v_np).unsqueeze(0)  # [1, N, 1024]
    t_tensor = torch.from_numpy(t_np).unsqueeze(0)  # [1, 768]

    v_sha256 = hashlib.sha256(v_np.astype("<f4").tobytes()).hexdigest().upper()
    t_sha256 = hashlib.sha256(t_np.astype("<f4").tobytes()).hexdigest().upper()

    # -------------------------------------------------------------
    # PATH A: PyTorch Reference Implementation
    # -------------------------------------------------------------
    ckpt_path = REPO_ROOT / "models" / "fusion_model_v1_1.pth"
    base_model = SummaryToPatchCrossAttention(vision_dim=1024, text_dim=768, num_heads=8)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    base_model.load_state_dict(state_dict, strict=True)
    base_model.eval()

    wrapper_model = CrossAttentionExportWrapper(base_model, export_attention=True, export_logit=True)
    wrapper_model.eval()

    with torch.no_grad():
        out_logit_A, out_prob_A, out_attn_A = wrapper_model(v_tensor, t_tensor)

    logit_A = float(out_logit_A.item())
    prob_A = float(out_prob_A.item())
    attn_A = out_attn_A.squeeze(0).squeeze(0).cpu().numpy()  # [N]
    class_A = classify(prob_A)

    # -------------------------------------------------------------
    # PATH B: Root Frozen ONNX Model (ONNX Runtime CPU)
    # -------------------------------------------------------------
    root_onnx_path = REPO_ROOT / "models" / "fusion_model_fp32.onnx"
    sess_opts_B = ort.SessionOptions()
    sess_opts_B.intra_op_num_threads = 2
    session_B = ort.InferenceSession(str(root_onnx_path), sess_opts_B, providers=["CPUExecutionProvider"])

    ort_inputs_B = {
        "vision_seq": v_tensor.numpy(),
        "text_summary": t_tensor.numpy()
    }
    ort_outs_B = session_B.run(["logit", "probability", "attention_weights"], ort_inputs_B)

    logit_B = float(ort_outs_B[0].item())
    prob_B = float(ort_outs_B[1].item())
    attn_B = ort_outs_B[2].squeeze()  # [N]
    class_B = classify(prob_B)

    # -------------------------------------------------------------
    # PATH C: Android Packaged ONNX Asset + Imported CRCB Bundle
    # -------------------------------------------------------------
    # We load the exported CRCB v2 bundle file created in Phase 5.1B
    bundle_path = AUDIT_DIR / "roundtrip_test" / f"{case_id}.crcbundle"
    if not bundle_path.exists():
        # fallback to standard bundle path if roundtrip wasn't executed
        bundle_path = REPO_ROOT / "data" / "specimen_bundles" / f"{case_id}.crcbundle"

    with open(bundle_path, "rb") as f:
        magic = f.read(4)
        ver = struct.unpack("<i", f.read(4))[0]
        id_len = struct.unpack("<i", f.read(4))[0]
        f.read(id_len)
        desc_len = struct.unpack("<i", f.read(4))[0]
        f.read(desc_len)
        r_patches, r_vdim, r_tdim, r_dtype = struct.unpack("<iiii", f.read(16))
        v_hash = f.read(32)
        t_hash = f.read(32)
        t_bytes = f.read(r_tdim * 4)
        v_bytes = f.read(r_patches * r_vdim * 4)

    # Decode bytes as FloatArray exactly as AuthenticBundleLoader.kt does
    v_c_arr = np.frombuffer(v_bytes, dtype="<f4").reshape(1, r_patches, r_vdim)
    t_c_arr = np.frombuffer(t_bytes, dtype="<f4").reshape(1, r_tdim)

    # Packaged assets ONNX model
    assets_onnx_path = REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "models" / "fusion_model_fp32.onnx"
    sess_opts_C = ort.SessionOptions()
    sess_opts_C.intra_op_num_threads = 2
    session_C = ort.InferenceSession(str(assets_onnx_path), sess_opts_C, providers=["CPUExecutionProvider"])

    ort_outs_C = session_C.run(
        ["logit", "probability", "attention_weights"],
        {"vision_seq": v_c_arr, "text_summary": t_c_arr}
    )

    logit_C = float(ort_outs_C[0].item())
    prob_C = float(ort_outs_C[1].item())
    attn_C = ort_outs_C[2].squeeze()
    class_C = classify(prob_C)

    # -------------------------------------------------------------
    # ERROR METRICS & COMPARISONS
    # -------------------------------------------------------------
    def compute_metrics(val1, val2):
        abs_err = abs(val1 - val2)
        rel_err = abs_err / (abs(val1) + 1e-12)
        return float(abs_err), float(rel_err)

    # A vs B (PyTorch vs ONNX)
    logit_err_AB, logit_rel_AB = compute_metrics(logit_A, logit_B)
    prob_err_AB, prob_rel_AB = compute_metrics(prob_A, prob_B)
    attn_max_err_AB = float(np.max(np.abs(attn_A - attn_B)))

    # B vs C (Root ONNX vs Android Packaged ONNX Asset)
    logit_err_BC, logit_rel_BC = compute_metrics(logit_B, logit_C)
    prob_err_BC, prob_rel_BC = compute_metrics(prob_B, prob_C)
    attn_max_err_BC = float(np.max(np.abs(attn_B - attn_C)))

    # A vs C (PyTorch Reference vs Android Packaged Inference)
    logit_err_AC, logit_rel_AC = compute_metrics(logit_A, logit_C)
    prob_err_AC, prob_rel_AC = compute_metrics(prob_A, prob_C)
    attn_max_err_AC = float(np.max(np.abs(attn_A - attn_C)))

    # Threshold margin: distance to 0.83
    margin_A = prob_A - THRESHOLD
    margin_B = prob_B - THRESHOLD
    margin_C = prob_C - THRESHOLD

    all_match_class = (class_A == class_B == class_C == "MSS")
    within_tolerance = (prob_err_AC <= ACCEPTANCE_TOLERANCE_MAX) and (logit_err_AC <= ACCEPTANCE_TOLERANCE_MAX)

    results = {
        "audit_phase": "PHASE_5_1C_SINGLE_CASE_NUMERICAL_PARITY",
        "case_id": case_id,
        "patch_count": num_patches,
        "input_tensors": {
            "vision_shape": [1, num_patches, 1024],
            "text_shape": [1, 768],
            "vision_sha256": v_sha256,
            "text_sha256": t_sha256
        },
        "threshold": THRESHOLD,
        "ground_truth_label": "MSS",
        "path_A_pytorch_reference": {
            "checkpoint": str(ckpt_path.relative_to(REPO_ROOT)),
            "logit": logit_A,
            "probability": prob_A,
            "predicted_class": class_A,
            "threshold_margin": margin_A,
            "attention_stats": {
                "mean": float(np.mean(attn_A)),
                "min": float(np.min(attn_A)),
                "max": float(np.max(attn_A)),
                "top5_patch_indices": [int(idx) for idx in np.argsort(-attn_A)[:5]]
            }
        },
        "path_B_onnx_runtime_root": {
            "model_path": str(root_onnx_path.relative_to(REPO_ROOT)),
            "logit": logit_B,
            "probability": prob_B,
            "predicted_class": class_B,
            "threshold_margin": margin_B,
            "attention_stats": {
                "mean": float(np.mean(attn_B)),
                "min": float(np.min(attn_B)),
                "max": float(np.max(attn_B)),
                "top5_patch_indices": [int(idx) for idx in np.argsort(-attn_B)[:5]]
            }
        },
        "path_C_android_packaged_inference": {
            "bundle_source": str(bundle_path.relative_to(REPO_ROOT)),
            "model_asset": str(assets_onnx_path.relative_to(REPO_ROOT)),
            "logit": logit_C,
            "probability": prob_C,
            "predicted_class": class_C,
            "threshold_margin": margin_C,
            "attention_stats": {
                "mean": float(np.mean(attn_C)),
                "min": float(np.min(attn_C)),
                "max": float(np.max(attn_C)),
                "top5_patch_indices": [int(idx) for idx in np.argsort(-attn_C)[:5]]
            }
        },
        "comparisons": {
            "A_vs_B_pytorch_vs_onnx": {
                "logit_abs_error": logit_err_AB,
                "logit_rel_error": logit_rel_AB,
                "prob_abs_error": prob_err_AB,
                "prob_rel_error": prob_rel_AB,
                "attention_max_abs_error": attn_max_err_AB,
                "classification_agreement": class_A == class_B
            },
            "B_vs_C_onnx_vs_android_packaged": {
                "logit_abs_error": logit_err_BC,
                "logit_rel_error": logit_rel_BC,
                "prob_abs_error": prob_err_BC,
                "prob_rel_error": prob_rel_BC,
                "attention_max_abs_error": attn_max_err_BC,
                "classification_agreement": class_B == class_C,
                "bit_for_bit_identical": (logit_err_BC == 0.0 and prob_err_BC == 0.0 and attn_max_err_BC == 0.0)
            },
            "A_vs_C_pytorch_vs_android_packaged": {
                "logit_abs_error": logit_err_AC,
                "logit_rel_error": logit_rel_AC,
                "prob_abs_error": prob_err_AC,
                "prob_rel_error": prob_rel_AC,
                "attention_max_abs_error": attn_max_err_AC,
                "classification_agreement": class_A == class_C
            }
        },
        "parity_audit_verdict": {
            "all_classes_match": all_match_class,
            "prob_error_le_1e4": prob_err_AC <= 1.0e-4,
            "logit_error_le_1e4": logit_err_AC <= 1.0e-4,
            "status": "PASS" if (all_match_class and within_tolerance) else "FAIL"
        }
    }

    out_file = AUDIT_DIR / "single_case_parity_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Single-case parity results saved to: {out_file}")
    print(f"Path A (PyTorch): Logit={logit_A:.6f}, Prob={prob_A:.6f}, Class={class_A}")
    print(f"Path B (ONNX):    Logit={logit_B:.6f}, Prob={prob_B:.6f}, Class={class_B}")
    print(f"Path C (Android): Logit={logit_C:.6f}, Prob={prob_C:.6f}, Class={class_C}")
    print(f"A vs B Prob Error: {prob_err_AB:.3e}")
    print(f"B vs C Bit-Identical: {results['comparisons']['B_vs_C_onnx_vs_android_packaged']['bit_for_bit_identical']}")
    print(f"A vs C Prob Error: {prob_err_AC:.3e}")
    print(f"Status: {results['parity_audit_verdict']['status']}")


if __name__ == "__main__":
    main()
