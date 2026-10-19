# Ordered Next Steps & Implementation Action Plan

**Audit Date:** 2026-10-12  
**Target:** Engineering Execution on `edge-deployment` Branch  

---

## 1. Immediate Non-Destructive Actions (No Retraining Required)

1. **Step 1 — Formalize Evaluation Seed for Determinism:**
   * In [cross_attention_eval.py](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/cross_attention_eval.py#L96), insert `torch.manual_seed(42)` before DataLoader evaluation. This locks the random sampling cap of $N \le 1,000$ patches and guarantees 100% deterministic, reproducible accuracy and ROC-AUC scores across all future runs.
2. **Step 2 — Lock the Training Patient Split Contract:**
   * Generate `results/actual_train_keys.json` containing the 260 non-validation patient IDs (`326 total - 66 validation`) to establish an immutable training contract alongside `actual_validation_keys.json`.
3. **Step 3 — Run the FP16 Precision Experiment on the Fusion Model:**
   * Create an FP16 evaluation script `src/eval/benchmark_fp16_fusion.py` to evaluate `fusion_model_v1_1.pth` in FP16 precision across the 66 validation patients.
   * Verify whether the predictions match the FP32 baseline within $\epsilon < 1e-4$ and confirm zero drop in ROC-AUC.
4. **Step 4 — Export Fusion Head to ONNX Format:**
   * Export `SummaryToPatchCrossAttention` to ONNX (`models/fusion_model_fp32.onnx` and `models/fusion_model_fp16.onnx`) with dynamic sequence length axes for $N$ patches.
   * Validate with ONNX Runtime to establish mobile-ready artifacts for Android.
5. **Step 5 — Submit Senior Requirements Request:**
   * Submit [`reports/audit/items_to_request_from_senior.md`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/reports/audit/items_to_request_from_senior.md) to clarify the checkpoint versioning and obtain the raw feature extraction script.
