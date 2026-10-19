# Evaluation & Inference Pipeline Audit

**Audit Date:** 2026-10-12  
**Target Scripts:** [`src/cross_attention_eval.py`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/cross_attention_eval.py), [`src/late_fusion_eval.py`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/late_fusion_eval.py)  

---

## 1. Pipeline Execution Flow

Both evaluation pipelines operate directly on the cached tensor store without needing raw image or report inputs:

```
[actual_validation_keys.json] (66 Patient IDs)
          │
          ▼
[MultimodalTCGADataset]
  ├── Loads data/clean_tensors/{id}_vision.pt [N, 1024]
  └── Loads data/clean_tensors/{id}_text.pt   [1, 768]
          │
          ▼
[DataLoader (batch_size=1, shuffle=False)]
          │
          ▼
[SummaryToPatchCrossAttention (fusion_model_v1_1.pth)]
  ├── Text query projects: 768-D → 1024-D
  ├── Multihead Attention: Q=Text, K/V=Vision
  ├── LayerNorm & Residual Connection
  └── Classifier MLP → Sigmoid → prob ∈ [0, 1]
          │
          ▼
[Clinical Decision Threshold (τ = 0.83)]
  ├── prob >= 0.83 → Class 1 (MSI)
  └── prob <  0.83 → Class 0 (MSS)
          │
          ▼
[Scikit-Learn Metrics & Visualizations]
  ├── Accuracy, ROC-AUC, F1-Score, Confusion Matrix
  ├── ROC Curves (results/ROC_Curve_C.png)
  └── Cross-Attention Heatmap (results/Real_Cross_Attention_Matrix.png)
```

---

## 2. Key Operational Findings

1. **Standalone Execution (Zero External Blockers):**
   * The evaluation pipeline does **not** invoke Hugging Face downloads, raw JPEG loaders, or heavy image decoders.
   * Execution on the 66 validation patients takes **~3 seconds total on host CPU**.
2. **Decision Thresholds:**
   * **Cross-Attention:** Threshold is calibrated at **$\tau = 0.83$** (chosen to suppress false-positive MSI classifications and achieve 0.75+ MSI precision).
   * **Late-Fusion:** Threshold is calibrated at **$\tau = 0.70$**.
3. **Reproducibility Vulnerability:**
   * Unseeded `torch.randperm` in `MultimodalTCGADataset.__getitem__` produces minor metric fluctuations for patients with $>1,000$ patches. Adding `torch.manual_seed(42)` makes the evaluation 100% deterministic.
