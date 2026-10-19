# Trained Checkpoint & Model State Audit

**Audit Date:** 2026-10-12  
**Target Directory:** [`models/`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models)  
**Status:** **VERIFIED (ALL 6 CHECKPOINTS INTACT & LOADABLE)**  

---

## 1. Checkpoint Verification Matrix

All six `.pth` checkpoint files were inspected, hashed via SHA-256, and loaded into their native model architectures using PyTorch:

| Checkpoint File | Byte Size | SHA-256 Checksum | Parameters | Stored Dtype | Target Architecture |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`fusion_model_v1_1.pth`**<br>*(★ Primary Model)* | 21,007,347 bytes | `e9a7bbcdbe1713a0ed14b6ecd511ac65d046a52e2f03c69443f5bd3dcf12d453` | **5,250,561** (~5.25 M) | `torch.float32` | `SummaryToPatchCrossAttention` |
| **`best_fusion_model.pth`** | 21,007,347 bytes | `fe408061100cf8d56d5a61fd7a5ede145f83eb31e95d802cf87d1c4275a05c0d` | **5,250,561** (~5.25 M) | `torch.float32` | `SummaryToPatchCrossAttention` |
| **`fusion_model_v1_0.pth`** | 21,007,347 bytes | `7b26b83b096465b67db28c6ae139ef0ce17341a08965b80b222cd835e6202d87` | **5,250,561** (~5.25 M) | `torch.float32` | `SummaryToPatchCrossAttention` |
| **`fusion_model_v1_2.pth`** | 21,007,347 bytes | `0cdf2f6728fdefd1b461629200c0fce9773c1a7ac701d674a777587efec0ce34` | **5,250,561** (~5.25 M) | `torch.float32` | `SummaryToPatchCrossAttention` |
| **`late_fusion_model_v1_0.pth`**<br>*(★ Primary Baseline)*| 5,255,861 bytes | `e98b945354578055aa691f2469390346753523ce7e50315f77098f33d2626ae7` | **1,312,770** (~1.31 M) | `torch.float32` | `MultimodalFusionNetwork` |
| **`late_fusion_model.pth`** | 5,255,717 bytes | `0a73c2eb72dea6d41e2bd76cd78b21014494b69bede47777bf6ca90cdf64851e` | **1,312,770** (~1.31 M) | `torch.float32` | `MultimodalFusionNetwork` |

---

## 2. Checkpoint Identity & Versioning Analysis

> [!IMPORTANT]
> **Checkpoints with Similar Names Are NOT Identical Clones:**
> * `best_fusion_model.pth` is **not** byte-identical to `fusion_model_v1_1.pth` (different SHA-256 hashes).
> * `fusion_model_v1_0.pth`, `v1_1.pth`, and `v1_2.pth` are distinct weights saved at different validation AUC milestones or training runs.
> * `late_fusion_model.pth` differs in both byte length (5,255,717 vs 5,255,861 bytes) and hash from `late_fusion_model_v1_0.pth`.
> * The active evaluation scripts explicitly designate `fusion_model_v1_1.pth` and `late_fusion_model_v1_0.pth` as their production checkpoints.

---

## 3. Structural & Key Breakdown

### Cross-Attention Architecture (`SummaryToPatchCrossAttention`)
* **State Dict Keys (12 tensors):**
  * `text_projection.weight` $[1024, 768]$ (786,432 parameters)
  * `text_projection.bias` $[1024]$ (1,024 parameters)
  * `cross_attn.in_proj_weight` $[3072, 1024]$ (3,145,728 parameters)
  * `cross_attn.in_proj_bias` $[3072]$ (3,072 parameters)
  * `cross_attn.out_proj.weight` $[1024, 1024]$ (1,048,576 parameters)
  * `cross_attn.out_proj.bias` $[1024]$ (1,024 parameters)
  * `layer_norm.weight` $[1024]$, `layer_norm.bias` $[1024]$ (2,048 parameters)
  * `classifier.0.weight` $[256, 1024]$, `classifier.0.bias` $[256]$ (262,400 parameters)
  * `classifier.3.weight` $[1, 256]$, `classifier.3.bias` $[1]$ (257 parameters)
  * **Total Parameters:** **5,250,561**

### Late-Fusion Architecture (`MultimodalFusionNetwork`)
* **State Dict Keys (10 tensors):**
  * `vision_attention.0.weight` $[512, 1024]$, `vision_attention.0.bias` $[512]$ (524,800 parameters)
  * `vision_attention.2.weight` $[1, 512]$, `vision_attention.2.bias` $[1]$ (513 parameters)
  * `text_projector.0.weight` $[512, 768]$, `text_projector.0.bias` $[512]$ (393,728 parameters)
  * `classifier.0.weight` $[256, 1536]$, `classifier.0.bias` $[256]$ (393,472 parameters)
  * `classifier.3.weight` $[1, 256]$, `classifier.3.bias` $[1]$ (257 parameters)
  * **Total Parameters:** **1,312,770**

---

## 4. Inference Smoke Test Verification

A live forward pass smoke test was executed on validation patient **`TCGA-D5-6540`** (using actual precomputed tensors: 345 vision patches @ 1024-D, 1 text summary @ 768-D):

```
Patient ID: TCGA-D5-6540
Vision Tensor Shape: torch.Size([345, 1024]) (torch.float32)
Text Tensor Shape:   torch.Size([1, 768]) (torch.float32)

1. Cross-Attention Model (fusion_model_v1_1.pth):
   • Forward Pass Status: SUCCESS
   • Output Probability: 0.9904 (MSI detected @ threshold >= 0.83)
   • Probability Shape:  torch.Size([1, 1])
   • Attention Shape:    torch.Size([1, 1, 345])

2. Late-Fusion Model (late_fusion_model_v1_0.pth):
   • Forward Pass Status: SUCCESS
   • Output Probability: 0.8677 (MSI detected @ threshold >= 0.70)
   • Probability Shape:  torch.Size([1, 1])
   • Attention Shape:    torch.Size([345, 1])
```

**Conclusion:** Both checkpoints are healthy, load cleanly with zero missing or unexpected keys, and yield mathematically valid predictions and attention distributions.
