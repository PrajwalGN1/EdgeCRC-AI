# Phase 2.2 — Environment Verification & ONNX FP32 Export Audit

> **Document ID:** `AUDIT-PHASE2.2-EXPORT`  
> **Repository:** `MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2`  
> **Target Checkpoint:** [`models/fusion_model_v1_1.pth`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_v1_1.pth)  
> **Exported ONNX Model:** [`models/fusion_model_fp32.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp32.onnx)  
> **Audit Status:** **VERIFIED & OPERATIONAL (FULL DYNAMIC SEQUENCE SUPPORT)**  

---

## 1. Executive Summary

Under Phase 2.2 authorization, the project environment was audited, authorized host dependencies (`onnx` and `onnxruntime`) were installed, and the multimodal fusion architecture ([`SummaryToPatchCrossAttention`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/architectures/cross_attention_fusion.py)) was successfully exported to a standalone ONNX FP32 model ([`models/fusion_model_fp32.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp32.onnx)).

Key achievements:
1. **Zero Conflict Installation:** Authorized packages installed into the active Python 3.13.5 environment without altering PyTorch (2.9.1+cpu) or existing dependencies. `python -m pip check` passed cleanly.
2. **Dynamic Sequence Length Preservation:** Solved a critical TorchScript tracer issue in PyTorch's `nn.MultiheadAttention` by isolating key/value tensor object identity during tracing (`val = vision_seq.clone() if torch.jit.is_tracing() else vision_seq`), enabling execution across variable patch sequences from $N=1$ to $N=4,305$ patches without graph mutation.
3. **Graph Integrity & Operator Compatibility:** Validated with `onnx.checker.check_model` (0 errors). All 109 nodes map to 19 standard ONNX operators supported by ONNX Runtime 1.31.0 `CPUExecutionProvider`.
4. **Preserved Output Semantics:** Exposes pre-sigmoid `logit`, post-sigmoid `probability`, and `attention_weights`.

---

## 2. Step 1 & 2: Environment Audit & Dependency Installation

### 2.1 Pre-Installation Environment State
* **Repository Root:** `d:\CRC Edge deployment\MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2`
* **Python Interpreter:** `C:\Python313\python.exe`
* **Python Version:** 3.13.5 (tags/v3.13.5:6559305, Feb  4 2025, 12:44:48) [MSC v.1942 64 bit (AMD64)]
* **Platform:** Windows 11 AMD64 (build 10.0.26100)
* **Virtual Environment:** False (System interpreter, confirmed project environment)
* **Pre-existing Core Packages:** `torch==2.9.1+cpu`, `numpy==2.2.3`, `scipy==1.15.2`, `scikit-learn==1.6.1`, `pip==25.0.1`
* **Git Status:** Clean working tree except for pre-existing UI/report modifications (`src/ui/app.py`, `results/`).

### 2.2 Authorized Package Installation
Executed on active interpreter:
```powershell
python -m pip install onnx onnxruntime
```
* **Installed Packages:**
  * `onnx`: **1.23.2**
  * `onnxruntime`: **1.31.0**
* **Dependency Check:**
  ```powershell
  python -m pip check
  # Output: No broken requirements found.
  # Exit Code: 0
  ```
* **Protected Asset Integrity Verification:**
  All protected asset SHA-256 fingerprints were re-computed immediately following package installation and verified 100% bit-intact:
  * Checkpoint (`models/fusion_model_v1_1.pth`): `e9a7bbcdbe1713a0ed14b6ecd511ac65d046a52e2f03c69443f5bd3dcf12d453`
  * Registry (`data/patient_registry.json`): `7d1c92c51741e080787607733a8b34cb07252c804f527cd567937345a5b22ce2`
  * Holdout Manifest (`results/actual_validation_keys.json`): `61d7f8d98388efbe9fc5e000d695b2f41022720a14a8308a6317bc9441930aa9`
  * Patch Manifest (`protocol1_patch_selection_manifest.json`): `e592f3e218169741fceae27deb75e305cf7549739cfd1fbdbb0087b05863e668`

---

## 3. Step 3: ONNX FP32 Export Architecture & Graph Verification

### 3.1 Export Configuration
* **Export Script:** [`src/eval/benchmark_onnx_parity.py`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/eval/benchmark_onnx_parity.py) using [`CrossAttentionExportWrapper`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/eval/export_onnx.py#L21-L107)
* **Opset Version:** 17
* **Exporter Mode:** TorchScript JIT exporter (`dynamo=False`)
* **Constant Folding:** Enabled (`do_constant_folding=True`)
* **Output Path:** [`models/fusion_model_fp32.onnx`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/models/fusion_model_fp32.onnx)
* **File Size:** 21,015,452 bytes (~20.04 MiB)
* **Model SHA-256:** `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73`

### 3.2 Dynamic Sequence Tracer Resolution
During initial dynamic shape export, PyTorch's `torch.nn.MultiheadAttention` exhibited a known tracer artifact: when `query` has sequence length 1 (`text_query` `[1, 1, 1024]`) and `key` and `value` share object identity (`key=vision_seq, value=vision_seq`), PyTorch's internal `multi_head_attention_forward` evaluates `if key is value:` as `True`. In TorchScript tracing, this collapsed the key/value reshape operations into static dimensions matching the dummy input sequence ($N=100$), causing runtime shape mismatches on variable sequences ($N \ne 100$).

**Resolution:** In [`CrossAttentionExportWrapper`](file:///d:/CRC%20Edge%20deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2/src/eval/export_onnx.py#L79-L84):
```python
val = vision_seq.clone() if torch.jit.is_tracing() else vision_seq
attn_output, attn_weights = self.base_model.cross_attn(
    query=text_query,
    key=vision_seq,
    value=val
)
```
* During standard PyTorch evaluation (`is_tracing() == False`), `val` is `vision_seq`, preserving 100% bit-exact output identity with the original model.
* During ONNX export (`is_tracing() == True`), `val` is a cloned tensor, breaking object identity (`key is value` evaluates to `False`). PyTorch traces dynamic shape-dependent reshapes (`Concat_3` / `Reshape_4`), generating a fully dynamic graph.

### 3.3 Graph Topology & Operator Inventory
The exported graph contains exactly 109 nodes and 16 initializers (5,250,561 float32 parameters):

| Operator Type | Count | Standard ONNX Support | Description |
| :--- | :---: | :---: | :--- |
| `Constant` | 30 | Opset 17 | Static weights, bias offsets, and scalar reshape constants |
| `Unsqueeze` | 18 | Opset 17 | Dimension alignment for broadcasting and attention heads |
| `Transpose` | 9 | Opset 17 | Multi-head attention permutations (`[B, N, H, D] -> [B, H, N, D]`) |
| `Shape` | 7 | Opset 17 | Dynamic sequence length and batch dimension extraction |
| `Gather` | 7 | Opset 17 | Indexing dimension sizes for dynamic reshaping |
| `MatMul` | 6 | Opset 17 | Scaled dot-product attention (`Q @ K^T` and `Attn @ V`) |
| `Concat` | 6 | Opset 17 | Dynamic shape vector construction |
| `Reshape` | 6 | Opset 17 | Attention head flattening and sequence reshaping |
| `Add` | 5 | Opset 17 | Bias addition and residual connection (`text_query + attn_output`) |
| `Mul` | 3 | Opset 17 | Attention scaling factor ($1/\sqrt{d_k}$) |
| `Gemm` | 3 | Opset 17 | Linear projections (`text_projection`, classifier MLP layers) |
| `Cast` | 2 | Opset 17 | Integer shape index casting |
| `Div` | 1 | Opset 17 | Layer normalization variance division |
| `Softmax` | 1 | Opset 17 | Attention weight distribution across patch dimension |
| `ReduceMean` | 1 | Opset 17 | Layer normalization mean calculation |
| `LayerNormalization` | 1 | Opset 17 | LayerNorm after residual fusion |
| `Squeeze` | 1 | Opset 17 | Pooling sequence dimension before classification MLP |
| `Relu` | 1 | Opset 17 | Activation function in classifier MLP |
| `Sigmoid` | 1 | Opset 17 | Probability squashing for MSI prediction |
| **Total Nodes** | **109** | **Opset 17** | **100% Supported by ONNX Runtime CPUExecutionProvider** |

### 3.4 Tensor I/O Contract Verification

| Name | Direction | Element Type | Formal Shape | Dynamic Axes |
| :--- | :---: | :---: | :--- | :--- |
| `vision_seq` | Input | `FLOAT` (float32) | `['batch_size', 'num_patches', 1024]` | Dim 0 (`batch_size`), Dim 1 (`num_patches`) |
| `text_summary` | Input | `FLOAT` (float32) | `['batch_size', 768]` | Dim 0 (`batch_size`) |
| `logit` | Output | `FLOAT` (float32) | `['batch_size', 1]` | Dim 0 (`batch_size`) |
| `probability` | Output | `FLOAT` (float32) | `['batch_size', 1]` | Dim 0 (`batch_size`) |
| `attention_weights` | Output | `FLOAT` (float32) | `['batch_size', 1, 'num_patches']` | Dim 0 (`batch_size`), Dim 2 (`num_patches`) |

### 3.5 Graph Checker & Dynamic Dimension Validation
1. **ONNX Checker:** `onnx.checker.check_model(onnx_model)` executed without errors or warnings.
2. **ONNX Runtime Session:** Initialized successfully with `CPUExecutionProvider` and basic graph optimizations.
3. **Dynamic Patch Length Verification:** Evaluated across representative sequence lengths:

| Patch Count ($N$) | Context | PyTorch Ref Prob | ONNX Prob | Max Prob Error | Max Logit Error | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **$N = 1$** | Single isolated patch | 0.923603 | 0.923603 | $5.96 \times 10^{-8}$ | $4.77 \times 10^{-7}$ | **PASS** |
| **$N = 6$** | Small sequence | 0.869295 | 0.869295 | $0.00 \times 10^0$ | $0.00 \times 10^0$ | **PASS** |
| **$N = 50$** | Low tissue density | 0.392143 | 0.392143 | $2.98 \times 10^{-8}$ | $1.49 \times 10^{-7}$ | **PASS** |
| **$N = 100$** | Standard biopsy | 0.542573 | 0.542573 | $0.00 \times 10^0$ | $8.94 \times 10^{-8}$ | **PASS** |
| **$N = 385$** | Cohort median benchmark | 0.471722 | 0.471722 | $0.00 \times 10^0$ | $6.71 \times 10^{-8}$ | **PASS** |
| **$N = 1,000$** | Protocol 1 cap limit | 0.653745 | 0.653745 | $0.00 \times 10^0$ | $5.96 \times 10^{-8}$ | **PASS** |
| **$N = 4,305$** | Maximum sequence bound | 0.605037 | 0.605037 | $0.00 \times 10^0$ | $0.00 \times 10^0$ | **PASS** |

---

## 4. Audit Sign-off

The ONNX FP32 export satisfies all structural, topological, operator, and dynamic sequence requirements. Environment checks and asset hash verifications pass without deviation.
