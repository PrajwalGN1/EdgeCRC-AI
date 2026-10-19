# Environment and System Inventory

**Audit Date:** 2026-10-12  
**Auditor Role:** Senior ML Engineer, Multimodal AI Specialist, Edge AI Deployment Engineer  
**Repository:** `MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2`  
**Active Branch:** `edge-deployment`  

---

## 1. Host Operating System & Execution Environment

| Property | Measured Value | Notes & Implications |
| :--- | :--- | :--- |
| **Repository Root** | `d:\CRC Edge deployment\MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2` | Verified absolute filesystem path. |
| **Operating System** | Windows 11 Pro (`10.0.26200-SP0`, AMD64) | Host execution is Windows-based; codebase contains path conversions for POSIX/Windows cross-platform compatibility. |
| **Python Version** | `3.13.5` (64-bit) | Executable at `C:\Python313\python.exe`. Fully operational. |
| **Virtual Environment** | None active (System Python) | Packages are installed in user/system site-packages (`AppData\Roaming\Python\Python313\site-packages`). |
| **PyTorch Version** | `2.9.1+cpu` | Official PyTorch CPU build. FP32 forward passes functional. |
| **CUDA Available** | `False` (0 GPUs detected) | No NVIDIA CUDA runtime driver detected on active Python interpreter. |
| **Apple MPS Available**| `False` | Running on x86_64 Windows host. |
| **Hardware Fallback** | Operating on Host CPU | Model inference scripts correctly fallback to `device = 'cpu'`. |

---

## 2. Python Dependency Audit

| Package | Installed Version | Manifest (`requirements.txt`) Status | Verification Notes |
| :--- | :--- | :--- | :--- |
| `torch` | `2.9.1+cpu` | Listed (`torch==2.9.1`) | Operational on CPU. |
| `transformers` | `4.38.2` | **Omitted from requirements.txt** | Installed in user site-packages; required for Phikon-v2/ClinicalBERT tokenization/configs. |
| `scikit-learn` | `1.7.0` | **Omitted from requirements.txt** | Installed in user site-packages; required for ROC-AUC, confusion matrix, classification reports. |
| `safetensors` | `0.7.0` | Listed (`safetensors==0.7.0`) | Verified. |
| `numpy` | `2.2.6` | Listed (`numpy==2.2.6`) | Verified. |
| `matplotlib` | `3.10.3` | **Omitted from requirements.txt** | Installed in user site-packages; required for plot generation. |
| `seaborn` | `0.13.2` | **Omitted from requirements.txt** | Installed in user site-packages; required for attention heatmaps. |
| `streamlit` | `1.52.2` | Listed (`streamlit==1.52.2`) | UI server framework operational. |

> [!WARNING]
> While all required libraries are functional in the developer's local environment, `requirements.txt` is missing critical runtime dependencies (`transformers`, `scikit-learn`, `matplotlib`, `seaborn`). A clean deployment inside Docker or CI/CD will fail without these packages.

---

## 3. Git Repository Status

* **Current Branch:** `edge-deployment`
* **Synchronized Base:** `origin/main` @ commit `189b6638c31d80344ecc8c45133c7058c3e47fa1` (`chore: remove roadmap/next-team and authors, keep mentor Mr Nageshwara Guptha — professional public version`)
* **Working Tree State:**
  * Modified tracked files:
    * `results/Confusion_Matrix_C.png`
    * `results/Cross_Attention_Model_Evaluation_Report.txt`
    * `results/ROC_Curve_C.png`
    * `results/Real_Cross_Attention_Matrix.png`
  * Untracked project assets:
    * `PROJECT_HANDOFF_AUDIT.md`
    * `REPOSITORY_FILE_MAP.md`
    * `results/computational_power_fp32.json`
    * `results/computational_power_fp32.txt`
    * `src/eval/benchmark_computational_power.py`
* **Git LFS:** Not configured. Checkpoint files (`models/*.pth`, ~21 MB each) are tracked directly in Git history rather than Git LFS.

---

## 4. Documentation & Test Infrastructure

* **Architectural Documentation:**
  * `docs/ARCHITECTURE.md` (Thorough, covers 4-phase pipeline, DFDs, mathematical formulation of Cross-Attention).
  * `docs/HANDOFF.md` (Operational instructions).
  * `PROJECT_HANDOFF_AUDIT.md` (Comprehensive historical audit from 2026-10-03).
* **Test Suite:**
  * No standardized test runner (`pytest` or `unittest`) is configured.
  * Testing relies on standalone procedural evaluation scripts (`src/cross_attention_eval.py`, `src/late_fusion_eval.py`, `src/eval/verify_cross_attention.py`).
