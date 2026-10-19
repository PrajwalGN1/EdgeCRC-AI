# Phase 1.5.3 — Boundary Case & Diagnostic Stability Audit

> **Document ID:** `AUDIT-PHASE1.5.3-BOUNDARY-CASES`  
> **Repository:** `MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2`  
> **Audit Status:** **AUDITED & VULNERABILITY-MAPPED**  
> **Boundary Definition:** Operating threshold $\tau = 0.83 \pm 0.05 \implies P \in [0.78, 0.88]$  

---

## 1. Executive Summary

In clinical decision support systems, predictions proximate to the decision boundary ($\tau = 0.83$) carry heightened sensitivity to floating-point perturbations. Even minor numerical divergence between runtime kernels (e.g., PyTorch ATen vs. ONNX Runtime CPU vs. mobile NPUs) can trigger an unintended diagnostic classification flip.

This audit:
1. Performs a complete census of all holdout cases falling within $[0.78, 0.88]$ for both Protocol 1 and Protocol 2.
2. Formally categorizes numerical vs. architectural sources of classification variation.
3. Quantifies the critical drift threshold for each at-risk case.
4. Reinforces that FP32 parity is distinct from and does not guarantee FP16 parity.

---

## 2. Complete Boundary Zone Census ($P \in [0.78, 0.88]$)

An exhaustive scan across all 66 holdout records reveals that the boundary zone consists of exactly **3 cases under Protocol 1** and **4 cases under Protocol 2**:

### Protocol 1 (Capped at 1,000 Patches via Manifest)

| De-identified ID | Patches ($N$) | Ground Truth | Baseline Logit | Baseline Prob ($P$) | Distance to $\tau=0.83$ | Reference Call | Flip Direction Risk |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`CASE_36`** | 80 | **MSI (1)** | +1.586948 | **0.830106** | $\mathbf{+0.000106}$ | **MSI (TP)** | $\Delta P < -0.000106 \implies$ **False Negative (FN)** |
| **`CASE_59`** | 166 | **MSS (0)** | +1.503461 | **0.817998** | $\mathbf{-0.012002}$ | **MSS (TN)** | $\Delta P > +0.012002 \implies$ **False Positive (FP)** |
| **`CASE_35`** | 361 | **MSI (1)** | +1.820067 | **0.860587** | $\mathbf{+0.030587}$ | **MSI (TP)** | $\Delta P < -0.030587 \implies$ **False Negative (FN)** |

### Protocol 2 (Uncapped Candidate, Full Sequence)

| De-identified ID | Patches ($N$) | Ground Truth | Baseline Logit | Baseline Prob ($P$) | Distance to $\tau=0.83$ | Reference Call | Flip Direction Risk |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`CASE_36`** | 80 | **MSI (1)** | +1.586948 | **0.830106** | $\mathbf{+0.000106}$ | **MSI (TP)** | $\Delta P < -0.000106 \implies$ **False Negative (FN)** |
| **`CASE_20`** | 3,914 | **MSS (0)** | +1.663588 | **0.840750** | $\mathbf{+0.010750}$ | **MSI (FP)** | $\Delta P < -0.010750 \implies$ **True Negative (TN)** |
| **`CASE_59`** | 166 | **MSS (0)** | +1.503461 | **0.817998** | $\mathbf{-0.012002}$ | **MSS (TN)** | $\Delta P > +0.012002 \implies$ **False Positive (FP)** |
| **`CASE_35`** | 361 | **MSI (1)** | +1.820067 | **0.860587** | $\mathbf{+0.030587}$ | **MSI (TP)** | $\Delta P < -0.030587 \implies$ **False Negative (FN)** |

---

## 3. Forensic Distinction: Numerical Drift vs. Protocol Differences

To prevent confusion during conversion audits, we establish strict technical definitions distinguishing causes of variation:

```
┌────────────────────────────────────────────────────────────────────────┐
│ TAXONOMY OF MODEL VARIATION                                            │
│                                                                        │
│ 1. Numerical Difference Within Tolerance:                              │
│    • |Δlogit| ≤ 1e-4, |ΔP| ≤ 1e-4.                                     │
│    • Expected floating-point reassociation across engines.             │
│                                                                        │
│ 2. Numerical Difference Exceeding Tolerance:                          │
│    • |ΔP| > 1e-4 on identical input tensors.                           │
│    • Indicates operator bug, incorrect graph fusion, or bad weights.   │
│                                                                        │
│ 3. Threshold Decision Flip (Boundary Instability):                     │
│    • Predicted class flips (e.g., TP -> FN) due to tiny numerical drift│
│      when distance to 0.83 is smaller than runtime epsilon.            │
│                                                                        │
│ 4. Protocol / Input Sequence Length Shift:                             │
│    • Variation caused by evaluating different patch counts (e.g.,      │
│      N=1,000 vs N=3,914). NOT an ONNX conversion error.                │
└────────────────────────────────────────────────────────────────────────┘
```

### Case Study: `CASE_20` Protocol Behavior
* Under Protocol 1 ($N=1,000$ patches via manifest): $P = 0.728124 < 0.83 \implies$ **MSS (True Negative)**. It sits outside the boundary zone ($|0.728124 - 0.83| = 0.1018 > 0.05$).
* Under Protocol 2 ($N=3,914$ full patches): $P = 0.840750 \ge 0.83 \implies$ **MSI (False Positive)**. It enters the boundary zone ($+0.0107$ above $\tau$).
* **Root Cause:** This 0.1126 probability shift is **100% caused by input patch context** ($1000$ vs $3914$ patches entering cross-attention), not by numerical or conversion drift.

### Case Study: `CASE_36` Ultra-Critical Vulnerability
* `CASE_36` has $N=80$ patches (identical inputs in both protocols).
* Probability is $P = 0.830106$, which is only **$+0.000106$ ($+1.06 \times 10^{-4}$)** above the decision threshold $\tau = 0.83$.
* **Risk Assessment:** Any execution environment that produces a negative probability drift of even $-0.000107$ will flip `CASE_36` from a True Positive MSI detection to a False Negative omission.
* **Phase 2 Monitoring Rule:** During ONNX FP32 validation, `CASE_36` will be isolated for bit-level operator tracking. If ONNX Runtime outputs $P \in [0.8299, 0.8301]$, it will be audited for mathematical correctness rather than dismissed as a failure.

---

## 4. Non-Equivalence of FP32 Parity to FP16 Parity

We formally state the following deployment constraint:

> [!CAUTION]
> **FP32 Parity Does Not Imply FP16 Parity:**  
> FP32 operates with a 24-bit significand ($\approx 7.22$ decimal digits; $\epsilon \approx 1.19 \times 10^{-7}$).  
> FP16 operates with an 11-bit significand ($\approx 3.31$ decimal digits; $\epsilon \approx 9.77 \times 10^{-4}$).  
> 
> Because the machine precision of FP16 ($\approx 0.001$) is **nearly 10 times larger than the threshold distance of `CASE_36` ($0.000106$)**, FP16 quantization will inevitably cause significant drift on borderline cases unless specific FP32 fallback layers are retained for sensitive operations (such as softmax/cross-attention accumulation and the sigmoid head).
> 
> FP16 evaluation requires an independent audit, a tailored tolerance contract, and separate user authorization under Phase 3.
