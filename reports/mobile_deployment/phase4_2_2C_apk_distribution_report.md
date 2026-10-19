# Phase 4.2.2C: Android Edge AI APK Distribution and Release Audit Report

**Repository:** `D:\CRC Edge deployment\MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2`  
**GitHub Remote:** `https://github.com/PrajwalGN1/EdgeCRC-AI.git`  
**Repository Visibility:** `PUBLIC`  
**Target Branch:** `edge-deployment`  
**Source Commit:** `19b27f49c168a34275b51e03a000301e2d46576d`  
**Audit Date:** 2026-10-19  
**Audit Timezone:** Asia/Kolkata (IST, UTC+05:30)  
**Auditor Roles:** Senior Android Release Engineer, GitHub Release Automation Engineer, Android Edge-AI Deployment Engineer, and Software Integrity Auditor  
**Disposition:** `GO (RELEASE PUBLISHED & AUDITED)`

---

## 1. Executive Summary

This report establishes the forensic audit, packaging, and distribution release of the **Multimodal CRC Edge AI** native Android application (`org.crc.multimodal`). The release provides a cryptographically verified, standalone Android APK bundling the frozen 5.25M-parameter Cross-Attention Fusion neural network (`fusion_model_fp32.onnx`) for direct installation on physical mobile hardware, specifically targeted for runtime validation on the **Samsung Galaxy S24** (Snapdragon 8 Gen 3 / Exynos 2400, Android 14 / One UI 6.1).

The release has been published directly to the public GitHub repository releases system, accompanied by exact cryptographic manifests, Samsung Galaxy S24 installation guides, and explicit non-falsification boundaries.

---

## 2. Release & Download Coordinates

| Coordinate | Value / URL | Status |
| :--- | :--- | :--- |
| **Release Tag** | `v0.1.0` | `PUBLISHED` |
| **Release Title** | `CRC Edge AI Android Test Build (v0.1.0)` | `PUBLISHED` |
| **Release Webpage URL** | [https://github.com/PrajwalGN1/EdgeCRC-AI/releases/tag/v0.1.0](https://github.com/PrajwalGN1/EdgeCRC-AI/releases/tag/v0.1.0) | `VERIFIED (PUBLIC)` |
| **Direct Asset Download URL** | [https://github.com/PrajwalGN1/EdgeCRC-AI/releases/download/v0.1.0/crc-edge-ai-android-debug.apk](https://github.com/PrajwalGN1/EdgeCRC-AI/releases/download/v0.1.0/crc-edge-ai-android-debug.apk) | `VERIFIED (PUBLIC)` |
| **Access Requirement** | None (Fully Public — no GitHub account or login required) | `VERIFIED` |
| **Asset Filename** | `crc-edge-ai-android-debug.apk` | `VERIFIED` |
| **Asset Size** | `72,731,704` bytes (69.36 MB) | `VERIFIED` |
| **Release Manifest File** | `release/release_manifest.json` | `VERIFIED` |
| **Installation Guide** | `release/INSTALLATION.md` | `VERIFIED` |

---

## 3. Cryptographic & Binary Audit Table

| Artifact | Filesystem Path | Size (Bytes) | SHA-256 Digest | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Original Built APK** | `android/app/build/outputs/apk/debug/app-debug.apk` | 72,731,704 | `22b94247a7fed5bab297fcff28a46f9461b7f93bd74a7620c32f33db22f758a9` | `VERIFIED` |
| **Staged Release APK** | `release/crc-edge-ai-android-debug.apk` | 72,731,704 | `22b94247a7fed5bab297fcff28a46f9461b7f93bd74a7620c32f33db22f758a9` | `BIT_FOR_BIT_IDENTICAL` |
| **Source FP32 ONNX Model** | `models/fusion_model_fp32.onnx` | 21,015,452 | `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` | `VERIFIED_FROZEN` |
| **Asset Model Copy** | `android/app/src/main/assets/models/fusion_model_fp32.onnx` | 21,015,452 | `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` | `BIT_FOR_BIT_IDENTICAL` |
| **Model inside APK** | `assets/models/fusion_model_fp32.onnx` (in APK archive) | 21,015,452 | `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` | `BIT_FOR_BIT_IDENTICAL` |

---

## 4. APK Manifest & Runtime Configuration

- **Application ID:** `org.crc.multimodal`
- **Application Label:** `CRC Multimodal Edge (Research)`
- **Version Code:** `1`
- **Version Name:** `1.0.0`
- **Min SDK:** `24` (Android 7.0 Nougat+)
- **Target SDK:** `34` (Android 14)
- **Compile SDK:** `34`
- **Main Activity:** `org.crc.multimodal.ui.MainActivity`
- **Native Architectures (ABIs):**
  - `arm64-v8a` (Native 64-bit ARMv9 target for Samsung Galaxy S24)
  - `armeabi-v7a` (32-bit fallback)
  - `x86_64` (Desktop/Emulator architecture)
- **Native Runtime Library:** `libonnxruntime.so` & `libonnxruntime4j_jni.so` (ONNX Runtime Mobile v1.18.0)
- **Signature Scheme:** APK Signature Scheme v2/v3 present (`APK Sig Block 42` verified)
- **Permissions Audit:**
  - `org.crc.multimodal.DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION`
  - **Zero Network Permissions:** `android.permission.INTERNET` is explicitly **omitted**.
  - **Offline Privacy Guarantee:** Verified 100% on-device local computation with zero external network transmission capability.

---

## 5. Security & Sensitive Data Audit

Prior to release packaging and publication, a complete sensitive-artifact scan was executed across the working tree:
1. **Credentials & Key Files:** Zero private keystores, `.pem`, `.key`, `.p12`, or `.env` files present in the release package.
2. **Clinical Patient Registry & Raw Data:** Zero unmasked patient identifiers, raw TCGA pathology reports, or raw whole-slide images were included in the release artifact. The APK contains only the neural model weights and application binary logic.
3. **Repository State:** Working tree clean (`19b27f49c168a34275b51e03a000301e2d46576d`).

---

## 6. Samsung Galaxy S24 Installation Instructions

For real-device runtime validation on a Samsung Galaxy S24:

### Direct Installation on Samsung Galaxy S24 (No PC Required)
1. Open the shareable release link in Samsung Internet or Chrome:  
   `https://github.com/PrajwalGN1/EdgeCRC-AI/releases/tag/v0.1.0`
2. Tap `crc-edge-ai-android-debug.apk` to download.
3. When downloaded, tap to open.
4. If Samsung One UI prompts that unknown apps are blocked:
   - Tap **Settings** → toggle **Allow from this source** to **ON**.
   - If Samsung **Auto Blocker** prevents sideloading (One UI 6.1):
     - Go to phone **Settings** → **Security and privacy** → **Auto Blocker**.
     - Temporarily toggle **Auto Blocker** to **OFF**.
5. Tap **Install** and launch **CRC Multimodal Edge (Research)** from the app drawer.

### Developer Installation via ADB (Command Line)
1. Enable **Developer options** on S24 (tap **Build number** 7 times under **About phone** → **Software information**).
2. Enable **USB debugging**.
3. Connect the phone to laptop via USB-C.
4. Execute:
   ```powershell
   adb devices -l
   adb install -r release/crc-edge-ai-android-debug.apk
   adb shell am start -n org.crc.multimodal/org.crc.multimodal.ui.MainActivity
   ```

---

## 7. Known Scope & Non-Falsification Boundaries

> [!WARNING]
> In accordance with the project invariants and engineering integrity principles:
> 1. **Distribution Does NOT Prove Runtime Parity:** Distributing and installing the APK confirms package delivery, asset unzipping, and Android runtime initialization. It does **not** substitute for running the full 66-case holdout evaluation on the physical phone.
> 2. **Precomputed Embeddings:** The application executes the Cross-Attention Fusion layer (`fusion_model_fp32.onnx`) taking precomputed patch embeddings ($V \in \mathbb{R}^{N \times 1024}$) and ClinicalBERT text CLS embedding ($T \in \mathbb{R}^{1 \times 768}$). ViT patch feature extraction is not performed on-device.
> 3. **Debug Build Status:** This APK is signed with the Android SDK debug certificate for testing. Android OS will display a security warning regarding an untrusted developer certificate during initial sideloading.
> 4. **Hardware Validation Gate:** Physical on-device inference logging on the Samsung S24 (capturing device execution latency, thermal behavior, and numerical output parity) remains the final active engineering gate.
