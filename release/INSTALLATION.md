# CRC Edge AI — Samsung Galaxy S24 Installation & Verification Guide

**Application:** CRC Multimodal Edge (Research)  
**Package ID:** `org.crc.multimodal`  
**Build Variant:** `debug`  
**Target Device:** Samsung Galaxy S24 (SM-S921 / Snapdragon 8 Gen 3 or Exynos 2400 / Android 14 / One UI 6.1)  
**APK Filename:** `crc-edge-ai-android-debug.apk`  
**APK SHA-256:** `22b94247a7fed5bab297fcff28a46f9461b7f93bd74a7620c32f33db22f758a9`  
**Model Bundled:** `fusion_model_fp32.onnx` (SHA-256: `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73`)

---

## 1. Quick Install directly on Samsung Galaxy S24 (Without PC)

1. **Download the APK:**
   - Open the GitHub Release page on the phone's browser (Chrome or Samsung Internet):
     `https://github.com/PrajwalGN1/EdgeCRC-AI/releases/tag/v0.1.0`
   - Download `crc-edge-ai-android-debug.apk`.

2. **Verify File Integrity (Optional but Recommended):**
   - In Samsung **My Files** app or a hash utility, confirm the SHA-256 starts with `22b94247...`.

3. **Allow Installation from Unknown Sources:**
   - Tap on the downloaded APK to open it.
   - If prompted: *"For your security, your phone is not allowed to install unknown apps from this source"*, tap **Settings**.
   - Toggle **Allow from this source** to **ON** for Chrome / Samsung Internet / My Files.
   - Tap **Install**.

4. **Samsung Auto Blocker (One UI 6.1 Security Setting):**
   - If Samsung One UI blocks the installation with a message like *"Installation blocked by Auto Blocker"*:
     1. Open phone **Settings** → **Security and privacy**.
     2. Tap **Auto Blocker**.
     3. Temporarily turn **Auto Blocker** **OFF** (or allow sideloading of debug test apps).
     4. Retry tapping the APK to install.

5. **Launch Application:**
   - Locate **CRC Multimodal Edge (Research)** on the home screen or app drawer.
   - Tap to open. The app initializes ONNX Runtime Mobile (`1.18.0`) in background threads and loads the bundled FP32 Cross-Attention Fusion model.

---

## 2. Developer Installation via USB / ADB (Recommended for Diagnostics)

If connecting the Samsung Galaxy S24 to a development computer via USB:

### A. Enable USB Debugging on Samsung S24
1. Go to **Settings** → **About phone** → **Software information**.
2. Tap **Build number** 7 times rapidly until *"Developer mode has been enabled"* appears.
3. Return to **Settings** → **Developer options** (at the bottom).
4. Turn on **USB debugging**.
5. Connect phone to laptop with a USB-C data cable.
6. When prompted on the phone: *"Allow USB debugging?"*, check *"Always allow from this computer"* and tap **Allow**.

### B. Verify Connection & Install via ADB
Open terminal in the repository root:

```bash
# 1. Verify ADB recognizes the Samsung S24
adb devices -l
# Expected output: <serial_number> device product:e1sxxx model:SM_S921... device:e1s

# 2. Install the APK
adb install -r release/crc-edge-ai-android-debug.apk
# Expected output: Performing Streamed Install -> Success
```

### C. Launch and Capture Real-Time Logcat
```bash
# Launch the main activity
adb shell am start -n org.crc.multimodal/org.crc.multimodal.ui.MainActivity

# Stream inference and model logs
adb logcat -v time -s "CrcMultimodal" "ONNXRuntime" "MainActivity"
```

---

## 3. Important Research Scope & Disclaimers

> [!IMPORTANT]
> - **Debug Test Build:** This APK is compiled with Android debug keys for research validation. It is not an end-user Google Play Store release.
> - **Precomputed Embeddings:** In the current release architecture, the app evaluates the multimodal Cross-Attention Fusion Network (`models/fusion_model_fp32.onnx`) using precomputed patch embeddings ($V \in \mathbb{R}^{N \times 1024}$) and ClinicalBERT text embeddings ($T \in \mathbb{R}^{1 \times 768}$). It does not run full ViT-L/16 patch extraction on-device.
> - **Edge Privacy:** The app contains zero network permissions (`android.permission.INTERNET` is omitted). All computation runs 100% locally on-device.
> - **Validation Gates:** Successful installation and UI rendering confirm APK delivery and native ONNX library loading, but full 66-case holdout numerical parity against PyTorch baseline remains a separate validation milestone.
