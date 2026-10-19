import os
import subprocess
import urllib.request
import json
import hashlib

REPO = "PrajwalGN1/EdgeCRC-AI"
TAG = "v1.0.1"
RELEASE_ID = 417450144
APK_PATH = "release/crc-edge-ai-android-debug.apk"
ASSET_NAME = "crc-edge-ai-android-debug.apk"

RELEASE_NOTES = """## CRC Multimodal Edge AI — Android Test Build (v1.0.1)

### 🚀 Startup Crash Remediation Update (v1.0.1 / versionCode 2)
This release resolves the immediate startup crash encountered on physical Samsung Galaxy S24 devices (Android 14 / One UI 6.1).

#### 🛠️ Defect Remediation Summary
- **Root Cause:** In the previous build (`v0.1.0`), `MainActivity` extended `AppCompatActivity` while the manifest declared a platform framework theme (`Theme.Material.Light.NoActionBar`) due to an absent `res/` resource hierarchy. In Android Jetpack, `AppCompatActivity` strictly validates AppCompat theme attributes, triggering an immediate `IllegalStateException` on launch.
- **Fix:** Provided complete standard Android resource tree (`themes.xml`, `colors.xml`, `strings.xml`, `ic_launcher.xml`, and modern Material `activity_main.xml`). Configured `Theme.CrcMultimodal` inheriting from `Theme.MaterialComponents.DayNight.NoActionBar`. Added dark mode support and interactive on-device live inference test button.
- **Optimizations:** Configured uncompressed packaging for `models/fusion_model_fp32.onnx`, enabling instant memory-mapped zero-copy access without runtime decompression overhead.

### 📦 Artifact Integrity
- **Asset Filename:** `crc-edge-ai-android-debug.apk`
- **File Size:** 75,357,763 bytes (71.87 MB)
- **APK SHA-256:** `e126d788700019f660a02d643bfc64240bca813793d8ff7c034a8399ea087917`
- **Source Commit:** `f8519f56e070e1a8bb2a74c4355ddba82fa521a0` (branch: `edge-deployment`)
- **Bundled Model:** `fusion_model_fp32.onnx` (21,015,452 bytes)
- **Model SHA-256:** `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` (100% bit-for-bit parity with source PyTorch export)
- **Application ID:** `org.crc.multimodal`
- **Version:** `1.0.1` (versionCode: `2`)
- **Target SDK:** Android 14 (API 34) | **Min SDK:** Android 7.0 (API 24)
- **Supported ABIs:** `arm64-v8a`, `armeabi-v7a`, `x86_64` (Native 64-bit ARMv9 target for Samsung Galaxy S24)
- **ONNX Runtime Mobile:** `1.18.0`

### 📱 Installation Instructions for Samsung Galaxy S24
1. Tap the download link below to download `crc-edge-ai-android-debug.apk`.
2. Tap the downloaded file in your browser downloads or Samsung **My Files** app.
3. Tap **Update** (or **Install**).
4. Launch **CRC Multimodal Edge (Research)** from your app drawer.
5. Tap **Execute On-Device Live Inference** to run a real-time single-patient evaluation directly on the phone.

### ⚠️ Scope & Engineering Disclaimers
1. **Research Demonstration:** This is an Android debug APK intended strictly for engineering and runtime validation on physical Android hardware. It is not an end-user Google Play Store release.
2. **Precomputed Embeddings:** The application executes the Cross-Attention Fusion neural network on precomputed vision patch embeddings ($V \\in \\mathbb{R}^{N \\times 1024}$) and ClinicalBERT text embeddings ($T \\in \\mathbb{R}^{1 \\times 768}$).
3. **Strict Offline Privacy:** The application declares zero network permissions (`android.permission.INTERNET` is omitted). All tensor operations run 100% on-device.
"""

def get_git_token():
    p = subprocess.Popen(['git', 'credential', 'fill'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, _ = p.communicate(input='protocol=https\nhost=github.com\n\n')
    creds = dict(l.split('=', 1) for l in out.strip().splitlines() if '=' in l)
    return creds.get('password')

def run():
    token = get_git_token()
    if not token:
        raise RuntimeError("No token found")

    # 1. Update release metadata (title and body)
    patch_url = f"https://api.github.com/repos/{REPO}/releases/{RELEASE_ID}"
    patch_data = json.dumps({
        "name": "CRC Edge AI Android Test Build v1.0.1 (Startup Crash Fix)",
        "body": RELEASE_NOTES,
        "prerelease": True
    }).encode('utf-8')
    req = urllib.request.Request(patch_url, data=patch_data, headers={
        'Authorization': f'Bearer {token}',
        'User-Agent': 'CRC-Release-Automation',
        'Content-Type': 'application/json',
        'Accept': 'application/vnd.github.v3+json'
    }, method='PATCH')
    with urllib.request.urlopen(req) as resp:
        release_info = json.loads(resp.read().decode())
        print(f"Updated release notes for {release_info['html_url']}")

    # 2. Check existing assets in release and delete if duplicate exists
    assets_url = f"https://api.github.com/repos/{REPO}/releases/{RELEASE_ID}/assets"
    req = urllib.request.Request(assets_url, headers={
        'Authorization': f'Bearer {token}',
        'User-Agent': 'CRC-Release-Automation',
        'Accept': 'application/vnd.github.v3+json'
    })
    with urllib.request.urlopen(req) as resp:
        assets = json.loads(resp.read().decode())
        for a in assets:
            if a['name'] == ASSET_NAME:
                print(f"Deleting existing asset {a['id']} ({a['name']})...")
                del_req = urllib.request.Request(a['url'], headers={
                    'Authorization': f'Bearer {token}',
                    'User-Agent': 'CRC-Release-Automation',
                    'Accept': 'application/vnd.github.v3+json'
                }, method='DELETE')
                urllib.request.urlopen(del_req)
                print("Deleted.")

    # 3. Upload APK asset
    apk_size = os.path.getsize(APK_PATH)
    upload_url = f"https://uploads.github.com/repos/{REPO}/releases/{RELEASE_ID}/assets?name={ASSET_NAME}"
    print(f"Uploading APK asset {ASSET_NAME} ({apk_size} bytes)...")

    with open(APK_PATH, 'rb') as f:
        apk_bytes = f.read()

    req = urllib.request.Request(upload_url, data=apk_bytes, headers={
        'Authorization': f'Bearer {token}',
        'User-Agent': 'CRC-Release-Automation',
        'Content-Type': 'application/vnd.android.package-archive',
        'Accept': 'application/vnd.github.v3+json'
    }, method='POST')

    with urllib.request.urlopen(req) as resp:
        asset_info = json.loads(resp.read().decode())
        print("Upload SUCCESSFUL!")
        print("Release URL:", release_info['html_url'])
        print("Download URL:", asset_info['browser_download_url'])
        print("Uploaded size:", asset_info['size'])

if __name__ == '__main__':
    run()
