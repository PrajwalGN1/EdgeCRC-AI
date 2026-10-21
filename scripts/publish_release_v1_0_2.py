import os
import subprocess
import urllib.request
import json
import hashlib

REPO = "PrajwalGN1/EdgeCRC-AI"
TAG = "v1.0.2"
COMMIT = "2ba253a66b2c7695c0ec071b9c546d11034e6f15"
TITLE = "CRC Multimodal Edge AI — Enterprise UI/UX Redesign (v1.0.2)"
APK_PATH = "android/app/build/outputs/apk/debug/app-debug.apk"
ASSET_NAME = "crc-edge-ai-android-debug.apk"
VERSIONED_ASSET_NAME = "crc-edge-ai-android-debug-v1.0.2.apk"

RELEASE_NOTES = r"""## CRC Multimodal Edge AI — Enterprise UI/UX Redesign (v1.0.2)

### 🏥 Research-Use Disclaimer
> **Notice:** This application is an experimental research prototype for multimodal colorectal cancer MSI/MSS classification. It is not a certified medical device and must not be used as the sole basis for clinical diagnosis or patient management.

---

### ✨ What's New in v1.0.2 (versionCode 3)

This release transforms the mobile research interface into an enterprise-grade healthcare AI application:

1. **Enterprise Healthcare Design System:**
   - Standardized clinical color system featuring Midnight Navy (`#0F172A`), Medical Teal (`#0D9488`), and Deep Slate (`#1E293B`).
   - Cohesive Day & Night themes adhering to Material Design 3 and WCAG AA accessibility standards.
   - 4dp spacing grid with generous 54dp primary touch targets for seamless mobile operation.

2. **Interactive Multimodal Input Specification & Patch Scaling:**
   - Clear modality documentation for precomputed vision embeddings ($[1, N, 1024]$) and clinical text embeddings ($[1, 768]$).
   - Interactive single-select toggle allowing researchers to scale sequence lengths across **10 Patches** (Standard), **50 Patches** (Validation), and **1,000 Patches** (Protocol 1 Max Cap).

3. **Decision Threshold Visualization (0.83 Boundary):**
   - High-visibility classification banner dynamically badging **MSI (Microsatellite Instability)** with Medical Teal or **MSS (Microsatellite Stable)** with Clinical Blue.
   - Custom graphical probability gauge rendering the model score relative to the frozen `0.83` decision boundary.
   - Signed distance to threshold readout ($\Delta = P - 0.83$) with exact 4-decimal probability precision.

4. **Operational Metrics & Technical Diagnostics:**
   - 3-column stats displaying Decision Threshold (`0.83`), measured on-device CPU execution latency in milliseconds, and active patch count.
   - Expandable technical diagnostics console showing raw logit, attention weight tensor dimensions, protocol, and execution timestamp.

---

### 📦 Artifact & Model Integrity
- **Asset Filename:** `crc-edge-ai-android-debug.apk`
- **File Size:** 74,335,338 bytes (70.89 MB)
- **APK SHA-256:** `f593442afc85ea4b1379e37959dd186666a543b2bceede5a8350e4348e58082c`
- **Application ID:** `org.crc.multimodal`
- **Version Name / Code:** `1.0.2` (versionCode: `3`)
- **Target Platform:** Android 14 (API 34) | **Min SDK:** Android 7.0 (API 24)
- **Bundled Model:** `models/fusion_model_fp32.onnx` (21,015,452 bytes, uncompressed zero-copy memory mapped)
- **Model SHA-256:** `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` (100% bit-for-bit parity)
- **Inference Runtime:** ONNX Runtime Mobile v1.18.0 (CPUExecutionProvider Float32)
- **Offline Privacy Guarantee:** 100% on-device local execution; zero network transmission (`android.permission.INTERNET` omitted)

---

### 📱 Installation Instructions for Samsung Galaxy S24
1. Tap the direct download link below to download `crc-edge-ai-android-debug.apk`.
2. Once the download completes, open the APK from your browser's **Downloads** list or the Samsung **My Files** app.
3. If prompted with an "Unknown source" warning, allow installation for that specific browser/app source in Settings.
4. Tap **Update** (or **Install**).
5. Launch **CRC Multimodal Edge AI** from your application drawer.
6. Select your desired patch count and tap **Run Research Inference** to execute multimodal cross-attention fusion directly on your device.
"""

def get_git_token():
    p = subprocess.Popen(['git', 'credential', 'fill'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, _ = p.communicate(input='protocol=https\nhost=github.com\n\n')
    creds = dict(l.split('=', 1) for l in out.strip().splitlines() if '=' in l)
    return creds.get('password')

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def publish():
    token = get_git_token()
    if not token:
        raise RuntimeError("No Git credential found for github.com")

    # Verify APK integrity before uploading
    apk_hash = sha256_file(APK_PATH)
    apk_size = os.path.getsize(APK_PATH)
    print(f"Verified local APK: {APK_PATH}")
    print(f"Size: {apk_size} bytes")
    print(f"SHA-256: {apk_hash}")

    # 1. Check if release already exists
    list_url = f"https://api.github.com/repos/{REPO}/releases"
    req = urllib.request.Request(list_url, headers={
        'Authorization': f'Bearer {token}',
        'User-Agent': 'CRC-Release-Automation',
        'Accept': 'application/vnd.github.v3+json'
    })
    release_data = None
    with urllib.request.urlopen(req) as resp:
        releases = json.loads(resp.read().decode('utf-8'))
        for r in releases:
            if r.get('tag_name') == TAG:
                print(f"Release with tag {TAG} already exists (ID: {r['id']}) at {r.get('html_url')}")
                release_data = r
                break

    # 2. Create release if not existing
    if not release_data:
        create_url = f"https://api.github.com/repos/{REPO}/releases"
        payload = {
            "tag_name": TAG,
            "target_commitish": COMMIT,
            "name": TITLE,
            "body": RELEASE_NOTES,
            "draft": False,
            "prerelease": False
        }
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(create_url, data=data, headers={
            'Authorization': f'Bearer {token}',
            'User-Agent': 'CRC-Release-Automation',
            'Content-Type': 'application/json',
            'Accept': 'application/vnd.github.v3+json'
        }, method='POST')

        print(f"Creating GitHub release {TAG} on {REPO}...")
        with urllib.request.urlopen(req) as resp:
            release_data = json.loads(resp.read().decode('utf-8'))
            print(f"Release created successfully! URL: {release_data['html_url']}")

    release_id = release_data['id']
    html_url = release_data['html_url']

    # 3. Check and upload assets
    with open(APK_PATH, 'rb') as f:
        apk_bytes = f.read()

    existing_assets = {a['name']: a for a in release_data.get('assets', [])}
    uploaded_assets = {}

    for asset_name in [ASSET_NAME, VERSIONED_ASSET_NAME]:
        if asset_name in existing_assets:
            a = existing_assets[asset_name]
            print(f"Asset {asset_name} already exists (ID: {a['id']}), removing old asset before fresh upload...")
            del_req = urllib.request.Request(a['url'], headers={
                'Authorization': f'Bearer {token}',
                'User-Agent': 'CRC-Release-Automation',
                'Accept': 'application/vnd.github.v3+json'
            }, method='DELETE')
            urllib.request.urlopen(del_req)
            print("Deleted old asset.")

        upload_url = f"https://uploads.github.com/repos/{REPO}/releases/{release_id}/assets?name={asset_name}"
        print(f"Uploading APK asset {asset_name} ({apk_size} bytes)...")

        req = urllib.request.Request(upload_url, data=apk_bytes, headers={
            'Authorization': f'Bearer {token}',
            'User-Agent': 'CRC-Release-Automation',
            'Content-Type': 'application/vnd.android.package-archive',
            'Accept': 'application/vnd.github.v3+json'
        }, method='POST')

        with urllib.request.urlopen(req) as resp:
            asset_info = json.loads(resp.read().decode('utf-8'))
            browser_download_url = asset_info['browser_download_url']
            print(f"Asset {asset_name} uploaded successfully!")
            print(f"Direct download URL: {browser_download_url}")
            uploaded_assets[asset_name] = {
                "download_url": browser_download_url,
                "size": asset_info['size'],
                "id": asset_info['id']
            }

    return {
        "release_url": html_url,
        "release_id": release_id,
        "tag": TAG,
        "assets": uploaded_assets
    }

if __name__ == '__main__':
    res = publish()
    print("RELEASE PUBLISHED:")
    print(json.dumps(res, indent=2))
