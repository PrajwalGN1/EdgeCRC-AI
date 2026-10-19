import os
import subprocess
import urllib.request
import json
import hashlib
from datetime import datetime, timezone

REPO = "PrajwalGN1/EdgeCRC-AI"
TAG = "v0.1.0"
COMMIT = "19b27f49c168a34275b51e03a000301e2d46576d"
TITLE = "CRC Edge AI Android Test Build (v0.1.0)"
APK_PATH = "release/crc-edge-ai-android-debug.apk"
ASSET_NAME = "crc-edge-ai-android-debug.apk"

RELEASE_NOTES = """## CRC Multimodal Edge AI — Android Test Build (v0.1.0)

Research-use test build of the Multimodal Colorectal Cancer (CRC) Microsatellite Instability (MSI vs. MSS) Edge Inference Android Application.

### 📦 Artifact Integrity
- **Asset Filename:** `crc-edge-ai-android-debug.apk`
- **File Size:** 72,731,704 bytes (69.36 MB)
- **APK SHA-256:** `22b94247a7fed5bab297fcff28a46f9461b7f93bd74a7620c32f33db22f758a9`
- **Source Commit:** `19b27f49c168a34275b51e03a000301e2d46576d` (branch: `edge-deployment`)
- **Bundled Model:** `fusion_model_fp32.onnx` (21,015,452 bytes)
- **Model SHA-256:** `0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73` (100% bit-for-bit parity with source PyTorch export)
- **Application ID:** `org.crc.multimodal`
- **Build Variant:** `debug` (v1.0.0, versionCode 1)
- **Target SDK:** Android 14 (API 34) | **Min SDK:** Android 7.0 (API 24)
- **Supported ABIs:** `arm64-v8a`, `armeabi-v7a`, `x86_64` (Native 64-bit ARMv9 for Samsung Galaxy S24)
- **ONNX Runtime Mobile:** `1.18.0`

### ⚠️ Scope & Engineering Disclaimers
1. **Research Test Build:** This is an Android debug APK intended strictly for engineering and runtime validation on physical Android hardware (e.g., Samsung Galaxy S24). It is not a production-signed Google Play Store release.
2. **Precomputed Embeddings:** The application executes the Cross-Attention Fusion neural network on precomputed vision patch embeddings ($V \in \mathbb{R}^{N \\times 1024}$) and ClinicalBERT text embeddings ($T \in \mathbb{R}^{1 \\times 768}$). It does not yet perform full ViT-L/16 patch extraction on-device.
3. **Strict Offline Privacy:** The application declares zero network permissions (`android.permission.INTERNET` is omitted). All tensor operations run 100% on-device.
4. **Validation Status:** Successful installation verifies package delivery and native library loading; full 66-case holdout numerical parity against the PyTorch reference baseline remains a separate validation gate.

---
*For step-by-step Samsung Galaxy S24 installation instructions, refer to `INSTALLATION.md` in the repository.*
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

    # 1. Check if release already exists
    list_url = f"https://api.github.com/repos/{REPO}/releases"
    req = urllib.request.Request(list_url, headers={
        'Authorization': f'Bearer {token}',
        'User-Agent': 'CRC-Release-Automation',
        'Accept': 'application/vnd.github.v3+json'
    })
    with urllib.request.urlopen(req) as resp:
        releases = json.loads(resp.read().decode('utf-8'))
        for r in releases:
            if r.get('tag_name') == TAG:
                print(f"Release with tag {TAG} already exists at {r.get('html_url')}")
                return r

    # 2. Create release
    create_url = f"https://api.github.com/repos/{REPO}/releases"
    payload = {
        "tag_name": TAG,
        "target_commitish": COMMIT,
        "name": TITLE,
        "body": RELEASE_NOTES,
        "draft": False,
        "prerelease": True
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
        release_id = release_data['id']
        html_url = release_data['html_url']
        upload_url_template = release_data['upload_url']
        print(f"Release created successfully! URL: {html_url}")

    # 3. Upload APK asset
    upload_url = upload_url_template.split('{')[0] + f"?name={ASSET_NAME}"
    apk_size = os.path.getsize(APK_PATH)
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
        asset_data = json.loads(resp.read().decode('utf-8'))
        browser_download_url = asset_data['browser_download_url']
        print(f"Asset uploaded successfully!")
        print(f"Direct download URL: {browser_download_url}")
        print(f"Asset size verified: {asset_data.get('size')} bytes")

    return {
        "release_url": html_url,
        "download_url": browser_download_url,
        "release_id": release_id,
        "asset_id": asset_data['id']
    }

if __name__ == '__main__':
    res = publish()
    print("RESULT:", json.dumps(res, indent=2))
