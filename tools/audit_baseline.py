"""
tools/audit_baseline.py
Captures baseline environment, model integrity hashes, git state,
and runtime configuration for Phase 5.1 audit.
"""

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path("D:/CRC Edge deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2")
AUDIT_DIR = REPO_ROOT / "reports" / "mobile_deployment" / "phase5_1_audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

def get_cmd_output(cmd, cwd=REPO_ROOT):
    try:
        res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()

def main():
    # 1. Git State
    branch = get_cmd_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    commit = get_cmd_output(["git", "rev-parse", "HEAD"])
    status = get_cmd_output(["git", "status", "--porcelain"])

    # 2. Python Environment
    import torch
    import onnx
    import onnxruntime as ort

    python_version = sys.version
    torch_version = torch.__version__
    onnx_version = onnx.__version__
    ort_version = ort.__version__

    # 3. Model Integrity Checks
    root_onnx = REPO_ROOT / "models" / "fusion_model_fp32.onnx"
    root_pth = REPO_ROOT / "models" / "fusion_model_v1_1.pth"
    assets_onnx = REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "models" / "fusion_model_fp32.onnx"
    apk_path = REPO_ROOT / "android" / "app" / "build" / "outputs" / "apk" / "debug" / "app-debug.apk"

    expected_onnx_sha256 = "0402F6C522BEAC5733E30EEECAD908927957B5D4164B6B896663F43BA3702D73"
    expected_onnx_size = 21015452
    expected_pth_size = 21007347

    root_onnx_size = root_onnx.stat().st_size if root_onnx.exists() else None
    root_onnx_sha = file_sha256(root_onnx) if root_onnx.exists() else None

    root_pth_size = root_pth.stat().st_size if root_pth.exists() else None
    root_pth_sha = file_sha256(root_pth) if root_pth.exists() else None

    assets_onnx_size = assets_onnx.stat().st_size if assets_onnx.exists() else None
    assets_onnx_sha = file_sha256(assets_onnx) if assets_onnx.exists() else None

    apk_onnx_sha = None
    apk_onnx_size = None
    if apk_path.exists():
        with zipfile.ZipFile(apk_path, "r") as z:
            # find fusion model inside APK
            for zname in z.namelist():
                if "fusion_model_fp32.onnx" in zname:
                    apk_data = z.read(zname)
                    apk_onnx_size = len(apk_data)
                    apk_onnx_sha = hashlib.sha256(apk_data).hexdigest().upper()
                    break

    # 4. ADB Status
    adb_path = Path("C:/Android/Sdk/platform-tools/adb.exe")
    adb_output = get_cmd_output([str(adb_path), "devices", "-l"]) if adb_path.exists() else "ADB not found"

    # 5. Build Baseline Manifest
    manifest = {
        "audit_phase": "PHASE_5_1_INDEPENDENT_ACCEPTANCE_AUDIT",
        "timestamp_utc": "2026-10-23T11:37:00Z",
        "git": {
            "branch": branch,
            "commit": commit,
            "working_tree_clean": len(status.strip()) == 0,
            "status_summary": status.splitlines()[:15]
        },
        "host_environment": {
            "python": python_version,
            "torch": torch_version,
            "onnx": onnx_version,
            "onnxruntime": ort_version,
            "os": os.name,
            "platform": sys.platform
        },
        "models": {
            "root_onnx": {
                "path": str(root_onnx.relative_to(REPO_ROOT)),
                "size_bytes": root_onnx_size,
                "expected_size_bytes": expected_onnx_size,
                "sha256": root_onnx_sha,
                "expected_sha256": expected_onnx_sha256,
                "size_matches": root_onnx_size == expected_onnx_size,
                "sha256_matches": root_onnx_sha == expected_onnx_sha256
            },
            "root_pth": {
                "path": str(root_pth.relative_to(REPO_ROOT)),
                "size_bytes": root_pth_size,
                "expected_size_bytes": expected_pth_size,
                "sha256": root_pth_sha,
                "size_matches": root_pth_size == expected_pth_size
            },
            "assets_onnx": {
                "path": str(assets_onnx.relative_to(REPO_ROOT)),
                "size_bytes": assets_onnx_size,
                "expected_size_bytes": expected_onnx_size,
                "sha256": assets_onnx_sha,
                "matches_root_onnx": assets_onnx_sha == root_onnx_sha
            },
            "apk_packaged_onnx": {
                "apk_path": str(apk_path.relative_to(REPO_ROOT)) if apk_path.exists() else None,
                "size_bytes": apk_onnx_size,
                "sha256": apk_onnx_sha,
                "matches_root_onnx": apk_onnx_sha == root_onnx_sha
            }
        },
        "runtime_threshold_contract": {
            "decision_threshold": 0.83,
            "class_0_label": "MSS",
            "class_1_label": "MSI",
            "decision_rule": "MSI if probability >= 0.83 else MSS",
            "vision_input_shape": "[1, N, 1024]",
            "text_input_shape": "[1, 768]",
            "data_type": "Float32"
        },
        "target_device": {
            "adb_executable": str(adb_path),
            "adb_devices_output": adb_output,
            "device_authorized": "device" in adb_output and "unauthorized" not in adb_output
        }
    }

    out_file = AUDIT_DIR / "baseline_manifest.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Baseline manifest written to: {out_file}")
    print(f"Root ONNX SHA-256 Match: {manifest['models']['root_onnx']['sha256_matches']}")
    print(f"Assets ONNX Match: {manifest['models']['assets_onnx']['matches_root_onnx']}")
    print(f"APK Packaged ONNX Match: {manifest['models']['apk_packaged_onnx']['matches_root_onnx']}")
    print(f"Target Device Authorized: {manifest['target_device']['device_authorized']}")

if __name__ == "__main__":
    main()
