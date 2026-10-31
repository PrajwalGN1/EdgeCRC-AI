"""
tools/audit_roundtrip.py
Phase 5.1B: Authentic Bundle Round-Trip Integrity Audit.
Exports TCGA-A6-3807 to CRCB v2 and JSON v2 in a disposable directory,
verifies bit-for-bit lossless round-trip, shape, dtype, and SHA-256 integrity.
"""

import base64
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path("D:/CRC Edge deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2")
sys.path.insert(0, str(REPO_ROOT))
AUDIT_DIR = REPO_ROOT / "reports" / "mobile_deployment" / "phase5_1_audit"
DISPOSABLE_DIR = AUDIT_DIR / "roundtrip_test"
DISPOSABLE_DIR.mkdir(parents=True, exist_ok=True)

from tools.export_specimen_bundle import (
    load_upstream_tensors,
    export_crcb_bundle,
    export_json_bundle,
    verify_bundle,
    MAGIC_CRCB,
    SCHEMA_VERSION,
    VISION_DIM,
    TEXT_DIM,
    DATA_TYPE_FLOAT32
)

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()

def main():
    case_id = "TCGA-A6-3807"
    tensors_dir = REPO_ROOT / "data" / "clean_tensors"
    
    # 1. Load original source tensors
    v_orig_np, t_orig_np = load_upstream_tensors(case_id, tensors_dir)
    num_patches = v_orig_np.shape[0]
    
    # Compute raw bytes and hashes of original tensors
    v_orig_bytes = v_orig_np.astype("<f4").tobytes()
    t_orig_bytes = t_orig_np.astype("<f4").tobytes()
    v_orig_sha256 = hashlib.sha256(v_orig_bytes).hexdigest().upper()
    t_orig_sha256 = hashlib.sha256(t_orig_bytes).hexdigest().upper()
    
    # 2. Export to CRCB v2 in disposable directory
    crcb_path = DISPOSABLE_DIR / f"{case_id}.crcbundle"
    export_crcb_bundle(
        case_id=case_id,
        v_np=v_orig_np,
        t_np=t_orig_np,
        out_path=crcb_path,
        description="Phase 5.1B Audit Specimen"
    )
    
    # 3. Export to JSON v2 in disposable directory
    json_path = DISPOSABLE_DIR / f"{case_id}.crcmanifest.json"
    export_json_bundle(
        case_id=case_id,
        v_np=v_orig_np,
        t_np=t_orig_np,
        out_path=json_path,
        description="Phase 5.1B Audit Specimen"
    )
    
    # 4. Read back CRCB v2 and verify
    with open(crcb_path, "rb") as f:
        magic = f.read(4)
        assert magic == MAGIC_CRCB
        ver = struct.unpack("<i", f.read(4))[0]
        id_len = struct.unpack("<i", f.read(4))[0]
        read_id = f.read(id_len).decode("utf-8")
        desc_len = struct.unpack("<i", f.read(4))[0]
        read_desc = f.read(desc_len).decode("utf-8")
        r_patches, r_vdim, r_tdim, r_dtype = struct.unpack("<iiii", f.read(16))
        r_v_sha = f.read(32).hex().upper()
        r_t_sha = f.read(32).hex().upper()
        r_t_bytes = f.read(r_tdim * 4)
        r_v_bytes = f.read(r_patches * r_vdim * 4)
        
    crcb_v_read = np.frombuffer(r_v_bytes, dtype="<f4").reshape(r_patches, r_vdim)
    crcb_t_read = np.frombuffer(r_t_bytes, dtype="<f4")
    
    crcb_v_diff = float(np.max(np.abs(crcb_v_read - v_orig_np)))
    crcb_t_diff = float(np.max(np.abs(crcb_t_read - t_orig_np)))
    
    # 5. Read back JSON v2 and verify
    with open(json_path, "r", encoding="utf-8") as f:
        j_data = json.load(f)
    
    j_v_bytes = base64.b64decode(j_data["vision_tensor"]["data_base64"])
    j_t_bytes = base64.b64decode(j_data["text_tensor"]["data_base64"])
    json_v_read = np.frombuffer(j_v_bytes, dtype="<f4").reshape(j_data["patch_count"], VISION_DIM)
    json_t_read = np.frombuffer(j_t_bytes, dtype="<f4")
    
    json_v_diff = float(np.max(np.abs(json_v_read - v_orig_np)))
    json_t_diff = float(np.max(np.abs(json_t_read - t_orig_np)))
    
    results = {
        "audit_phase": "PHASE_5_1B_AUTHENTIC_BUNDLE_ROUNDTRIP",
        "case_id": case_id,
        "source_tensors": {
            "vision_path": str((tensors_dir / f"{case_id}_vision.pt").relative_to(REPO_ROOT)),
            "text_path": str((tensors_dir / f"{case_id}_text.pt").relative_to(REPO_ROOT)),
            "patch_count": num_patches,
            "vision_shape": list(v_orig_np.shape),
            "text_shape": list(t_orig_np.shape),
            "vision_dtype": str(v_orig_np.dtype),
            "text_dtype": str(t_orig_np.dtype),
            "vision_sha256": v_orig_sha256,
            "text_sha256": t_orig_sha256
        },
        "crcb_v2_bundle": {
            "path": str(crcb_path.relative_to(REPO_ROOT)),
            "file_size_bytes": crcb_path.stat().st_size,
            "bundle_sha256": file_sha256(crcb_path),
            "parsed_version": ver,
            "parsed_specimen_id": read_id,
            "parsed_patch_count": r_patches,
            "parsed_vision_dim": r_vdim,
            "parsed_text_dim": r_tdim,
            "parsed_data_type": r_dtype,
            "payload_vision_sha256": r_v_sha,
            "payload_text_sha256": r_t_sha,
            "sha256_vision_match": r_v_sha == v_orig_sha256,
            "sha256_text_match": r_t_sha == t_orig_sha256,
            "max_abs_vision_diff": crcb_v_diff,
            "max_abs_text_diff": crcb_t_diff,
            "lossless_exact_match": (crcb_v_diff == 0.0) and (crcb_t_diff == 0.0)
        },
        "json_v2_bundle": {
            "path": str(json_path.relative_to(REPO_ROOT)),
            "file_size_bytes": json_path.stat().st_size,
            "bundle_sha256": file_sha256(json_path),
            "schema_version": j_data["schema_version"],
            "specimen_id": j_data["specimen_id"],
            "patch_count": j_data["patch_count"],
            "declared_vision_shape": j_data["vision_tensor"]["shape"],
            "declared_text_shape": j_data["text_tensor"]["shape"],
            "declared_vision_sha256": j_data["vision_tensor"]["sha256"].upper(),
            "declared_text_sha256": j_data["text_tensor"]["sha256"].upper(),
            "sha256_vision_match": j_data["vision_tensor"]["sha256"].upper() == v_orig_sha256,
            "sha256_text_match": j_data["text_tensor"]["sha256"].upper() == t_orig_sha256,
            "max_abs_vision_diff": json_v_diff,
            "max_abs_text_diff": json_t_diff,
            "lossless_exact_match": (json_v_diff == 0.0) and (json_t_diff == 0.0)
        },
        "overall_roundtrip_status": "PASS" if (crcb_v_diff == 0.0 and crcb_t_diff == 0.0 and json_v_diff == 0.0 and json_t_diff == 0.0) else "FAIL"
    }
    
    out_file = AUDIT_DIR / "bundle_roundtrip_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"Round-trip results saved to: {out_file}")
    print(f"CRCB Lossless Exact Match: {results['crcb_v2_bundle']['lossless_exact_match']}")
    print(f"JSON Lossless Exact Match: {results['json_v2_bundle']['lossless_exact_match']}")
    print(f"Overall Status: {results['overall_roundtrip_status']}")

if __name__ == "__main__":
    main()
