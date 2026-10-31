#!/usr/bin/env python3
"""
tools/export_specimen_bundle.py

Deterministic conversion and export utility for packaging upstream precomputed
multimodal embedding tensors (Phikon-v2 [N, 1024] and ClinicalBERT [1, 768])
into verified authentic feature bundles for the Android Edge AI workstation.

Specifications Supported:
1. CRCB Binary Bundle (Version 2):
   - Header: Magic 'CRCB' (4 bytes), Version=2 (int32), Specimen ID, Description,
     Patch Count N (int32), Vision Dim=1024 (int32), Text Dim=768 (int32),
     Data Type=1 (Float32 Little-Endian), Vision SHA-256 (32 bytes), Text SHA-256 (32 bytes)
   - Payloads: Text (768 * 4 bytes), Vision (N * 1024 * 4 bytes)
2. JSON Bundle (Version 2.0):
   - Structured JSON with shapes, data types, SHA-256 hashes, and Base64-encoded
     Little-Endian Float32 arrays.

Preserves exact numerical values with zero loss (diff = 0.0).
"""

import argparse
import base64
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TENSORS_DIR = REPO_ROOT / "data" / "clean_tensors"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "specimen_bundles"

MAGIC_CRCB = b"CRCB"
SCHEMA_VERSION = 2
DATA_TYPE_FLOAT32 = 1
VISION_DIM = 1024
TEXT_DIM = 768


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_upstream_tensors(case_id: str, tensors_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads upstream PyTorch tensors for the given case ID.
    Returns: (vision_np, text_np) as float32 numpy arrays.
    """
    vision_path = tensors_dir / f"{case_id}_vision.pt"
    text_path = tensors_dir / f"{case_id}_text.pt"

    if not vision_path.exists():
        raise FileNotFoundError(f"Missing upstream vision tensor: {vision_path}")
    if not text_path.exists():
        raise FileNotFoundError(f"Missing upstream text tensor: {text_path}")

    v_tensor = torch.load(vision_path, weights_only=True)
    t_tensor = torch.load(text_path, weights_only=True)

    v_np = v_tensor.detach().cpu().numpy().astype(np.float32)
    t_np = t_tensor.detach().cpu().numpy().astype(np.float32)

    # Validate shapes
    if v_np.ndim == 2:
        num_patches, v_dim = v_np.shape
    elif v_np.ndim == 3 and v_np.shape[0] == 1:
        num_patches, v_dim = v_np.shape[1], v_np.shape[2]
        v_np = v_np.squeeze(0)
    else:
        raise ValueError(f"Invalid vision tensor shape for {case_id}: {v_np.shape}")

    if v_dim != VISION_DIM:
        raise ValueError(f"Invalid vision feature dimension: expected {VISION_DIM}, got {v_dim}")

    t_flat = t_np.flatten()
    if len(t_flat) != TEXT_DIM:
        raise ValueError(f"Invalid text feature dimension: expected {TEXT_DIM}, got {len(t_flat)}")

    # Validate finiteness
    if not np.isfinite(v_np).all():
        raise ValueError(f"Vision tensor for {case_id} contains non-finite values (NaN or Inf)")
    if not np.isfinite(t_flat).all():
        raise ValueError(f"Text tensor for {case_id} contains non-finite values (NaN or Inf)")

    return v_np, t_flat


def export_crcb_bundle(
    case_id: str,
    v_np: np.ndarray,
    t_np: np.ndarray,
    out_path: Path,
    description: str = ""
) -> Path:
    """
    Exports tensors to CRCB v2 binary format.
    """
    num_patches = v_np.shape[0]
    id_bytes = case_id.strip().encode("utf-8")
    desc_bytes = description.strip().encode("utf-8")

    # Text bytes Little-Endian Float32
    text_bytes = t_np.astype("<f4").tobytes()
    assert len(text_bytes) == TEXT_DIM * 4

    # Vision bytes Little-Endian Float32
    vision_bytes = v_np.astype("<f4").tobytes()
    assert len(vision_bytes) == num_patches * VISION_DIM * 4

    text_hash = hashlib.sha256(text_bytes).digest()
    vision_hash = hashlib.sha256(vision_bytes).digest()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        # Magic
        f.write(MAGIC_CRCB)
        # Version
        f.write(struct.pack("<i", SCHEMA_VERSION))
        # Specimen ID
        f.write(struct.pack("<i", len(id_bytes)))
        f.write(id_bytes)
        # Description
        f.write(struct.pack("<i", len(desc_bytes)))
        f.write(desc_bytes)
        # Patch count, vision dim, text dim, data type
        f.write(struct.pack("<iiii", num_patches, VISION_DIM, TEXT_DIM, DATA_TYPE_FLOAT32))
        # SHA-256 digests (32 bytes each)
        f.write(vision_hash)
        f.write(text_hash)
        # Text payload
        f.write(text_bytes)
        # Vision payload
        f.write(vision_bytes)

    return out_path


def export_json_bundle(
    case_id: str,
    v_np: np.ndarray,
    t_np: np.ndarray,
    out_path: Path,
    description: str = ""
) -> Path:
    """
    Exports tensors to structured JSON bundle.
    """
    num_patches = v_np.shape[0]
    text_bytes = t_np.astype("<f4").tobytes()
    vision_bytes = v_np.astype("<f4").tobytes()

    text_b64 = base64.b64encode(text_bytes).decode("ascii")
    vision_b64 = base64.b64encode(vision_bytes).decode("ascii")

    bundle_dict = {
        "schema_version": f"{SCHEMA_VERSION}.0",
        "specimen_id": case_id.strip(),
        "description": description.strip(),
        "patch_count": num_patches,
        "vision_tensor": {
            "shape": [1, num_patches, VISION_DIM],
            "dtype": "float32",
            "endianness": "little",
            "sha256": compute_sha256(vision_bytes),
            "data_base64": vision_b64
        },
        "text_tensor": {
            "shape": [1, TEXT_DIM],
            "dtype": "float32",
            "endianness": "little",
            "sha256": compute_sha256(text_bytes),
            "data_base64": text_b64
        }
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle_dict, f, indent=2)

    return out_path


def verify_bundle(bundle_path: Path, expected_v: np.ndarray, expected_t: np.ndarray) -> bool:
    """
    Verifies that a serialized bundle matches original tensors with zero loss.
    """
    if bundle_path.suffix.lower() == ".json":
        with open(bundle_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        v_raw = base64.b64decode(data["vision_tensor"]["data_base64"])
        t_raw = base64.b64decode(data["text_tensor"]["data_base64"])
        v_read = np.frombuffer(v_raw, dtype="<f4").reshape(-1, VISION_DIM)
        t_read = np.frombuffer(t_raw, dtype="<f4")
    else:
        with open(bundle_path, "rb") as f:
            magic = f.read(4)
            assert magic == MAGIC_CRCB, f"Invalid magic: {magic}"
            version = struct.unpack("<i", f.read(4))[0]
            assert version == SCHEMA_VERSION, f"Invalid version: {version}"
            id_len = struct.unpack("<i", f.read(4))[0]
            f.read(id_len)
            desc_len = struct.unpack("<i", f.read(4))[0]
            f.read(desc_len)
            num_patches, v_dim, t_dim, data_type = struct.unpack("<iiii", f.read(16))
            assert v_dim == VISION_DIM and t_dim == TEXT_DIM
            v_hash = f.read(32)
            t_hash = f.read(32)
            t_bytes = f.read(t_dim * 4)
            v_bytes = f.read(num_patches * v_dim * 4)
            assert hashlib.sha256(t_bytes).digest() == t_hash, "Text SHA-256 verification failed"
            assert hashlib.sha256(v_bytes).digest() == v_hash, "Vision SHA-256 verification failed"
            t_read = np.frombuffer(t_bytes, dtype="<f4")
            v_read = np.frombuffer(v_bytes, dtype="<f4").reshape(num_patches, VISION_DIM)

    # Numerical exactness check
    max_v_diff = np.max(np.abs(v_read - expected_v))
    max_t_diff = np.max(np.abs(t_read - expected_t))
    assert max_v_diff == 0.0, f"Vision numerical difference > 0: {max_v_diff}"
    assert max_t_diff == 0.0, f"Text numerical difference > 0: {max_t_diff}"
    return True


def main():
    parser = argparse.ArgumentParser(description="Export authentic multimodal feature bundles.")
    parser.add_argument("--case", type=str, help="Specific case ID (e.g. TCGA-A6-3807)")
    parser.add_argument("--keys", type=str, help="Path to JSON file containing list of case IDs")
    parser.add_argument("--all", action="store_true", help="Export all clean_tensors")
    parser.add_argument("--format", choices=["crcbundle", "json", "npy"], default="crcbundle", help="Bundle format")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("--tensors-dir", type=str, default=str(TENSORS_DIR), help="clean_tensors directory")
    parser.add_argument("--verify", action="store_true", default=True, help="Verify bundles after export")
    args = parser.parse_args()

    tensors_dir = Path(args.tensors_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.case:
        cases = [args.case.strip()]
    elif args.keys:
        with open(args.keys, "r") as f:
            cases = json.load(f)
    elif args.all:
        vision_files = list(tensors_dir.glob("*_vision.pt"))
        cases = sorted([f.name.replace("_vision.pt", "") for f in vision_files])
    else:
        # Default sample cases for Phase 5 verification
        cases = ["TCGA-A6-3807", "TCGA-A6-6138", "TCGA-D5-6540", "TCGA-CM-6675"]

    print(f"Exporting {len(cases)} cases in format '{args.format}' to '{out_dir}'...")

    exported = 0
    for case_id in cases:
        try:
            v_np, t_np = load_upstream_tensors(case_id, tensors_dir)
            num_patches = v_np.shape[0]

            if args.format == "json":
                out_file = out_dir / f"{case_id}.crcmanifest.json"
                export_json_bundle(case_id, v_np, t_np, out_file)
                if args.verify:
                    verify_bundle(out_file, v_np, t_np)
                file_size_kb = out_file.stat().st_size / 1024.0
                print(f"  [OK] {case_id:15s} | N={num_patches:4d} patches | File: {out_file.name} ({file_size_kb:.1f} KB)")
            elif args.format == "npy":
                v_out = out_dir / f"{case_id}_vision.npy"
                t_out = out_dir / f"{case_id}_text.npy"
                np.save(v_out, v_np.astype("<f4"))
                np.save(t_out, t_np.astype("<f4"))
                if args.verify:
                    v_read = np.load(v_out)
                    t_read = np.load(t_out)
                    assert np.max(np.abs(v_read - v_np)) == 0.0, "Vision NPY verification failed"
                    assert np.max(np.abs(t_read - t_np)) == 0.0, "Text NPY verification failed"
                total_kb = (v_out.stat().st_size + t_out.stat().st_size) / 1024.0
                print(f"  [OK] {case_id:15s} | N={num_patches:4d} patches | Files: {v_out.name}, {t_out.name} ({total_kb:.1f} KB)")
            else:
                out_file = out_dir / f"{case_id}.crcbundle"
                export_crcb_bundle(case_id, v_np, t_np, out_file)
                if args.verify:
                    verify_bundle(out_file, v_np, t_np)
                file_size_kb = out_file.stat().st_size / 1024.0
                print(f"  [OK] {case_id:15s} | N={num_patches:4d} patches | File: {out_file.name} ({file_size_kb:.1f} KB)")

            exported += 1
        except Exception as e:
            print(f"  [ERROR] {case_id}: {e}", file=sys.stderr)

    print(f"\nSuccessfully exported and verified {exported}/{len(cases)} bundles.")


if __name__ == "__main__":
    main()
