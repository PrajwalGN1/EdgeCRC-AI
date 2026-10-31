"""
Benchmark script for Phase 5 authentic feature bundle ingestion and FP32 fusion inference.
Measures parsing time, validation time, memory footprint, pure ONNX runtime inference,
and end-to-end processing pipeline latency.
"""

import base64
import glob
import hashlib
import json
import os
import struct
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODEL_PATH = "android/app/src/main/assets/models/fusion_model_fp32.onnx"
BUNDLE_DIR = "data/specimen_bundles"
OUTPUT_JSON = "reports/mobile_deployment/phase5_benchmarks/ingestion_benchmark_results.json"
OUTPUT_CSV = "reports/mobile_deployment/phase5_benchmarks/ingestion_benchmark_results.csv"

MAGIC_CRCB = b"CRCB"
SCHEMA_VERSION = 2
VISION_DIM = 1024
TEXT_DIM = 768


def parse_crcb_bundle(file_path: str):
    t0 = time.perf_counter()
    with open(file_path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC_CRCB:
            raise ValueError(f"Invalid magic: {magic}")
        version = struct.unpack("<i", f.read(4))[0]
        id_len = struct.unpack("<i", f.read(4))[0]
        specimen_id = f.read(id_len).decode("utf-8")
        desc_len = struct.unpack("<i", f.read(4))[0]
        description = f.read(desc_len).decode("utf-8") if desc_len > 0 else ""
        num_patches, v_dim, t_dim, data_type = struct.unpack("<iiii", f.read(16))
        v_hash = f.read(32)
        t_hash = f.read(32)
        text_bytes = f.read(t_dim * 4)
        vision_bytes = f.read(num_patches * v_dim * 4)
        
    t1 = time.perf_counter()
    parse_time_ms = (t1 - t0) * 1000.0

    metadata = {
        "schema_version": version,
        "specimen_id": specimen_id,
        "description": description,
        "num_patches": num_patches,
        "vision_dim": v_dim,
        "text_dim": t_dim,
        "data_type": data_type,
        "vision_hash": v_hash,
        "text_hash": t_hash,
    }
    return metadata, vision_bytes, text_bytes, parse_time_ms


def parse_json_bundle(file_path: str):
    t0 = time.perf_counter()
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    v_raw = base64.b64decode(data["vision_tensor"]["data_base64"])
    t_raw = base64.b64decode(data["text_tensor"]["data_base64"])
    t1 = time.perf_counter()
    parse_time_ms = (t1 - t0) * 1000.0

    metadata = {
        "schema_version": data.get("schema_version"),
        "specimen_id": data.get("specimen_id"),
        "num_patches": data.get("patch_count"),
        "vision_dim": VISION_DIM,
        "text_dim": TEXT_DIM,
        "vision_sha256": data["vision_tensor"]["sha256"],
        "text_sha256": data["text_tensor"]["sha256"],
    }
    return metadata, v_raw, t_raw, parse_time_ms


def validate_bundle(metadata, vision_bytes, text_bytes):
    t0 = time.perf_counter()
    num_patches = metadata["num_patches"]
    v_dim = metadata["vision_dim"]
    t_dim = metadata["text_dim"]

    if v_dim != VISION_DIM or t_dim != TEXT_DIM:
        raise ValueError(f"Dimension mismatch: vision={v_dim}, text={t_dim}")
    if len(vision_bytes) != num_patches * v_dim * 4:
        raise ValueError(f"Vision byte length mismatch: {len(vision_bytes)}")
    if len(text_bytes) != t_dim * 4:
        raise ValueError(f"Text byte length mismatch: {len(text_bytes)}")

    # Checksum validation
    if "vision_hash" in metadata:
        if hashlib.sha256(vision_bytes).digest() != metadata["vision_hash"]:
            raise ValueError("Vision SHA-256 mismatch")
    elif "vision_sha256" in metadata:
        if hashlib.sha256(vision_bytes).hexdigest() != metadata["vision_sha256"]:
            raise ValueError("Vision SHA-256 mismatch")

    if "text_hash" in metadata:
        if hashlib.sha256(text_bytes).digest() != metadata["text_hash"]:
            raise ValueError("Text SHA-256 mismatch")
    elif "text_sha256" in metadata:
        if hashlib.sha256(text_bytes).hexdigest() != metadata["text_sha256"]:
            raise ValueError("Text SHA-256 mismatch")

    v_arr = np.frombuffer(vision_bytes, dtype="<f4").reshape(1, num_patches, v_dim)
    t_arr = np.frombuffer(text_bytes, dtype="<f4").reshape(1, t_dim)

    # Check finite
    if not np.isfinite(v_arr).all() or not np.isfinite(t_arr).all():
        raise ValueError("Non-finite values detected (NaN or Inf)")

    t1 = time.perf_counter()
    val_time_ms = (t1 - t0) * 1000.0
    return v_arr, t_arr, val_time_ms


def run_benchmarks():
    print(f"Loading ONNX Model: {MODEL_PATH}")
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 2
    session = ort.InferenceSession(MODEL_PATH, session_options, providers=["CPUExecutionProvider"])

    files = [f for f in os.listdir(BUNDLE_DIR) if f.endswith(".crcbundle")]
    files.sort()

    results = []

    for fname in files:
        fpath = os.path.join(BUNDLE_DIR, fname)
        file_size_bytes = os.path.getsize(fpath)

        # Benchmark Parsing (30 iterations)
        parse_times = []
        for _ in range(30):
            meta, v_bytes, t_bytes, p_ms = parse_crcb_bundle(fpath)
            parse_times.append(p_ms)
        avg_parse_ms = float(np.mean(parse_times))
        std_parse_ms = float(np.std(parse_times))

        # Benchmark Validation (30 iterations)
        val_times = []
        for _ in range(30):
            v_arr, t_arr, v_ms = validate_bundle(meta, v_bytes, t_bytes)
            val_times.append(v_ms)
        avg_val_ms = float(np.mean(val_times))
        std_val_ms = float(np.std(val_times))

        # Memory metrics
        v_bytes_calc = v_arr.nbytes
        t_bytes_calc = t_arr.nbytes
        total_tensor_mem_bytes = v_bytes_calc + t_bytes_calc

        # Warmup ONNX (10 runs)
        for _ in range(10):
            _ = session.run(None, {"vision_seq": v_arr, "text_summary": t_arr})

        # Benchmark Inference (50 runs)
        infer_times = []
        outputs = None
        for _ in range(50):
            t0 = time.perf_counter()
            outputs = session.run(None, {"vision_seq": v_arr, "text_summary": t_arr})
            t1 = time.perf_counter()
            infer_times.append((t1 - t0) * 1000.0)

        avg_infer_ms = float(np.mean(infer_times))
        std_infer_ms = float(np.std(infer_times))
        p95_infer_ms = float(np.percentile(infer_times, 95))

        # Output extraction & threshold
        logit = float(outputs[0][0][0])
        prob = float(outputs[1][0][0])
        pred_class = "MSI" if prob >= 0.83 else "MSS"

        # End-to-end latency: parse + validate + inference
        avg_e2e_ms = avg_parse_ms + avg_val_ms + avg_infer_ms

        entry = {
            "specimen_id": meta["specimen_id"],
            "bundle_file": fname,
            "format": "CRCB_v2",
            "file_size_bytes": file_size_bytes,
            "patch_count": meta["num_patches"],
            "vision_shape": [1, meta["num_patches"], VISION_DIM],
            "text_shape": [1, TEXT_DIM],
            "tensor_memory_bytes": total_tensor_mem_bytes,
            "avg_parse_time_ms": round(avg_parse_ms, 3),
            "std_parse_time_ms": round(std_parse_ms, 3),
            "avg_validation_time_ms": round(avg_val_ms, 3),
            "std_validation_time_ms": round(std_val_ms, 3),
            "avg_onnx_inference_ms": round(avg_infer_ms, 3),
            "std_onnx_inference_ms": round(std_infer_ms, 3),
            "p95_onnx_inference_ms": round(p95_infer_ms, 3),
            "avg_end_to_end_ms": round(avg_e2e_ms, 3),
            "output_logit": round(logit, 6),
            "msi_probability": round(prob, 6),
            "predicted_class": pred_class,
            "threshold": 0.83,
            "provenance": "AUTHENTIC_IMPORTED",
        }
        results.append(entry)
        print(
            f"Specimen {entry['specimen_id']} (N={entry['patch_count']}): "
            f"Parse={avg_parse_ms:.2f}ms, Val={avg_val_ms:.2f}ms, Infer={avg_infer_ms:.2f}ms, "
            f"E2E={avg_e2e_ms:.2f}ms, Prob={prob:.4f} ({pred_class})"
        )

    # Benchmark JSON format for comparison
    json_path = os.path.join(BUNDLE_DIR, "TCGA-A6-3807.crcmanifest.json")
    if os.path.exists(json_path):
        f_sz = os.path.getsize(json_path)
        meta_j, v_j, t_j, p_ms_j = parse_json_bundle(json_path)
        v_arr_j, t_arr_j, v_ms_j = validate_bundle(meta_j, v_j, t_j)
        j_infer_times = []
        for _ in range(50):
            t0 = time.perf_counter()
            outputs_j = session.run(None, {"vision_seq": v_arr_j, "text_summary": t_arr_j})
            t1 = time.perf_counter()
            j_infer_times.append((t1 - t0) * 1000.0)
        prob_j = float(outputs_j[1][0][0])
        logit_j = float(outputs_j[0][0][0])
        pred_class_j = "MSI" if prob_j >= 0.83 else "MSS"
        json_entry = {
            "specimen_id": meta_j["specimen_id"],
            "bundle_file": "TCGA-A6-3807.crcmanifest.json",
            "format": "JSON_v2",
            "file_size_bytes": f_sz,
            "patch_count": meta_j["num_patches"],
            "vision_shape": [1, meta_j["num_patches"], VISION_DIM],
            "text_shape": [1, TEXT_DIM],
            "tensor_memory_bytes": v_arr_j.nbytes + t_arr_j.nbytes,
            "avg_parse_time_ms": round(p_ms_j, 3),
            "std_parse_time_ms": 0.0,
            "avg_validation_time_ms": round(v_ms_j, 3),
            "std_validation_time_ms": 0.0,
            "avg_onnx_inference_ms": round(float(np.mean(j_infer_times)), 3),
            "std_onnx_inference_ms": round(float(np.std(j_infer_times)), 3),
            "p95_onnx_inference_ms": round(float(np.percentile(j_infer_times, 95)), 3),
            "avg_end_to_end_ms": round(p_ms_j + v_ms_j + float(np.mean(j_infer_times)), 3),
            "output_logit": round(logit_j, 6),
            "msi_probability": round(prob_j, 6),
            "predicted_class": pred_class_j,
            "threshold": 0.83,
            "provenance": "AUTHENTIC_IMPORTED",
        }
        results.append(json_entry)
        print(
            f"JSON Format (TCGA-A6-3807): Parse={p_ms_j:.2f}ms, Val={v_ms_j:.2f}ms, "
            f"Infer={json_entry['avg_onnx_inference_ms']:.2f}ms, Size={f_sz:,}B"
        )

    # Write JSON results
    payload = {
        "phase": "PHASE_5",
        "benchmark_type": "AUTHENTIC_BUNDLE_INGESTION_AND_INFERENCE",
        "model_path": MODEL_PATH,
        "model_sha256": "0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73",
        "threshold": 0.83,
        "intra_op_threads": 2,
        "execution_provider": "CPUExecutionProvider",
        "results": results,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Write CSV results
    import csv

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "specimen_id",
            "format",
            "patch_count",
            "file_size_bytes",
            "tensor_memory_bytes",
            "avg_parse_time_ms",
            "avg_validation_time_ms",
            "avg_onnx_inference_ms",
            "avg_end_to_end_ms",
            "msi_probability",
            "predicted_class",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\nSaved benchmark results to {OUTPUT_JSON} and {OUTPUT_CSV}")


if __name__ == "__main__":
    run_benchmarks()
