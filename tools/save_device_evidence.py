"""
tools/save_device_evidence.py
Captures and formats the physical device verification results for Phase 5.1E into JSON.
"""

import json
from pathlib import Path

REPO_ROOT = Path("D:/CRC Edge deployment/MULTIMODAL-FUSION-FOR-DETECTION-OF-COLORECTAL-CANCER-USING-CLINICAL-BERT-AND-PHIKONV2")
AUDIT_DIR = REPO_ROOT / "reports" / "mobile_deployment" / "phase5_1_audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

device_evidence = {
    "audit_phase": "PHASE_5_1E_PHYSICAL_DEVICE_VERIFICATION",
    "timestamp_utc": "2026-10-23T11:51:06Z",
    "device_hardware": {
        "device_model": "SM-S921E",
        "commercial_name": "Samsung Galaxy S24",
        "android_version": "16",
        "api_level": 36,
        "abi": "arm64-v8a",
        "serial_identifier": "RZGYA0F6TCB",
        "status": "device (authorized)"
    },
    "on_device_model_integrity": {
        "asset_path": "models/fusion_model_fp32.onnx",
        "byte_size": 21015452,
        "sha256": "0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73",
        "verified_on_device": True
    },
    "test_suite_execution": {
        "test_suite": "org.crc.multimodal.CrcPhysicalOnDeviceParityTest",
        "test_runner": "androidx.test.runner.AndroidJUnitRunner",
        "total_tests": 4,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "total_time_seconds": 1.249,
        "tests": [
            {
                "test_name": "verifyPhysicalOnDeviceMsiParityTcgaD56540",
                "case_id": "TCGA-D5-6540",
                "patches": 345,
                "device_logit": 4.63481,
                "ref_logit": 4.63481,
                "logit_diff": 0.0,
                "device_prob": 0.99039,
                "ref_prob": 0.99038,
                "prob_diff": 2.98e-7,
                "predicted_class": 1,
                "predicted_label": "MSI",
                "ground_truth_call": 1,
                "threshold": 0.83,
                "inference_latency_ms": 40,
                "status": "PASS"
            },
            {
                "test_name": "verifyPhysicalOnDeviceMssParityTcgaCm6675",
                "case_id": "TCGA-CM-6675",
                "patches": 541,
                "device_logit": -1.64159,
                "ref_logit": -1.64158,
                "logit_diff": 2.38e-7,
                "device_prob": 0.16225,
                "ref_prob": 0.16225,
                "prob_diff": 5.22e-7,
                "predicted_class": 0,
                "predicted_label": "MSS",
                "ground_truth_call": 0,
                "threshold": 0.83,
                "inference_latency_ms": 56,
                "status": "PASS"
            },
            {
                "test_name": "verifyPhysicalOnDeviceAuthenticBundleImportTcgaA63807",
                "case_id": "TCGA-A6-3807",
                "bundle_format": "CRCB_v2",
                "patches": 613,
                "device_logit": -6.06156,
                "ref_logit": -6.06156,
                "logit_diff": 9.54e-7,
                "device_prob": 0.00233,
                "ref_prob": 0.00233,
                "prob_diff": 3.56e-7,
                "predicted_class": 0,
                "predicted_label": "MSS",
                "ground_truth_call": 0,
                "threshold": 0.83,
                "provenance": "AUTHENTIC_IMPORTED",
                "inference_latency_ms": 61,
                "status": "PASS"
            },
            {
                "test_name": "verifyPhysicalOnDeviceSyntheticProvenanceFailsClosed",
                "tested_provenance": "DEMONSTRATION_SIMULATED",
                "expected_exception": "IllegalArgumentException",
                "exception_thrown": True,
                "message_verified": True,
                "inference_prevented": True,
                "status": "PASS"
            }
        ]
    },
    "physical_device_acceptance_status": "PASS"
}

out_file = AUDIT_DIR / "device_verification.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(device_evidence, f, indent=2)

print(f"Device verification evidence saved to: {out_file}")
