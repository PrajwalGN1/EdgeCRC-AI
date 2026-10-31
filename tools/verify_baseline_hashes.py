import hashlib
import json
import os

files = [
    "models/fusion_model_fp32.onnx",
    "models/fusion_model_v1_1.pth",
    "android/app/src/main/assets/models/fusion_model_fp32.onnx"
]

expected = {
    "models/fusion_model_fp32.onnx": {
        "expected_bytes": 21015452,
        "expected_sha256": "0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73"
    },
    "models/fusion_model_v1_1.pth": {
        "expected_bytes": 21007347,
        "expected_sha256": "e9a7bbcdbe1713a0ed14b6ecd511ac65d046a52e2f03c69443f5bd3dcf12d453"
    },
    "android/app/src/main/assets/models/fusion_model_fp32.onnx": {
        "expected_bytes": 21015452,
        "expected_sha256": "0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73"
    }
}

results = {}
all_match = True

for path in files:
    with open(path, "rb") as f:
        data = f.read()
    size = len(data)
    digest = hashlib.sha256(data).hexdigest()
    
    exp_info = expected[path]
    size_match = (size == exp_info["expected_bytes"])
    hash_match = (digest == exp_info["expected_sha256"])
    
    if not (size_match and hash_match):
        all_match = False
        
    results[path] = {
        "actual_bytes": size,
        "expected_bytes": exp_info["expected_bytes"],
        "bytes_match": size_match,
        "actual_sha256": digest,
        "expected_sha256": exp_info["expected_sha256"],
        "sha256_match": hash_match
    }
    print(f"Path: {path}")
    print(f"  Bytes : {size} (Match: {size_match})")
    print(f"  SHA256: {digest} (Match: {hash_match})")

print(f"\nOverall match: {all_match}")

os.makedirs("reports/mobile_deployment/phase7_tensor_first/wp7_4_cohort_evaluation", exist_ok=True)
with open("reports/mobile_deployment/phase7_tensor_first/wp7_4_cohort_evaluation/06_frozen_baseline_verification.json", "w") as f:
    json.dump({
        "all_match": all_match,
        "files": results
    }, f, indent=2)

os.makedirs("reports/mobile_deployment/phase7_tensor_first/wp7_5_security_audit", exist_ok=True)
with open("reports/mobile_deployment/phase7_tensor_first/wp7_5_security_audit/06_frozen_baseline_verification.json", "w") as f:
    json.dump({
        "all_match": all_match,
        "files": results
    }, f, indent=2)

os.makedirs("reports/mobile_deployment/phase7_tensor_first/wp7_6_performance_monitoring", exist_ok=True)
with open("reports/mobile_deployment/phase7_tensor_first/wp7_6_performance_monitoring/06_frozen_baseline_verification.json", "w") as f:
    json.dump({
        "all_match": all_match,
        "files": results
    }, f, indent=2)
print("Saved wp7_6_performance_monitoring/06_frozen_baseline_verification.json")

os.makedirs("reports/mobile_deployment/phase7_tensor_first/wp7_7_final_release", exist_ok=True)
with open("reports/mobile_deployment/phase7_tensor_first/wp7_7_final_release/06_frozen_baseline_verification.json", "w") as f:
    json.dump({
        "all_match": all_match,
        "files": results
    }, f, indent=2)
print("Saved wp7_7_final_release/06_frozen_baseline_verification.json")
