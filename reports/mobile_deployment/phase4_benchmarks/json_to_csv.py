import json
import csv
from pathlib import Path

json_path = Path("reports/mobile_deployment/phase4_benchmarks/benchmark_results.json")
csv_path = Path("reports/mobile_deployment/phase4_benchmarks/benchmark_results.csv")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

fieldnames = [
    "case_id",
    "num_patches",
    "thread_count",
    "session_init_ms",
    "tensor_parse_ms",
    "stage_d_alloc_median_ms",
    "stage_d_alloc_mean_ms",
    "stage_e_purerun_min_ms",
    "stage_e_purerun_median_ms",
    "stage_e_purerun_p90_ms",
    "stage_e_purerun_p95_ms",
    "stage_e_purerun_p99_ms",
    "stage_e_purerun_mean_ms",
    "stage_e_purerun_stddev_ms",
    "stage_f_decode_median_ms",
    "stage_f_decode_mean_ms",
    "stage_g_pipeline_min_ms",
    "stage_g_pipeline_median_ms",
    "stage_g_pipeline_p90_ms",
    "stage_g_pipeline_p95_ms",
    "stage_g_pipeline_mean_ms",
    "on_device_logit",
    "on_device_prob",
    "ref_logit",
    "ref_prob",
    "diff_logit",
    "diff_prob",
    "parity_pass"
]

rows = []
for entry in data:
    rows.append({
        "case_id": entry["caseId"],
        "num_patches": entry["numPatches"],
        "thread_count": entry["threadCount"],
        "session_init_ms": entry["sessionInitMs"],
        "tensor_parse_ms": entry["tensorParseMs"],
        "stage_d_alloc_median_ms": entry["stageD_allocation"]["medianMs"],
        "stage_d_alloc_mean_ms": entry["stageD_allocation"]["meanMs"],
        "stage_e_purerun_min_ms": entry["stageE_pureRun"]["minMs"],
        "stage_e_purerun_median_ms": entry["stageE_pureRun"]["medianMs"],
        "stage_e_purerun_p90_ms": entry["stageE_pureRun"]["p90Ms"],
        "stage_e_purerun_p95_ms": entry["stageE_pureRun"]["p95Ms"],
        "stage_e_purerun_p99_ms": entry["stageE_pureRun"]["p99Ms"],
        "stage_e_purerun_mean_ms": entry["stageE_pureRun"]["meanMs"],
        "stage_e_purerun_stddev_ms": entry["stageE_pureRun"]["stdDevMs"],
        "stage_f_decode_median_ms": entry["stageF_outputDecode"]["medianMs"],
        "stage_f_decode_mean_ms": entry["stageF_outputDecode"]["meanMs"],
        "stage_g_pipeline_min_ms": entry["stageG_pipeline"]["minMs"],
        "stage_g_pipeline_median_ms": entry["stageG_pipeline"]["medianMs"],
        "stage_g_pipeline_p90_ms": entry["stageG_pipeline"]["p90Ms"],
        "stage_g_pipeline_p95_ms": entry["stageG_pipeline"]["p95Ms"],
        "stage_g_pipeline_mean_ms": entry["stageG_pipeline"]["meanMs"],
        "on_device_logit": entry["onDeviceLogit"],
        "on_device_prob": entry["onDeviceProb"],
        "ref_logit": entry["refLogit"],
        "ref_prob": entry["refProb"],
        "diff_logit": entry["diffLogit"],
        "diff_prob": entry["diffProb"],
        "parity_pass": entry["parityPass"]
    })

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} records to {csv_path}")
