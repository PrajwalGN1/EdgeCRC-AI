import pytest
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
import json

REPO_ROOT = Path(__file__).resolve().parents[1]
FP16_PATH = REPO_ROOT / "models" / "fusion_model_fp16.onnx"
FP32_PATH = REPO_ROOT / "models" / "fusion_model_fp32.onnx"
PARITY_PATH = REPO_ROOT / "reports" / "mobile_deployment" / "phase3_fp16_benchmark_data.json"


@pytest.fixture(scope="module")
def onnx_session_fp16():
    assert FP16_PATH.exists(), f"FP16 model missing at {FP16_PATH}"
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    session = ort.InferenceSession(str(FP16_PATH), sess_opts, providers=["CPUExecutionProvider"])
    return session


def test_fp16_model_storage_reduction():
    assert FP16_PATH.exists()
    assert FP32_PATH.exists()
    fp32_size = FP32_PATH.stat().st_size
    fp16_size = FP16_PATH.stat().st_size
    reduction_pct = ((fp32_size - fp16_size) / fp32_size) * 100.0
    assert 10_000_000 <= fp16_size <= 11_000_000, f"Unexpected FP16 size: {fp16_size}"
    assert 49.0 <= reduction_pct <= 51.0, f"Expected ~50% reduction, got {reduction_pct:.2f}%"


def test_fp16_model_graph_validity():
    model = onnx.load(str(FP16_PATH))
    onnx.checker.check_model(model)
    inputs = {i.name: i.type.tensor_type.elem_type for i in model.graph.input}
    outputs = {o.name: o.type.tensor_type.elem_type for o in model.graph.output}
    
    assert inputs["vision_seq"] == onnx.TensorProto.FLOAT16
    assert inputs["text_summary"] == onnx.TensorProto.FLOAT16
    assert outputs["logit"] == onnx.TensorProto.FLOAT16
    assert outputs["probability"] == onnx.TensorProto.FLOAT16
    assert outputs["attention_weights"] == onnx.TensorProto.FLOAT16


@pytest.mark.parametrize("n_patches", [1, 6, 50, 100, 385, 1000, 4305])
def test_fp16_dynamic_sequence_execution(onnx_session_fp16, n_patches):
    np.random.seed(200 + n_patches)
    v16 = np.random.randn(1, n_patches, 1024).astype(np.float16)
    t16 = np.random.randn(1, 768).astype(np.float16)
    
    outs = onnx_session_fp16.run(None, {"vision_seq": v16, "text_summary": t16})
    assert len(outs) == 3
    logit, prob, attn = outs[0], outs[1], outs[2]
    
    assert logit.shape == (1, 1)
    assert prob.shape == (1, 1)
    assert attn.shape == (1, 1, n_patches)
    assert 0.0 <= float(prob[0][0]) <= 1.0


def test_fp16_cohort_parity_and_zero_flips():
    assert PARITY_PATH.exists()
    data = json.load(open(PARITY_PATH, encoding="utf-8"))
    
    p1 = data["protocol_1"]
    p2 = data["protocol_2"]
    
    assert p1["summary"]["flips_vs_fp32"] == 0, "Protocol 1 has unexpected classification flips vs FP32"
    assert p2["summary"]["flips_vs_fp32"] == 0, "Protocol 2 has unexpected classification flips vs FP32"
    
    assert p1["summary"]["max_prob_drift"] <= 5.0e-4
    assert p2["summary"]["max_prob_drift"] <= 5.0e-4
    assert p1["summary"]["mean_prob_drift"] <= 1.0e-4
    assert p2["summary"]["mean_prob_drift"] <= 1.0e-4


def test_fp16_critical_boundary_case_36():
    data = json.load(open(PARITY_PATH, encoding="utf-8"))
    c36_p1 = next(c for c in data["protocol_1"]["cases"] if c["deidentified_id"] == "CASE_36")
    c36_p2 = next(c for c in data["protocol_2"]["cases"] if c["deidentified_id"] == "CASE_36")
    
    assert c36_p1["fp16_call"] == 1, "CASE_36 flipped from MSI to MSS in Protocol 1!"
    assert c36_p2["fp16_call"] == 1, "CASE_36 flipped from MSI to MSS in Protocol 2!"
    assert c36_p1["fp16_prob"] >= 0.83, f"CASE_36 FP16 prob ({c36_p1['fp16_prob']}) dropped below 0.83!"
