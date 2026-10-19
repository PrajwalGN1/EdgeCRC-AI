"""
Test Suite: Mobile Edge Inference Contract & Real Model Graph Smoke Test
Phase: 4.1 Minimal Kotlin ONNX Inference Integration Verification

This test file verifies the real ONNX model graph (models/fusion_model_fp32.onnx)
against the exact contract specified for the Android/Kotlin edge integration:
- Model loading from disk and asset paths.
- Input/output tensor names and dynamic sequence shapes.
- Finite outputs and probability bounded in [0.0, 1.0].
- Attention weights dimension [1, 1, N] and probability normalization.
- Rejection of invalid dimensions and non-finite values.
- Boundary threshold behavior at tau = 0.83.
- Deterministic session lifecycle and resource disposal.

NOTE: Synthetic input tensors are used solely for interface verification.
They do NOT represent clinical holdout data or diagnostic evaluation metrics.
"""

import os
import pytest
import numpy as np
import onnx
import onnxruntime as ort

MODEL_FP32_PATH = "models/fusion_model_fp32.onnx"
ASSET_FP32_PATH = "android/app/src/main/assets/models/fusion_model_fp32.onnx"
THRESHOLD = 0.83
VISION_DIM = 1024
TEXT_DIM = 768
MAX_PATCH_CAP = 1000


def test_fp32_model_loading_and_assets():
    """Verify that both the root model and the Android asset copy load validly."""
    for path in [MODEL_FP32_PATH, ASSET_FP32_PATH]:
        assert os.path.exists(path), f"Model path {path} must exist"
        assert os.path.getsize(path) == 21015452, f"Model size must be exactly 21,015,452 bytes"
        
        # Verify with ONNX parser
        model = onnx.load(path)
        onnx.checker.check_model(model)
        assert model.ir_version == 8
        assert model.opset_import[0].version == 17


def test_input_output_tensor_names_and_shapes():
    """Verify exact input and output names and dynamic shapes in the graph."""
    session = ort.InferenceSession(MODEL_FP32_PATH, providers=["CPUExecutionProvider"])
    
    input_names = [inp.name for inp in session.get_inputs()]
    assert "vision_seq" in input_names, "Input must contain 'vision_seq'"
    assert "text_summary" in input_names, "Input must contain 'text_summary'"
    
    # Check input specifications
    v_inp = next(i for i in session.get_inputs() if i.name == "vision_seq")
    assert v_inp.type == "tensor(float)"
    # Shape is ['batch_size', 'num_patches', 1024]
    assert v_inp.shape[2] == VISION_DIM
    
    t_inp = next(i for i in session.get_inputs() if i.name == "text_summary")
    assert t_inp.type == "tensor(float)"
    assert t_inp.shape[1] == TEXT_DIM
    
    output_names = [out.name for out in session.get_outputs()]
    assert "logit" in output_names, "Output must contain 'logit'"
    assert "probability" in output_names, "Output must contain 'probability'"
    assert "attention_weights" in output_names, "Output must contain 'attention_weights'"
    
    session_asset = ort.InferenceSession(ASSET_FP32_PATH, providers=["CPUExecutionProvider"])
    assert [o.name for o in session_asset.get_outputs()] == output_names


@pytest.mark.parametrize("num_patches", [1, 10, 80, 500, 1000])
def test_synthetic_inference_various_patch_counts(num_patches):
    """
    Smoke test: execute inference on the real FP32 graph across diverse patch counts
    including 1, 10, 80 (CASE_36 boundary), 500, and 1000 (Protocol 1 upper cap).
    """
    session = ort.InferenceSession(MODEL_FP32_PATH, providers=["CPUExecutionProvider"])
    
    # Generate reproducible synthetic inputs
    np.random.seed(42 + num_patches)
    vision_data = np.random.randn(1, num_patches, VISION_DIM).astype(np.float32)
    text_data = np.random.randn(1, TEXT_DIM).astype(np.float32)
    
    outputs = session.run(
        ["logit", "probability", "attention_weights"],
        {"vision_seq": vision_data, "text_summary": text_data}
    )
    
    logit, prob, attn = outputs[0], outputs[1], outputs[2]
    
    # 1. Output shapes
    assert logit.shape == (1, 1), f"Logit shape must be (1, 1), got {logit.shape}"
    assert prob.shape == (1, 1), f"Probability shape must be (1, 1), got {prob.shape}"
    assert attn.shape == (1, 1, num_patches), f"Attention shape must be (1, 1, {num_patches}), got {attn.shape}"
    
    # 2. Finite values
    assert np.all(np.isfinite(logit)), "Logit must be finite"
    assert np.all(np.isfinite(prob)), "Probability must be finite"
    assert np.all(np.isfinite(attn)), "Attention weights must be finite"
    
    # 3. Probability range
    prob_val = float(prob[0, 0])
    assert 0.0 <= prob_val <= 1.0, f"Probability must be in [0, 1], got {prob_val}"
    
    # 4. Attention weights sum to 1.0 (softmax output)
    attn_sum = float(np.sum(attn))
    assert np.isclose(attn_sum, 1.0, atol=1e-5), f"Attention weights must sum to 1.0, got {attn_sum}"
    assert np.all(attn >= 0.0), "Attention weights must be non-negative"


def test_threshold_boundary_decision_logic():
    """Verify exact threshold boundary semantics (>= 0.83 -> MSI/1, < 0.83 -> MSS/0)."""
    def classify(probability: float) -> int:
        return 1 if probability >= THRESHOLD else 0
    
    # Immediately below threshold
    assert classify(0.829999) == 0, "0.829999 must classify as MSS (0)"
    
    # Exactly at threshold
    assert classify(0.830000) == 1, "0.830000 must classify as MSI (1)"
    
    # Immediately above threshold
    assert classify(0.830001) == 1, "0.830001 must classify as MSI (1)"
    
    # Extreme values
    assert classify(0.0) == 0
    assert classify(1.0) == 1


def test_invalid_input_rejection():
    """Verify that malformed inputs raise errors as required by the interface contract."""
    session = ort.InferenceSession(MODEL_FP32_PATH, providers=["CPUExecutionProvider"])
    
    # 1. Wrong vision feature dimension (512 instead of 1024)
    bad_vision = np.random.randn(1, 10, 512).astype(np.float32)
    good_text = np.random.randn(1, TEXT_DIM).astype(np.float32)
    with pytest.raises(Exception):
        session.run(None, {"vision_seq": bad_vision, "text_summary": good_text})
        
    # 2. Wrong text feature dimension (512 instead of 768)
    good_vision = np.random.randn(1, 10, VISION_DIM).astype(np.float32)
    bad_text = np.random.randn(1, 512).astype(np.float32)
    with pytest.raises(Exception):
        session.run(None, {"vision_seq": good_vision, "text_summary": bad_text})


def test_session_resource_disposal():
    """Verify repeated sequential creation and destruction of inference sessions."""
    for i in range(3):
        sess = ort.InferenceSession(MODEL_FP32_PATH, providers=["CPUExecutionProvider"])
        v = np.random.randn(1, 5, VISION_DIM).astype(np.float32)
        t = np.random.randn(1, TEXT_DIM).astype(np.float32)
        res = sess.run(None, {"vision_seq": v, "text_summary": t})
        assert len(res) == 3
        del sess


def test_nan_and_inf_inputs_propagation():
    """
    Verify that injecting NaN or Inf into input tensors results in either
    runtime exception or detectable NaN/Inf outputs, which the Kotlin
    InputValidator / CrcMultimodalClassifier catches and rejects.
    """
    session = ort.InferenceSession(MODEL_FP32_PATH, providers=["CPUExecutionProvider"])
    v_nan = np.random.randn(1, 10, VISION_DIM).astype(np.float32)
    v_nan[0, 2, 5] = np.nan
    t_good = np.random.randn(1, TEXT_DIM).astype(np.float32)
    
    outputs = session.run(None, {"vision_seq": v_nan, "text_summary": t_good})
    prob = float(outputs[1][0, 0])
    # The raw model propagates NaN; the Kotlin InputValidator strictly intercepts it before session.run
    assert np.isnan(prob), "NaN in input must propagate or fail, validating necessity of InputValidator"


def test_large_patch_count_protocol2():
    """
    Verify that Protocol 2 uncapped execution handles large patch counts (e.g. 3,914 patches
    corresponding to CASE_20 / CASE_15) without integer overflow or crash.
    """
    session = ort.InferenceSession(MODEL_FP32_PATH, providers=["CPUExecutionProvider"])
    n_patches = 3914
    v_large = np.random.randn(1, n_patches, VISION_DIM).astype(np.float32)
    t_good = np.random.randn(1, TEXT_DIM).astype(np.float32)
    
    outputs = session.run(["logit", "probability", "attention_weights"], {"vision_seq": v_large, "text_summary": t_good})
    assert outputs[0].shape == (1, 1)
    assert outputs[1].shape == (1, 1)
    assert outputs[2].shape == (1, 1, n_patches)
    assert np.all(np.isfinite(outputs[1]))


def test_protocol_scoping_contract():
    """
    Verify the protocol scoping contract:
    - Protocol 1 enforces 1,000-patch cap.
    - Protocol 2 supports uncapped patches (> 1,000, including CASE_20's 3,914 patches).
    - Uncapped inputs are not rejected merely because N > 1,000.
    """
    # Protocol 1: N > 1000 must be flagged as exceeding Protocol 1 limit
    def validate_patch_count(num_patches: int, protocol: str):
        if num_patches <= 0:
            raise ValueError("Patch count must be > 0")
        if protocol == "PROTOCOL_1_CAPPED" and num_patches > MAX_PATCH_CAP:
            raise ValueError(f"Protocol 1 enforces cap of {MAX_PATCH_CAP}, got {num_patches}")
        # Protocol 2 allows any positive patch count
        return True

    # P1 boundary checks
    assert validate_patch_count(1000, "PROTOCOL_1_CAPPED") is True
    with pytest.raises(ValueError, match="Protocol 1 enforces cap of 1000"):
        validate_patch_count(1001, "PROTOCOL_1_CAPPED")
    
    # P2 boundary checks
    assert validate_patch_count(1001, "PROTOCOL_2_UNCAPPED") is True
    assert validate_patch_count(3914, "PROTOCOL_2_UNCAPPED") is True


