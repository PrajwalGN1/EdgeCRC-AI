import pytest
import numpy as np
import torch
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from architectures.cross_attention_fusion import SummaryToPatchCrossAttention
from eval.export_onnx import CrossAttentionExportWrapper, check_onnx_environment

CHECKPOINT_PATH = REPO_ROOT / "models" / "fusion_model_v1_1.pth"
ONNX_MODEL_PATH = REPO_ROOT / "models" / "fusion_model_fp32.onnx"


@pytest.fixture(scope="module")
def onnx_session():
    import onnxruntime as ort
    assert ONNX_MODEL_PATH.exists(), f"ONNX model missing at {ONNX_MODEL_PATH}"
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    session = ort.InferenceSession(str(ONNX_MODEL_PATH), sess_opts, providers=["CPUExecutionProvider"])
    return session


@pytest.fixture(scope="module")
def wrapper_model():
    base_model = SummaryToPatchCrossAttention(vision_dim=1024, text_dim=768, num_heads=8)
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    base_model.load_state_dict(state_dict, strict=True)
    base_model.eval()

    wrapper = CrossAttentionExportWrapper(base_model, export_attention=True, export_logit=True)
    wrapper.eval()
    return wrapper


def test_onnx_model_file_properties():
    assert ONNX_MODEL_PATH.exists()
    size_bytes = ONNX_MODEL_PATH.stat().st_size
    # ~21.0 MB (5,250,561 float32 params * 4 bytes + proto overhead)
    assert 20_000_000 <= size_bytes <= 22_000_000


def test_onnx_model_graph_validity():
    import onnx
    model = onnx.load(str(ONNX_MODEL_PATH))
    onnx.checker.check_model(model)

    input_names = [i.name for i in model.graph.input]
    output_names = [o.name for o in model.graph.output]
    assert input_names == ["vision_seq", "text_summary"]
    assert output_names == ["logit", "probability", "attention_weights"]


@pytest.mark.parametrize("n_patches", [1, 6, 50, 100, 385, 1000, 4305])
def test_onnx_dynamic_shape_parity(onnx_session, wrapper_model, n_patches):
    torch.manual_seed(100 + n_patches)
    v = torch.randn(1, n_patches, 1024, dtype=torch.float32)
    t = torch.randn(1, 768, dtype=torch.float32)

    with torch.no_grad():
        ref_logit, ref_prob, ref_attn = wrapper_model(v, t)

    v_np = v.numpy()
    t_np = t.numpy()

    ort_outs = onnx_session.run(
        ["logit", "probability", "attention_weights"],
        {"vision_seq": v_np, "text_summary": t_np}
    )
    onnx_logit = ort_outs[0].item()
    onnx_prob = ort_outs[1].item()
    onnx_attn = ort_outs[2]

    # Approved FP32 tolerances
    logit_err = abs(onnx_logit - ref_logit.item())
    prob_err = abs(onnx_prob - ref_prob.item())
    attn_err = float(np.max(np.abs(onnx_attn - ref_attn.numpy())))

    assert logit_err <= 1e-4, f"N={n_patches}: logit error {logit_err} exceeds 1e-4"
    assert prob_err <= 1e-4, f"N={n_patches}: prob error {prob_err} exceeds 1e-4"
    assert attn_err <= 1e-4, f"N={n_patches}: attn error {attn_err} exceeds 1e-4"
