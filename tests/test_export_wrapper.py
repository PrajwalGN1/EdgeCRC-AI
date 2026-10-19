import pytest
import torch
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from architectures.cross_attention_fusion import SummaryToPatchCrossAttention
from eval.export_onnx import CrossAttentionExportWrapper, verify_wrapper_equivalence, check_onnx_environment

CHECKPOINT_PATH = REPO_ROOT / "models" / "fusion_model_v1_1.pth"


@pytest.fixture(scope="module")
def loaded_models():
    base_model = SummaryToPatchCrossAttention(vision_dim=1024, text_dim=768, num_heads=8)
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    base_model.load_state_dict(state_dict, strict=True)
    base_model.eval()

    wrapper = CrossAttentionExportWrapper(base_model, export_attention=True)
    wrapper.eval()
    return base_model, wrapper


def test_wrapper_equivalence_across_patch_counts(loaded_models):
    base_model, wrapper = loaded_models
    res = verify_wrapper_equivalence(
        base_model,
        wrapper,
        patch_counts=(1, 6, 50, 100, 385, 1000, 4305)
    )
    assert res["all_exact_matches"] is True
    for patch_key, data in res["per_patch_results"].items():
        assert data["bit_exact"] is True
        assert data["max_prob_diff"] == 0.0
        assert data["max_attn_diff"] == 0.0


def test_wrapper_single_output_mode(loaded_models):
    base_model, _ = loaded_models
    wrapper_prob_only = CrossAttentionExportWrapper(base_model, export_attention=False)
    wrapper_prob_only.eval()

    dummy_v = torch.randn(1, 100, 1024)
    dummy_t = torch.randn(1, 768)

    with torch.no_grad():
        out = wrapper_prob_only(dummy_v, dummy_t)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 1)


def test_onnx_env_check():
    env = check_onnx_environment()
    assert isinstance(env, dict)
    assert "onnx_available" in env
    assert "onnxruntime_available" in env
