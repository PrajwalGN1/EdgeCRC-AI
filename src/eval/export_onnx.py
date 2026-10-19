#!/usr/bin/env python3
"""
export_onnx.py

Phase 2 — ONNX FP32 Export Wrapper and Conversion Utilities
Provides an isolated, non-invasive export wrapper for SummaryToPatchCrossAttention.
Bypasses the unused NumPy conversion at line 48 of cross_attention_fusion.py to enable
clean PyTorch JIT tracing into an ONNX computational graph while maintaining 100%
mathematical equivalence.
"""

import sys
from pathlib import Path
from typing import Tuple, Union, Optional
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from architectures.cross_attention_fusion import SummaryToPatchCrossAttention


class CrossAttentionExportWrapper(nn.Module):
    """
    Export wrapper for SummaryToPatchCrossAttention.

    Preserves exact learned submodules and forward calculations:
    1. self.base_model.text_projection(text_summary) -> [Batch, 1, 1024]
    2. self.base_model.cross_attn(query, key, value) -> attn_output [Batch, 1, 1024], attn_weights [Batch, 1, Num_Patches]
    3. self.base_model.layer_norm(text_query + attn_output) -> [Batch, 1, 1024]
    4. Squeeze sequence dimension -> [Batch, 1024]
    5. self.base_model.classifier(pooled_features) -> [Batch, 1] (Sigmoid probability)

    Bypasses line 48:
        real_matrix = attn_weights[0].detach().cpu().numpy()
    which is unused in forward() but incompatible with PyTorch JIT tracing.
    """

    def __init__(
        self,
        base_model: SummaryToPatchCrossAttention,
        export_attention: bool = False,
        export_logit: bool = False
    ):
        super().__init__()
        self.base_model = base_model
        self.export_attention = export_attention
        self.export_logit = export_logit

    def forward(
        self,
        vision_seq: torch.Tensor,
        text_summary: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Args:
            vision_seq: FloatTensor of shape [Batch, Num_Patches, 1024] or [Num_Patches, 1024]
            text_summary: FloatTensor of shape [Batch, 768] or [Batch, 1, 768] or [768]
        Returns:
            probability: FloatTensor of shape [Batch, 1] (if not export_logit and not export_attention)
            or tuple of (logit, probability, [attn_weights]) depending on configuration.
        """
        # Ensure text summary has [Batch, 1, 768] dimension
        if text_summary.dim() == 2:
            text_summary = text_summary.unsqueeze(1)
        elif text_summary.dim() == 1:
            text_summary = text_summary.unsqueeze(0).unsqueeze(1)

        # Ensure vision sequence has [Batch, Num_Patches, 1024] dimension
        if vision_seq.dim() == 2:
            vision_seq = vision_seq.unsqueeze(0)

        # 1. Project text embedding to match vision dimension
        text_query = self.base_model.text_projection(text_summary)

        # 2. Cross-attention: Text searches image patches
        # NOTE: During standard PyTorch evaluation, passing key=vision_seq and value=vision_seq
        # preserves bit-exact identity with base_model. During ONNX tracing (torch.jit.is_tracing()),
        # passing value=vision_seq.clone() breaks object identity (key is not value), forcing
        # the TorchScript tracer to trace dynamic dimensions rather than baking static shapes into Reshape_4.
        val = vision_seq.clone() if torch.jit.is_tracing() else vision_seq
        attn_output, attn_weights = self.base_model.cross_attn(
            query=text_query,
            key=vision_seq,
            value=val
        )

        # 3. Residual connection & LayerNorm
        fused_features = self.base_model.layer_norm(text_query + attn_output)

        # 4. Squeeze sequence dimension
        pooled_features = fused_features.squeeze(1)

        # 5. Classifier MLP + Sigmoid
        if self.export_logit:
            h = self.base_model.classifier[0](pooled_features)  # Linear(1024, 256)
            h = self.base_model.classifier[1](h)                # ReLU()
            h = self.base_model.classifier[2](h)                # Dropout
            logit = self.base_model.classifier[3](h)            # Linear(256, 1)
            probability = self.base_model.classifier[4](logit)  # Sigmoid()
            if self.export_attention:
                return logit, probability, attn_weights
            return logit, probability
        else:
            probability = self.base_model.classifier(pooled_features)
            if self.export_attention:
                return probability, attn_weights
            return probability


def verify_wrapper_equivalence(
    base_model: SummaryToPatchCrossAttention,
    wrapper: CrossAttentionExportWrapper,
    patch_counts = (1, 6, 50, 100, 385, 1000, 4305),
    seed: int = 1234
) -> dict:
    """
    Mathematically verifies that base_model and wrapper produce bit-exact
    identical outputs on identical inputs across specified patch counts.
    """
    torch.manual_seed(seed)
    results = {}
    all_exact = True

    base_model.eval()
    wrapper.eval()

    for n in patch_counts:
        v = torch.randn(1, n, 1024, dtype=torch.float32)
        t = torch.randn(1, 768, dtype=torch.float32)

        with torch.no_grad():
            orig_prob, orig_attn = base_model(v, t)
            wrap_prob, wrap_attn = wrapper(v, t) if wrapper.export_attention else (wrapper(v, t), None)

        prob_diff = torch.max(torch.abs(orig_prob - wrap_prob)).item()
        attn_diff = torch.max(torch.abs(orig_attn - wrap_attn)).item() if wrap_attn is not None else 0.0

        is_exact = (prob_diff == 0.0 and attn_diff == 0.0)
        if not is_exact:
            all_exact = False

        results[f"N_{n}"] = {
            "num_patches": n,
            "orig_probability": float(orig_prob.item()),
            "wrap_probability": float(wrap_prob.item()),
            "max_prob_diff": float(prob_diff),
            "max_attn_diff": float(attn_diff),
            "bit_exact": is_exact
        }

    return {
        "all_exact_matches": all_exact,
        "tested_patch_counts": list(patch_counts),
        "per_patch_results": results
    }


def check_onnx_environment() -> dict:
    """
    Checks whether the required ONNX and ONNX Runtime packages are available.
    """
    env_status = {
        "onnx_available": False,
        "onnx_version": None,
        "onnxruntime_available": False,
        "onnxruntime_version": None
    }

    try:
        import onnx
        env_status["onnx_available"] = True
        env_status["onnx_version"] = onnx.__version__
    except ImportError:
        pass

    try:
        import onnxruntime
        env_status["onnxruntime_available"] = True
        env_status["onnxruntime_version"] = onnxruntime.__version__
    except ImportError:
        pass

    return env_status


def export_model_to_onnx(
    checkpoint_path: Path,
    output_onnx_path: Path,
    opset_version: int = 17,
    export_attention: bool = False,
    export_logit: bool = False
) -> Path:
    """
    Exports SummaryToPatchCrossAttention to an ONNX model file with dynamic patch length.
    """
    env_info = check_onnx_environment()
    if not env_info["onnx_available"]:
        raise ModuleNotFoundError(
            "Required dependency 'onnx' is not installed in the current Python environment. "
            "Please authorize installation of 'onnx' and 'onnxruntime' to proceed with Phase 2.2 export."
        )

    base_model = SummaryToPatchCrossAttention(vision_dim=1024, text_dim=768, num_heads=8)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    base_model.load_state_dict(state_dict, strict=True)
    base_model.eval()

    wrapper = CrossAttentionExportWrapper(
        base_model,
        export_attention=export_attention,
        export_logit=export_logit
    )
    wrapper.eval()

    dummy_vision = torch.randn(1, 100, 1024, dtype=torch.float32)
    dummy_text = torch.randn(1, 768, dtype=torch.float32)

    output_names = []
    dynamic_axes = {
        "vision_seq": {0: "batch_size", 1: "num_patches"},
        "text_summary": {0: "batch_size"},
    }
    if export_logit:
        output_names.append("logit")
        dynamic_axes["logit"] = {0: "batch_size"}
    output_names.append("probability")
    dynamic_axes["probability"] = {0: "batch_size"}
    if export_attention:
        output_names.append("attention_weights")
        dynamic_axes["attention_weights"] = {0: "batch_size", 2: "num_patches"}

    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (dummy_vision, dummy_text),
        str(output_onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["vision_seq", "text_summary"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False
    )

    return output_onnx_path


if __name__ == "__main__":
    CHECKPOINT = REPO_ROOT / "models" / "fusion_model_v1_1.pth"
    base = SummaryToPatchCrossAttention()
    sd = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    base.load_state_dict(sd, strict=True)
    
    wrap = CrossAttentionExportWrapper(base, export_attention=True)
    eq_res = verify_wrapper_equivalence(base, wrap)
    print("Wrapper Equivalence Check across patch counts:")
    for k, v in eq_res["per_patch_results"].items():
        print(f"  {k}: diff={v['max_prob_diff']:.2e}, bit_exact={v['bit_exact']}")
    print(f"All Exact? {eq_res['all_exact_matches']}")

    env = check_onnx_environment()
    print("\nONNX Environment Status:")
    print(f"  onnx installed: {env['onnx_available']} (version: {env['onnx_version']})")
    print(f"  onnxruntime installed: {env['onnxruntime_available']} (version: {env['onnxruntime_version']})")
