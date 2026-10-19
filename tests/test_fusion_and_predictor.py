"""
Unit and integration tests for SummaryToPatchCrossAttention and MultimodalPredictor.
"""

from pathlib import Path
import pytest
import torch
import numpy as np
from PIL import Image

from src.architectures.cross_attention_fusion import SummaryToPatchCrossAttention
from src.inference.input_validation import MissingInputError, InvalidImageError, InvalidReportError
from src.inference.vision_encoder import MockVisionEncoder, MissingModelWeightsError
from src.inference.report_encoder import MockReportEncoder
from src.inference.multimodal_predictor import MultimodalPredictor, MultimodalPrediction


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = REPO_ROOT / "models" / "fusion_model_v1_1.pth"
LATE_FUSION_CHECKPOINT = REPO_ROOT / "models" / "late_fusion_model_v1_0.pth"


def test_checkpoint_strict_load_and_cpu_execution():
    """SummaryToPatchCrossAttention loads fusion_model_v1_1.pth strictly on CPU."""
    model = SummaryToPatchCrossAttention()
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Total parameters must be exactly 5,250,561
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 5250561

    # Valid forward pass test on CPU
    v = torch.randn(1, 10, 1024, dtype=torch.float32)
    t = torch.randn(1, 768, dtype=torch.float32)
    prob, attn = model(v, t)

    assert prob.shape == (1, 1)
    assert attn.shape == (1, 1, 10)
    assert 0.0 <= prob.item() <= 1.0


def test_incompatible_checkpoint_fails():
    """Attempting to load baseline late_fusion_model weights into SummaryToPatchCrossAttention must error."""
    model = SummaryToPatchCrossAttention()
    late_sd = torch.load(LATE_FUSION_CHECKPOINT, map_location="cpu", weights_only=True)
    with pytest.raises(Exception):
        model.load_state_dict(late_sd, strict=True)


def test_embedding_dimension_and_finite_checks():
    """Predictor rejects NaN or Inf embeddings, as well as invalid shapes."""
    predictor = MultimodalPredictor(
        checkpoint_path=CHECKPOINT_PATH,
        vision_encoder=MockVisionEncoder(seed=42),
        report_encoder=MockReportEncoder(seed=42),
        device="cpu",
    )

    # NaN in vision
    bad_v = torch.randn(1, 5, 1024)
    bad_v[0, 0, 0] = float("nan")
    valid_t = torch.randn(1, 768)

    with pytest.raises(ValueError, match="NaN or Inf"):
        predictor._validate_embeddings(bad_v, valid_t, expected_num_patches=5)

    # Inf in text
    valid_v = torch.randn(1, 5, 1024)
    bad_t = torch.randn(1, 768)
    bad_t[0, 0] = float("inf")

    with pytest.raises(ValueError, match="NaN or Inf"):
        predictor._validate_embeddings(valid_v, bad_t, expected_num_patches=5)

    # Wrong vision dim (512 instead of 1024)
    wrong_dim_v = torch.randn(1, 5, 512)
    with pytest.raises(ValueError, match="Invalid vision tensor shape"):
        predictor._validate_embeddings(wrong_dim_v, valid_t, expected_num_patches=5)

    # Wrong text dim (512 instead of 768)
    wrong_dim_t = torch.randn(1, 512)
    with pytest.raises(ValueError, match="Invalid text tensor feature dimension"):
        predictor._validate_embeddings(valid_v, wrong_dim_t, expected_num_patches=5)


def test_threshold_and_class_mapping():
    """Threshold τ = 0.83 strictly maps probabilities: P >= 0.83 -> MSI (1), P < 0.83 -> MSS (0)."""
    predictor = MultimodalPredictor(
        checkpoint_path=CHECKPOINT_PATH,
        vision_encoder=MockVisionEncoder(seed=42),
        report_encoder=MockReportEncoder(seed=42),
        device="cpu",
    )

    assert predictor.DECISION_THRESHOLD == 0.83
    assert predictor.CLASS_MAPPING[0] == "MSS"
    assert predictor.CLASS_MAPPING[1] == "MSI"


def test_end_to_end_mock_prediction():
    """
    Integration test with injected mock encoders:
    Validates end-to-end chaining of raw image + text through preprocessing,
    mock encoding, and real fusion checkpoint.
    """
    predictor = MultimodalPredictor(
        checkpoint_path=CHECKPOINT_PATH,
        vision_encoder=MockVisionEncoder(seed=42),
        report_encoder=MockReportEncoder(seed=42),
        device="cpu",
    )

    # Create synthetic 224x224 RGB image
    img = Image.fromarray(np.random.randint(40, 160, size=(224, 224, 3), dtype=np.uint8))
    text = "Surgical pathology report: adenocarcinoma in ascending colon, right hemicolectomy."

    result = predictor.predict(image_input=img, report_input=text, case_id="CASE-UNIT-01")

    assert isinstance(result, MultimodalPrediction)
    assert result.case_id == "CASE-UNIT-01"
    assert result.prediction in ("MSI", "MSS")
    assert result.class_id in (0, 1)
    assert 0.0 <= result.probability <= 1.0
    assert 0.0 <= result.confidence_pct <= 100.0
    assert result.num_patches == 1
    assert result.threshold == 0.83
    assert len(result.top_patch_indices) == 1
    assert result.is_mock is True  # Visibly marked as mock
    assert "latencies_ms" in result.metadata
    assert "image_preprocessing_ms" in result.metadata["latencies_ms"]
    assert "fusion_forward_ms" in result.metadata["latencies_ms"]


def test_missing_input_rejection():
    """Predictor rejects missing image or missing report immediately."""
    predictor = MultimodalPredictor(
        checkpoint_path=CHECKPOINT_PATH,
        vision_encoder=MockVisionEncoder(seed=42),
        report_encoder=MockReportEncoder(seed=42),
        device="cpu",
    )

    img = Image.fromarray(np.random.randint(40, 160, size=(224, 224, 3), dtype=np.uint8))
    text = "Surgical pathology report text for test case."

    with pytest.raises(MissingInputError):
        predictor.predict(image_input=None, report_input=text)

    with pytest.raises(MissingInputError):
        predictor.predict(image_input=img, report_input=None)


def test_production_mode_fails_closed_without_weights():
    """
    In production mode (no mock injected), predictor attempts to load real Phikon-v2
    and ClinicalBERT, and must fail closed with MissingModelWeightsError.
    It must NEVER return a fabricated medical prediction.
    """
    predictor = MultimodalPredictor(
        checkpoint_path=CHECKPOINT_PATH,
        vision_encoder=None,  # Production mode
        report_encoder=None,  # Production mode
        device="cpu",
    )

    img = Image.fromarray(np.random.randint(40, 160, size=(224, 224, 3), dtype=np.uint8))
    text = "Surgical pathology report text for test case."

    with pytest.raises(MissingModelWeightsError):
        predictor.predict(image_input=img, report_input=text)
