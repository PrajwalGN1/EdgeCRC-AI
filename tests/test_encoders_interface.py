"""
Unit tests for vision and report encoder interfaces.
"""

import pytest
import torch
from PIL import Image

from src.inference.vision_encoder import (
    BaseVisionEncoder,
    PhikonV2VisionEncoder,
    MockVisionEncoder,
    MissingModelWeightsError as VisionWeightsError,
)
from src.inference.report_encoder import (
    BaseReportEncoder,
    ClinicalBERTReportEncoder,
    MockReportEncoder,
    MissingModelWeightsError as ReportWeightsError,
)


def test_mock_vision_encoder():
    """Mock vision encoder outputs [1, N, 1024] float32 tensor without NaNs."""
    encoder = MockVisionEncoder(seed=42)
    assert encoder.embedding_dim == 1024
    assert encoder.is_mock is True

    # 3 dummy patches
    patches = [Image.new("RGB", (224, 224), color=(100, 50, 150)) for _ in range(3)]
    tensor = encoder.encode(patches)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 1024)
    assert tensor.dtype == torch.float32
    assert not torch.isnan(tensor).any()
    assert not torch.isinf(tensor).any()


def test_mock_report_encoder():
    """Mock report encoder outputs [1, 768] float32 tensor without NaNs."""
    encoder = MockReportEncoder(seed=42)
    assert encoder.embedding_dim == 768
    assert encoder.is_mock is True

    text = "Colon biopsy reveals adenocarcinoma with mucinous components."
    tensor = encoder.encode(text)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 768)
    assert tensor.dtype == torch.float32
    assert not torch.isnan(tensor).any()
    assert not torch.isinf(tensor).any()


def test_phikon_missing_weights_fails_closed():
    """
    Attempting to instantiate the real Phikon-v2 encoder when weights are missing
    must raise MissingModelWeightsError (failing closed).
    """
    with pytest.raises(VisionWeightsError, match="Phikon-v2"):
        PhikonV2VisionEncoder(local_files_only=True)


def test_clinicalbert_missing_weights_fails_closed():
    """
    Attempting to instantiate the real ClinicalBERT encoder when weights are incomplete
    must raise MissingModelWeightsError (failing closed).
    """
    with pytest.raises(ReportWeightsError, match="Bio_ClinicalBERT"):
        ClinicalBERTReportEncoder(local_files_only=True)
