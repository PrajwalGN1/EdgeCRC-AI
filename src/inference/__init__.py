"""
Multimodal CRC Inference Package
Provides end-to-end raw-input inference combining Phikon-v2 ViT-L/16, Bio_ClinicalBERT,
and SummaryToPatchCrossAttention for MSI/MSS Colorectal Cancer Biomarker Prediction.
"""

from .input_validation import (
    InferenceValidationError,
    MissingInputError,
    InvalidImageError,
    InvalidReportError,
    CaseMismatchError,
    ValidatedCase,
    validate_multimodal_inputs,
)

from .image_preprocessing import (
    ImagePreprocessingError,
    PatchCollection,
    HistologyImagePreprocessor,
)

from .vision_encoder import (
    MissingModelWeightsError,
    BaseVisionEncoder,
    PhikonV2VisionEncoder,
    MockVisionEncoder,
)

from .report_encoder import (
    BaseReportEncoder,
    ClinicalBERTReportEncoder,
    MockReportEncoder,
)

from .multimodal_predictor import (
    MultimodalPrediction,
    MultimodalPredictor,
)

__all__ = [
    "InferenceValidationError",
    "MissingInputError",
    "InvalidImageError",
    "InvalidReportError",
    "CaseMismatchError",
    "ValidatedCase",
    "validate_multimodal_inputs",
    "ImagePreprocessingError",
    "PatchCollection",
    "HistologyImagePreprocessor",
    "MissingModelWeightsError",
    "BaseVisionEncoder",
    "PhikonV2VisionEncoder",
    "MockVisionEncoder",
    "BaseReportEncoder",
    "ClinicalBERTReportEncoder",
    "MockReportEncoder",
    "MultimodalPrediction",
    "MultimodalPredictor",
]
