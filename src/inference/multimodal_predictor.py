"""
Multimodal Predictor Module for Colorectal Cancer Biomarker Classification.

Coordinates end-to-end inference:
1. Input validation & case association (fails closed if either modality is missing).
2. Image preprocessing & patch extraction (tiling, Otsu filtering, bounded N ≤ 1000).
3. Vision embedding extraction (Phikon-v2 ViT-L/16 -> [1, N, 1024]).
4. Report embedding extraction (Bio_ClinicalBERT -> [1, 768]).
5. Embedding shape, dtype, and finite-value sanity validation.
6. Multimodal fusion forward pass (SummaryToPatchCrossAttention + fusion_model_v1_1.pth).
7. Decision classification at validated threshold τ = 0.83 (1: MSI, 0: MSS).
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image
import torch

from .input_validation import (
    InferenceValidationError,
    MissingInputError,
    ValidatedCase,
    validate_multimodal_inputs,
)
from .image_preprocessing import (
    HistologyImagePreprocessor,
    PatchCollection,
)
from .vision_encoder import (
    BaseVisionEncoder,
    PhikonV2VisionEncoder,
    MissingModelWeightsError,
)
from .report_encoder import (
    BaseReportEncoder,
    ClinicalBERTReportEncoder,
)

logger = logging.getLogger(__name__)


@dataclass
class MultimodalPrediction:
    """Structured response container for multimodal MSI/MSS biomarker prediction."""
    case_id: str
    prediction: str            # "MSI" or "MSS"
    class_id: int              # 1 for MSI, 0 for MSS
    probability: float         # Raw model sigmoid output in [0.0, 1.0]
    confidence_pct: float      # Confidence percentage relative to call
    threshold: float           # Decision boundary (strictly 0.83)
    num_patches: int           # Number of histological patches evaluated (N)
    top_patch_indices: List[int] # Indices of patches receiving highest cross-attention
    attention_weights: Optional[np.ndarray] = None # Softmax attention vector [1, N]
    is_mock: bool = False      # Visibly flagged if any synthetic mock encoder was used
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts prediction to a serializable dictionary."""
        return {
            "case_id": self.case_id,
            "prediction": self.prediction,
            "class_id": self.class_id,
            "probability": round(self.probability, 4),
            "confidence_pct": round(self.confidence_pct, 2),
            "threshold": self.threshold,
            "num_patches": self.num_patches,
            "top_patch_indices": self.top_patch_indices,
            "is_mock": self.is_mock,
            "metadata": self.metadata,
        }


class MultimodalPredictor:
    """
    Production-grade coordinator for multimodal MSI/MSS biomarker prediction.
    """

    DECISION_THRESHOLD = 0.83
    CLASS_MAPPING = {0: "MSS", 1: "MSI"}

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        vision_encoder: Optional[BaseVisionEncoder] = None,
        report_encoder: Optional[BaseReportEncoder] = None,
        preprocessor: Optional[HistologyImagePreprocessor] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initializes the predictor, hardware binding, and fusion model checkpoint.
        
        Args:
            checkpoint_path: Path to fusion_model_v1_1.pth.
            vision_encoder: Optional pre-instantiated vision encoder (supports dependency injection for testing).
            report_encoder: Optional pre-instantiated report encoder (supports dependency injection for testing).
            preprocessor: Optional custom image preprocessor.
            device: Target torch device.
        """
        # Resolve device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Encoders & Preprocessing
        self.preprocessor = preprocessor or HistologyImagePreprocessor()
        self.vision_encoder = vision_encoder
        self.report_encoder = report_encoder

        # Resolve Checkpoint Path
        if checkpoint_path is None:
            # Auto-locate relative to project root
            repo_root = Path(__file__).resolve().parents[2]
            candidates = [
                repo_root / "models" / "fusion_model_v1_1.pth",
                repo_root / "models" / "best_fusion_model.pth",
            ]
            self.checkpoint_path = next((p for p in candidates if p.exists()), candidates[0])
        else:
            self.checkpoint_path = Path(checkpoint_path)

        # Load Fusion Model
        self.fusion_model = self._load_fusion_model()

    def _load_fusion_model(self) -> torch.nn.Module:
        """Instantiates SummaryToPatchCrossAttention and loads strict weights."""
        from ..architectures.cross_attention_fusion import SummaryToPatchCrossAttention

        model = SummaryToPatchCrossAttention(vision_dim=1024, text_dim=768, num_heads=8)
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Fusion model checkpoint not found at: {self.checkpoint_path}. "
                f"Cannot initialize MultimodalPredictor without trained weights."
            )

        try:
            state_dict = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint '{self.checkpoint_path}' into SummaryToPatchCrossAttention: {e}"
            ) from e

        model.to(self.device)
        model.eval()
        return model

    def _get_vision_encoder(self) -> BaseVisionEncoder:
        """Retrieves or instantiates the production vision encoder (Phikon-v2)."""
        if self.vision_encoder is None:
            # Lazy load production encoder
            self.vision_encoder = PhikonV2VisionEncoder(local_files_only=True, device=self.device)
        return self.vision_encoder

    def _get_report_encoder(self) -> BaseReportEncoder:
        """Retrieves or instantiates the production report encoder (ClinicalBERT)."""
        if self.report_encoder is None:
            # Lazy load production encoder
            self.report_encoder = ClinicalBERTReportEncoder(local_files_only=True, device=self.device)
        return self.report_encoder

    def predict(
        self,
        image_input: Any,
        report_input: Any,
        case_id: Optional[str] = None,
        image_case_id: Optional[str] = None,
        report_case_id: Optional[str] = None,
        reinhard_target_mean: Optional[np.ndarray] = None,
        reinhard_target_std: Optional[np.ndarray] = None,
    ) -> MultimodalPrediction:
        """
        Executes end-to-end multimodal inference from raw inputs.
        
        Args:
            image_input: Histology image (PIL, array, path, or bytes).
            report_input: Clinical pathology text string.
            case_id: Optional unified case ID.
            image_case_id: Optional image case ID.
            report_case_id: Optional report case ID.
            reinhard_target_mean: Optional stain normalization target mean.
            reinhard_target_std: Optional stain normalization target std.
            
        Returns:
            MultimodalPrediction with prediction class, probability, confidence, and metadata.
            
        Raises:
            MissingInputError: If either modality is omitted.
            InferenceValidationError: If validation fails.
            MissingModelWeightsError: If required encoder weights are missing.
        """
        latencies = {}

        # 1. Input Validation & Case Association
        t0 = time.perf_counter()
        validated_case = validate_multimodal_inputs(
            image_input=image_input,
            report_input=report_input,
            case_id=case_id,
            image_case_id=image_case_id,
            report_case_id=report_case_id,
        )
        latencies["validation_ms"] = (time.perf_counter() - t0) * 1000.0

        # 2. Image Preprocessing & Tiling
        t0 = time.perf_counter()
        patch_collection: PatchCollection = self.preprocessor.preprocess(
            image=validated_case.image,
            reinhard_target_mean=reinhard_target_mean,
            reinhard_target_std=reinhard_target_std,
        )
        latencies["image_preprocessing_ms"] = (time.perf_counter() - t0) * 1000.0
        num_patches = len(patch_collection)

        # 3. Vision Encoding (Phikon-v2 ViT-L/16)
        t0 = time.perf_counter()
        v_encoder = self._get_vision_encoder()
        vision_tensor = v_encoder.encode(patch_collection.patches, device=self.device)
        latencies["vision_encoding_ms"] = (time.perf_counter() - t0) * 1000.0

        # 4. Report Encoding (Bio_ClinicalBERT)
        t0 = time.perf_counter()
        r_encoder = self._get_report_encoder()
        text_tensor = r_encoder.encode(validated_case.report_text, device=self.device)
        latencies["report_encoding_ms"] = (time.perf_counter() - t0) * 1000.0

        # 5. Sanity Checks on Embedding Shapes and Finite Math
        self._validate_embeddings(vision_tensor, text_tensor, expected_num_patches=num_patches)

        # 6. Multimodal Fusion Forward Pass
        t0 = time.perf_counter()
        vision_tensor = vision_tensor.to(self.device)
        text_tensor = text_tensor.to(self.device)

        with torch.no_grad():
            prob_tensor, attn_weights = self.fusion_model(vision_tensor, text_tensor)

        latencies["fusion_forward_ms"] = (time.perf_counter() - t0) * 1000.0

        # 7. Classification Decision at Threshold τ = 0.83
        score = float(prob_tensor.squeeze().item())
        if np.isnan(score) or np.isinf(score):
            raise RuntimeError(f"Fusion model output non-finite probability score: {score}")

        score = max(0.0, min(1.0, score))  # Clamp strictly to [0, 1]
        is_msi = score >= self.DECISION_THRESHOLD
        class_id = 1 if is_msi else 0
        predicted_class = self.CLASS_MAPPING[class_id]

        # Clinical confidence formula:
        confidence_pct = (score * 100.0) if is_msi else ((1.0 - score) * 100.0)

        # Attention weight post-processing
        attn_np = attn_weights.squeeze().detach().cpu().numpy()
        attn_np = np.atleast_1d(attn_np)
        if attn_np.ndim > 1:
            attn_np = attn_np.reshape(-1)
        top_indices = np.argsort(attn_np)[::-1].tolist()[:min(5, len(attn_np))]

        is_mock = v_encoder.is_mock or r_encoder.is_mock

        return MultimodalPrediction(
            case_id=validated_case.case_id,
            prediction=predicted_class,
            class_id=class_id,
            probability=score,
            confidence_pct=confidence_pct,
            threshold=self.DECISION_THRESHOLD,
            num_patches=num_patches,
            top_patch_indices=top_indices,
            attention_weights=attn_np,
            is_mock=is_mock,
            metadata={
                "latencies_ms": latencies,
                "device": str(self.device),
                "checkpoint": self.checkpoint_path.name,
                "patch_metadata": patch_collection.metadata,
                "report_length_chars": len(validated_case.report_text),
                "is_mock_warning": "SYNTHETIC MOCK ENCODER USED - NOT FOR CLINICAL USE" if is_mock else None,
            },
        )

    def _validate_embeddings(
        self,
        vision_tensor: torch.Tensor,
        text_tensor: torch.Tensor,
        expected_num_patches: int,
    ) -> None:
        """Verifies tensor shapes, dtypes, and finite values before the fusion pass."""
        # Vision checks: [1, N, 1024]
        if vision_tensor.ndim != 3:
            raise ValueError(
                f"Vision tensor must be 3-dimensional [1, N, 1024], but received shape: {vision_tensor.shape}"
            )
        if vision_tensor.shape[0] != 1 or vision_tensor.shape[2] != 1024:
            raise ValueError(
                f"Invalid vision tensor shape: {vision_tensor.shape}. Expected [1, {expected_num_patches}, 1024]."
            )
        if vision_tensor.shape[1] != expected_num_patches:
            raise ValueError(
                f"Vision patch dimension mismatch: tensor has {vision_tensor.shape[1]} patches, "
                f"expected {expected_num_patches}."
            )
        if torch.isnan(vision_tensor).any() or torch.isinf(vision_tensor).any():
            raise ValueError("Vision embedding tensor contains NaN or Inf values.")

        # Text checks: [1, 768] or [1, 1, 768]
        if text_tensor.ndim not in (2, 3):
            raise ValueError(
                f"Text tensor must be 2D [1, 768] or 3D [1, 1, 768], but received shape: {text_tensor.shape}"
            )
        if text_tensor.shape[-1] != 768:
            raise ValueError(
                f"Invalid text tensor feature dimension: {text_tensor.shape[-1]}. Expected 768."
            )
        if torch.isnan(text_tensor).any() or torch.isinf(text_tensor).any():
            raise ValueError("Text embedding tensor contains NaN or Inf values.")
