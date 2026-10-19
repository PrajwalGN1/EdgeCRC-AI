"""
Vision Encoder Module for Multimodal CRC Histopathology Inference.

Integrates Phikon-v2 (DINOv2 ViT-L/16, 307M parameters) to extract 1024-dimensional
patch CLS embeddings matching the precomputed tensor store.

Enforces:
- Exact feature extraction logic matching senior notebook `tcga-coad (1).ipynb`.
- Bounded-memory sub-batching for edge and CPU inference.
- Strict fail-closed error handling when model weights are not cached locally.
- Explicit separation between production Phikon-v2 encoder and test-only mock encoders.
"""

from abc import ABC, abstractmethod
import logging
from typing import List, Optional, Union

import numpy as np
from PIL import Image
import torch

logger = logging.getLogger(__name__)


class MissingModelWeightsError(RuntimeError):
    """Raised when pretrained foundation encoder weights are missing from local storage."""
    pass


class BaseVisionEncoder(ABC):
    """Abstract base class for vision encoders."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Returns the output feature embedding dimension (expected: 1024)."""
        pass

    @property
    @abstractmethod
    def is_mock(self) -> bool:
        """Indicates whether this encoder is a synthetic mock for testing."""
        pass

    @abstractmethod
    def encode(
        self,
        patches: List[Image.Image],
        batch_size: int = 16,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Encodes a list of 224x224 RGB image patches into a pooled feature tensor.
        
        Args:
            patches: List of PIL images (each 224x224).
            batch_size: Sub-batch size to bound peak memory.
            device: Target torch device (cuda, mps, or cpu).
            
        Returns:
            torch.Tensor of shape [1, N, 1024] and dtype torch.float32.
        """
        pass


class PhikonV2VisionEncoder(BaseVisionEncoder):
    """
    Production Vision Encoder wrapping Owkin's Phikon-v2 (DINOv2 ViT-L/16).
    Extracts the 1024-dimensional [CLS] token embedding for each tissue patch.
    """

    MODEL_ID = "owkin/phikon-v2"
    EXPECTED_DIM = 1024

    def __init__(
        self,
        local_files_only: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initializes the Phikon-v2 processor and model.
        
        Args:
            local_files_only: If True, restricts weight loading strictly to local cache.
            device: Torch device (defaults to CUDA if available, else MPS/CPU).
        """
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.local_files_only = local_files_only
        self._processor = None
        self._model = None
        self._load_model()

    @property
    def embedding_dim(self) -> int:
        return self.EXPECTED_DIM

    @property
    def is_mock(self) -> bool:
        return False

    def _load_model(self) -> None:
        """Attempts to load Phikon-v2 processor and model weights."""
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as e:
            raise RuntimeError(
                "The 'transformers' library is required to run Phikon-v2 inference. "
                "Install via: pip install transformers"
            ) from e

        try:
            # 1. Load Image Processor
            self._processor = AutoImageProcessor.from_pretrained(
                self.MODEL_ID,
                local_files_only=self.local_files_only,
            )
        except Exception as e:
            raise MissingModelWeightsError(
                f"Phikon-v2 Image Processor could not be loaded from local cache "
                f"('{self.MODEL_ID}'). Details: {e}. "
                f"Ensure 'preprocessor_config.json' is present in Hugging Face cache."
            ) from e

        try:
            # 2. Load Model Graph and Trained Weights
            self._model = AutoModel.from_pretrained(
                self.MODEL_ID,
                local_files_only=self.local_files_only,
            )
            self._model.to(self.device)
            self._model.eval()
        except Exception as e:
            raise MissingModelWeightsError(
                f"Phikon-v2 model weights could not be loaded locally for '{self.MODEL_ID}'. "
                f"Error: {e}. "
                f"The complete ~1.22 GB 'model.safetensors' file must be downloaded to "
                f"~/.cache/huggingface/hub/models--owkin--phikon-v2 before live inference can proceed."
            ) from e

    def encode(
        self,
        patches: List[Image.Image],
        batch_size: int = 16,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Extracts 1024-D CLS embeddings across all patches in memory-bounded batches.
        Matches senior notebook extraction: outputs.last_hidden_state[:, 0, :].
        """
        if not patches:
            raise ValueError("Cannot encode an empty list of image patches.")

        target_device = torch.device(device) if device else self.device
        if self._model.device != target_device:
            self._model.to(target_device)

        patch_features = []

        with torch.no_grad():
            for i in range(0, len(patches), batch_size):
                sub_batch = patches[i : i + batch_size]
                
                # Standard ViT preprocessing: 224x224, ImageNet normalization
                inputs = self._processor(images=sub_batch, return_tensors="pt")
                pixel_values = inputs.pixel_values.to(target_device)

                outputs = self._model(pixel_values=pixel_values)

                # Extract [CLS] token (position 0)
                cls_tokens = outputs.last_hidden_state[:, 0, :].cpu()
                patch_features.append(cls_tokens)

        # Concatenate into [N, 1024]
        combined = torch.cat(patch_features, dim=0).to(torch.float32)

        # Return with batch dimension: [1, N, 1024]
        return combined.unsqueeze(0)


class MockVisionEncoder(BaseVisionEncoder):
    """
    TEST-ONLY Vision Encoder that generates controlled synthetic embeddings.
    Visibly flagged as mock. Must NEVER be used in clinical production.
    """

    EXPECTED_DIM = 1024

    def __init__(self, seed: int = 42):
        self.seed = seed

    @property
    def embedding_dim(self) -> int:
        return self.EXPECTED_DIM

    @property
    def is_mock(self) -> bool:
        return True

    def encode(
        self,
        patches: List[Image.Image],
        batch_size: int = 16,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """Generates deterministic synthetic embeddings of shape [1, N, 1024]."""
        if not patches:
            raise ValueError("Cannot encode an empty list of image patches.")

        num_patches = len(patches)
        generator = torch.Generator().manual_seed(self.seed)
        
        # Approximate empirical Phikon-v2 distribution: N(0, 0.28^2)
        mock_features = torch.randn(
            (1, num_patches, self.EXPECTED_DIM),
            generator=generator,
            dtype=torch.float32,
        ) * 0.2846

        if device:
            mock_features = mock_features.to(device)

        return mock_features
