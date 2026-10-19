"""
Report Encoder Module for Multimodal CRC Clinical Text Inference.

Integrates Bio_ClinicalBERT (108.31M parameters, fine-tuned on 2M MIMIC-III clinical notes)
to extract 768-dimensional report CLS embeddings matching the precomputed tensor store.

Enforces:
- Exact tokenization and pooling matching the trained baseline:
  WordPiece tokenizer, max_length=512, truncation=True, padding="max_length".
- Extraction of [CLS] token at position 0 from last_hidden_state.
- Strict fail-closed error handling when model weights are not cached locally.
- Explicit separation between production ClinicalBERT encoder and test-only mock encoders.
"""

from abc import ABC, abstractmethod
import logging
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)


class MissingModelWeightsError(RuntimeError):
    """Raised when pretrained foundation encoder weights are missing from local storage."""
    pass


class BaseReportEncoder(ABC):
    """Abstract base class for report encoders."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Returns the output feature embedding dimension (expected: 768)."""
        pass

    @property
    @abstractmethod
    def is_mock(self) -> bool:
        """Indicates whether this encoder is a synthetic mock for testing."""
        pass

    @abstractmethod
    def encode(
        self,
        report_text: str,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Encodes a raw clinical report string into a 768-dimensional summary embedding.
        
        Args:
            report_text: Pathology / clinical note text.
            device: Target torch device (cuda, mps, or cpu).
            
        Returns:
            torch.Tensor of shape [1, 768] and dtype torch.float32.
        """
        pass


class ClinicalBERTReportEncoder(BaseReportEncoder):
    """
    Production Report Encoder wrapping emilyalsentzer/Bio_ClinicalBERT.
    Tokenizes pathology text and extracts the 768-D [CLS] token embedding.
    """

    MODEL_ID = "emilyalsentzer/Bio_ClinicalBERT"
    EXPECTED_DIM = 768
    MAX_LENGTH = 512

    def __init__(
        self,
        local_files_only: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initializes the ClinicalBERT tokenizer and model.
        
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
        self._tokenizer = None
        self._model = None
        self._load_model()

    @property
    def embedding_dim(self) -> int:
        return self.EXPECTED_DIM

    @property
    def is_mock(self) -> bool:
        return False

    def _load_model(self) -> None:
        """Attempts to load ClinicalBERT tokenizer and model weights."""
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError as e:
            raise RuntimeError(
                "The 'transformers' library is required to run ClinicalBERT inference. "
                "Install via: pip install transformers"
            ) from e

        try:
            # 1. Load Tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_ID,
                local_files_only=self.local_files_only,
            )
        except Exception as e:
            raise MissingModelWeightsError(
                f"Bio_ClinicalBERT Tokenizer could not be loaded from local cache "
                f"('{self.MODEL_ID}'). Details: {e}. "
                f"Ensure 'vocab.txt' and 'config.json' are present in Hugging Face cache."
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
                f"Bio_ClinicalBERT model weights could not be loaded locally for '{self.MODEL_ID}'. "
                f"Error: {e}. "
                f"The complete ~436 MB 'pytorch_model.bin' file must be downloaded to "
                f"~/.cache/huggingface/hub/models--emilyalsentzer--Bio_ClinicalBERT before live inference can proceed."
            ) from e

    def encode(
        self,
        report_text: str,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Tokenizes the input text and extracts the 768-D [CLS] embedding.
        Matches senior project pipeline: outputs.last_hidden_state[:, 0, :].
        """
        if not report_text or not report_text.strip():
            raise ValueError("Cannot encode empty report text.")

        target_device = torch.device(device) if device else self.device
        if self._model.device != target_device:
            self._model.to(target_device)

        with torch.no_grad():
            encoded = self._tokenizer(
                report_text,
                padding="max_length",
                truncation=True,
                max_length=self.MAX_LENGTH,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(target_device)
            attention_mask = encoded["attention_mask"].to(target_device)

            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)

            # Extract [CLS] token summary: shape [1, 768]
            cls_token = outputs.last_hidden_state[:, 0, :].cpu().to(torch.float32)

        return cls_token


class MockReportEncoder(BaseReportEncoder):
    """
    TEST-ONLY Report Encoder that generates controlled synthetic embeddings.
    Visibly flagged as mock. Must NEVER be used in clinical production.
    """

    EXPECTED_DIM = 768

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
        report_text: str,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """Generates deterministic synthetic text embeddings of shape [1, 768]."""
        if not report_text or not report_text.strip():
            raise ValueError("Cannot encode empty report text.")

        # Derive a stable numeric seed from text content + base seed
        content_hash = abs(hash(report_text[:100])) % (2**31)
        generator = torch.Generator().manual_seed(self.seed + content_hash)

        # Approximate empirical ClinicalBERT distribution: N(-0.0066, 0.5017^2)
        mock_embedding = torch.randn(
            (1, self.EXPECTED_DIM),
            generator=generator,
            dtype=torch.float32,
        ) * 0.5017 - 0.0066

        if device:
            mock_embedding = mock_embedding.to(device)

        return mock_embedding
