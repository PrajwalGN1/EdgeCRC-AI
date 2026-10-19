"""
Input Validation Module for Multimodal CRC Raw-Input Inference.

Validates that both modalities (H&E Histology Image and Clinical Pathology Report)
are present, uncorrupted, properly formatted, and safely associated with a single patient case.
Fails closed immediately on missing or invalid data without fabricating dummy features.
"""

from dataclasses import dataclass, field
import io
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import uuid

import numpy as np
from PIL import Image


class InferenceValidationError(Exception):
    """Base exception for all multimodal inference input validation failures."""
    pass


class MissingInputError(InferenceValidationError):
    """Raised when either required modality (image or report text) is missing."""
    pass


class InvalidImageError(InferenceValidationError):
    """Raised when the histological image cannot be decoded or violates geometric constraints."""
    pass


class InvalidReportError(InferenceValidationError):
    """Raised when the clinical report text is empty, degenerate, or invalid."""
    pass


class CaseMismatchError(InferenceValidationError):
    """Raised when case IDs associated with the image and text inputs conflict."""
    pass


@dataclass
class ValidatedCase:
    """Container for validated multimodal case inputs ready for preprocessing and inference."""
    case_id: str
    image: Image.Image
    report_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_and_validate_image(
    image_input: Union[Image.Image, np.ndarray, Path, str, bytes, io.BytesIO],
    min_dimension: int = 224,
    max_dimension: int = 50000,
) -> Image.Image:
    """
    Decodes and validates a raw histological image input.
    
    Args:
        image_input: PIL Image, NumPy array, file path, or bytes buffer.
        min_dimension: Minimum acceptable height and width (default: 224).
        max_dimension: Upper resource safety limit for image dimensions.
        
    Returns:
        Validated PIL.Image in RGB mode.
        
    Raises:
        MissingInputError: If image_input is None.
        InvalidImageError: If decoding fails or geometry/color requirements are violated.
    """
    if image_input is None:
        raise MissingInputError(
            "Histology image input is required. Cannot proceed with multimodal inference without an image."
        )

    pil_img: Optional[Image.Image] = None

    try:
        if isinstance(image_input, Image.Image):
            pil_img = image_input
        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                raise InvalidImageError("Input NumPy image array is empty (size 0).")
            if image_input.ndim == 2:
                raise InvalidImageError(
                    f"Histological image must have 3 RGB color channels, but received 2D array: {image_input.shape}."
                )
            if image_input.ndim != 3 or image_input.shape[2] not in (3, 4):
                raise InvalidImageError(
                    f"Unsupported NumPy image array shape: {image_input.shape}. Expected (H, W, 3) or (H, W, 4)."
                )
            pil_img = Image.fromarray(image_input)
        elif isinstance(image_input, (str, Path)):
            path_obj = Path(image_input)
            if not path_obj.exists():
                raise InvalidImageError(f"Image file does not exist on disk: {path_obj}")
            if path_obj.stat().st_size == 0:
                raise InvalidImageError(f"Image file is empty (0 bytes): {path_obj}")
            pil_img = Image.open(path_obj)
        elif isinstance(image_input, (bytes, bytearray)):
            if len(image_input) == 0:
                raise InvalidImageError("Input image byte buffer is empty (0 bytes).")
            pil_img = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, io.BytesIO):
            image_input.seek(0)
            data = image_input.read()
            if len(data) == 0:
                raise InvalidImageError("Input BytesIO stream contains 0 bytes.")
            image_input.seek(0)
            pil_img = Image.open(image_input)
        else:
            raise InvalidImageError(
                f"Unsupported image input type: {type(image_input)}. "
                f"Expected PIL Image, NumPy array, file path, or bytes."
            )

        # Force load image data to catch corrupt headers / truncated file streams
        pil_img.load()

    except Exception as e:
        if isinstance(e, InferenceValidationError):
            raise
        raise InvalidImageError(f"Failed to decode or parse histological image: {str(e)}") from e

    width, height = pil_img.size
    if width < min_dimension or height < min_dimension:
        raise InvalidImageError(
            f"Image dimensions ({width}x{height}) are smaller than the required minimum "
            f"tile size ({min_dimension}x{min_dimension})."
        )
    if width > max_dimension or height > max_dimension:
        raise InvalidImageError(
            f"Image dimensions ({width}x{height}) exceed maximum resource limit ({max_dimension}x{max_dimension})."
        )

    # Convert to RGB color space
    if pil_img.mode != "RGB":
        if pil_img.mode in ("RGBA", "RGBX"):
            pil_img = pil_img.convert("RGB")
        elif pil_img.mode == "L":
            raise InvalidImageError(
                "Histological H&E image must be color (RGB), but a 1-channel grayscale image was provided."
            )
        else:
            try:
                pil_img = pil_img.convert("RGB")
            except Exception as e:
                raise InvalidImageError(
                    f"Unsupported image color mode '{pil_img.mode}' could not be converted to RGB: {e}"
                )

    # Sanity check: Ensure image has non-zero visual entropy (not a flat single-color blank)
    img_arr = np.array(pil_img)
    if img_arr.std() < 1e-3:
        raise InvalidImageError(
            "Image appears to be completely blank or uniform (standard deviation ≈ 0). "
            "Valid histology tissue content is required."
        )

    return pil_img


def validate_report_text(
    report_input: Optional[str],
    min_length_chars: int = 10,
    max_length_chars: int = 500000,
) -> str:
    """
    Validates clinical/pathology report text.
    
    Args:
        report_input: Raw string containing surgical or pathology notes.
        min_length_chars: Minimum required non-whitespace character length.
        max_length_chars: Upper boundary on character length.
        
    Returns:
        Sanitized, non-empty text string.
        
    Raises:
        MissingInputError: If report_input is None.
        InvalidReportError: If text is empty, too short, or non-textual.
    """
    if report_input is None:
        raise MissingInputError(
            "Clinical pathology report is required. Cannot proceed with multimodal inference without report text."
        )

    if not isinstance(report_input, str):
        raise InvalidReportError(
            f"Clinical report must be a string, but received: {type(report_input).__name__}."
        )

    cleaned_text = report_input.strip()
    if len(cleaned_text) < min_length_chars:
        raise InvalidReportError(
            f"Clinical pathology report is too short ({len(cleaned_text)} characters). "
            f"A meaningful pathology narrative of at least {min_length_chars} characters is required."
        )

    if len(cleaned_text) > max_length_chars:
        raise InvalidReportError(
            f"Clinical report exceeds maximum character limit ({len(cleaned_text)} > {max_length_chars})."
        )

    # Check for minimal alphabetic content to reject pure numbers/punctuation
    alpha_chars = sum(1 for c in cleaned_text if c.isalpha())
    if alpha_chars < 5:
        raise InvalidReportError(
            "Clinical report lacks sufficient alphabetic text to represent a medical pathology report."
        )

    return cleaned_text


def validate_multimodal_inputs(
    image_input: Any,
    report_input: Any,
    case_id: Optional[str] = None,
    image_case_id: Optional[str] = None,
    report_case_id: Optional[str] = None,
) -> ValidatedCase:
    """
    Validates paired image and text inputs and confirms case association.
    
    Args:
        image_input: Histology image (PIL, array, path, bytes).
        report_input: Clinical pathology text string.
        case_id: Optional user-specified case identifier.
        image_case_id: Optional case identifier attached to the image.
        report_case_id: Optional case identifier attached to the report.
        
    Returns:
        ValidatedCase ready for preprocessing.
        
    Raises:
        MissingInputError: If either modality is omitted.
        InvalidImageError: If image validation fails.
        InvalidReportError: If text validation fails.
        CaseMismatchError: If image and report case IDs conflict.
    """
    # 1. Enforce Case Association
    if image_case_id and report_case_id and image_case_id.strip() != report_case_id.strip():
        raise CaseMismatchError(
            f"Case association mismatch detected: Image belongs to case '{image_case_id}', "
            f"but clinical report belongs to case '{report_case_id}'."
        )

    resolved_case_id = (
        case_id or image_case_id or report_case_id or f"CASE_{uuid.uuid4().hex[:8].upper()}"
    ).strip()

    # 2. Validate Image
    validated_image = load_and_validate_image(image_input)

    # 3. Validate Report
    validated_report = validate_report_text(report_input)

    return ValidatedCase(
        case_id=resolved_case_id,
        image=validated_image,
        report_text=validated_report,
        metadata={
            "image_width": validated_image.width,
            "image_height": validated_image.height,
            "report_length_chars": len(validated_report),
        },
    )
