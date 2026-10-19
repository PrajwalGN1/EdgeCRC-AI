"""
Unit tests for src/inference/input_validation.py.
"""

import io
import numpy as np
import pytest
from PIL import Image

from src.inference.input_validation import (
    InferenceValidationError,
    MissingInputError,
    InvalidImageError,
    InvalidReportError,
    CaseMismatchError,
    load_and_validate_image,
    validate_report_text,
    validate_multimodal_inputs,
)


def create_synthetic_tissue_image(width=224, height=224):
    """Creates a synthetic histological-like image with non-zero entropy."""
    arr = np.random.randint(50, 200, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_valid_inputs():
    """Happy path: valid 224x224 RGB image and valid pathology report string."""
    img = create_synthetic_tissue_image(224, 224)
    text = "Surgical specimen: invasive adenocarcinoma of ascending colon, poorly differentiated."
    case = validate_multimodal_inputs(image_input=img, report_input=text, case_id="TCGA-TEST-001")
    assert case.case_id == "TCGA-TEST-001"
    assert case.image.size == (224, 224)
    assert case.report_text == text
    assert case.metadata["report_length_chars"] == len(text)


def test_missing_image():
    """Omitting the image must raise MissingInputError immediately."""
    text = "Valid clinical report text excerpt for testing."
    with pytest.raises(MissingInputError, match="Histology image input is required"):
        validate_multimodal_inputs(image_input=None, report_input=text)


def test_missing_report():
    """Omitting the report must raise MissingInputError immediately."""
    img = create_synthetic_tissue_image(224, 224)
    with pytest.raises(MissingInputError, match="Clinical pathology report is required"):
        validate_multimodal_inputs(image_input=img, report_input=None)


def test_empty_or_whitespace_report():
    """Empty or pure whitespace report text must raise InvalidReportError."""
    img = create_synthetic_tissue_image(224, 224)
    with pytest.raises(InvalidReportError, match="too short"):
        validate_multimodal_inputs(image_input=img, report_input="   \n\t  ")


def test_report_too_short():
    """Reports with fewer than 10 characters must be rejected."""
    img = create_synthetic_tissue_image(224, 224)
    with pytest.raises(InvalidReportError, match="too short"):
        validate_multimodal_inputs(image_input=img, report_input="Short")


def test_report_pure_numbers():
    """Reports lacking alphabetic content must be rejected."""
    img = create_synthetic_tissue_image(224, 224)
    with pytest.raises(InvalidReportError, match="lacks sufficient alphabetic text"):
        validate_multimodal_inputs(image_input=img, report_input="1234567890 987654321")


def test_corrupt_image_bytes():
    """Corrupted image bytes must raise InvalidImageError."""
    bad_bytes = b"NOT_A_VALID_JPEG_OR_PNG_IMAGE_DATA_STREAM"
    with pytest.raises(InvalidImageError, match="Failed to decode or parse"):
        load_and_validate_image(bad_bytes)


def test_empty_image_bytes():
    """Empty image buffer must raise InvalidImageError."""
    with pytest.raises(InvalidImageError, match="empty"):
        load_and_validate_image(b"")


def test_image_too_small():
    """Images smaller than 224x224 must raise InvalidImageError."""
    small_img = create_synthetic_tissue_image(100, 100)
    with pytest.raises(InvalidImageError, match="smaller than the required minimum"):
        load_and_validate_image(small_img)


def test_grayscale_image_rejected():
    """1-channel grayscale images cannot represent H&E and must be rejected."""
    gray_arr = np.random.randint(50, 200, size=(224, 224), dtype=np.uint8)
    gray_img = Image.fromarray(gray_arr, mode="L")
    with pytest.raises(InvalidImageError, match="grayscale"):
        load_and_validate_image(gray_img)


def test_uniform_blank_image_rejected():
    """Single-color blank images (std dev ~ 0) must be rejected."""
    blank_arr = np.full((224, 224, 3), 255, dtype=np.uint8)
    blank_img = Image.fromarray(blank_arr)
    with pytest.raises(InvalidImageError, match="completely blank"):
        load_and_validate_image(blank_img)


def test_case_id_mismatch():
    """Conflicting case IDs between image and report must raise CaseMismatchError."""
    img = create_synthetic_tissue_image(224, 224)
    text = "Valid clinical pathology notes for colon carcinoma."
    with pytest.raises(CaseMismatchError, match="Case association mismatch detected"):
        validate_multimodal_inputs(
            image_input=img,
            report_input=text,
            image_case_id="PATIENT_A",
            report_case_id="PATIENT_B",
        )
