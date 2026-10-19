"""
Unit tests for src/inference/image_preprocessing.py.
"""

import numpy as np
import pytest
from PIL import Image

from src.inference.image_preprocessing import (
    ImagePreprocessingError,
    HistologyImagePreprocessor,
    PatchCollection,
)


def create_tissue_tile(size=224, tissue_level="high"):
    """Creates a synthetic patch with high or low tissue coverage."""
    if tissue_level == "high":
        # Simulates stained tissue: darker values, typical H&E absorption
        arr = np.random.randint(40, 160, size=(size, size, 3), dtype=np.uint8)
    else:
        # Simulates slide background glass: near-white
        arr = np.random.randint(230, 255, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_single_tile_preprocessing():
    """Valid 224x224 tile with high tissue coverage processes cleanly."""
    preprocessor = HistologyImagePreprocessor(tile_size=224, tissue_threshold=0.50)
    tile = create_tissue_tile(224, "high")
    collection = preprocessor.preprocess(tile)

    assert isinstance(collection, PatchCollection)
    assert len(collection) == 1
    assert collection.coordinates == [(0, 0)]
    assert collection.patches[0].size == (224, 224)
    assert collection.metadata["source_type"] == "single_tile"


def test_single_tile_low_tissue_rejected():
    """A 224x224 tile consisting of empty slide glass must raise ImagePreprocessingError."""
    preprocessor = HistologyImagePreprocessor(tile_size=224, tissue_threshold=0.50)
    empty_tile = create_tissue_tile(224, "low")
    with pytest.raises(ImagePreprocessingError, match="insufficient tissue content"):
        preprocessor.preprocess(empty_tile)


def test_large_image_tiling():
    """A 672x448 image should be tiled into a 3x2 grid (up to 6 tiles)."""
    preprocessor = HistologyImagePreprocessor(tile_size=224, tissue_threshold=0.30)
    large_img = create_tissue_tile(672, "high").crop((0, 0, 672, 448))  # 672 width x 448 height
    collection = preprocessor.preprocess(large_img)

    assert len(collection) == 6  # (672 // 224) * (448 // 224) = 3 * 2 = 6
    assert collection.metadata["source_type"] == "whole_slide_grid"
    assert len(collection.coordinates) == 6
    # Verify coordinate bounds
    for x, y in collection.coordinates:
        assert 0 <= x <= 448
        assert 0 <= y <= 224


def test_deterministic_patch_subsampling():
    """
    When the number of patches exceeds max_patches, subsampling with a fixed seed
    must produce bit-identical patch selections across runs.
    """
    preprocessor_1 = HistologyImagePreprocessor(tile_size=224, max_patches=3, seed=42)
    preprocessor_2 = HistologyImagePreprocessor(tile_size=224, max_patches=3, seed=42)
    preprocessor_3 = HistologyImagePreprocessor(tile_size=224, max_patches=3, seed=999)

    # 448 x 448 produces 4 patches
    img = create_tissue_tile(448, "high")
    
    col_1 = preprocessor_1.preprocess(img)
    col_2 = preprocessor_2.preprocess(img)
    col_3 = preprocessor_3.preprocess(img)

    assert len(col_1) == 3
    assert len(col_2) == 3
    assert col_1.metadata["subsampled"] is True

    # Same seed -> exact same coordinates selected
    assert col_1.coordinates == col_2.coordinates

    # Different seed -> different selection
    assert col_1.coordinates != col_3.coordinates
