"""
Image Preprocessing and Tiling Module for Multimodal CRC Histopathology Inference.

Handles both:
1. Supported single 224x224 histological image tiles.
2. Large whole-slide scans / micrographs via a documented grid-tiling, Otsu tissue-filtering,
   and deterministic patch-selection interface.

Preserves the project's baseline:
- Target patch size: 224x224 pixels at 10x magnification (~1.0 µm/px).
- Otsu thresholding tissue filtering (discards tiles with >50% non-tissue background).
- Bounded-memory processing and deterministic patch selection (capped at N ≤ 1000).
- Explicit coordinate and metadata tracking.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch


class ImagePreprocessingError(Exception):
    """Base exception for image preprocessing and tiling failures."""
    pass


@dataclass
class PatchCollection:
    """Container for preprocessed patches, spatial coordinates, and extraction metadata."""
    patches: List[Image.Image]
    coordinates: List[Tuple[int, int]]  # (x, y) top-left in original image
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.patches)


class HistologyImagePreprocessor:
    """
    Preprocessor for histological images tailored for Phikon-v2 feature extraction.
    """

    def __init__(
        self,
        tile_size: int = 224,
        tissue_threshold: float = 0.50,
        max_patches: int = 1000,
        seed: int = 42,
    ):
        """
        Args:
            tile_size: Target square tile dimension in pixels (default: 224).
            tissue_threshold: Minimum fraction of pixels containing tissue (Otsu mask) (default: 0.50).
            max_patches: Maximum number of patches retained per slide to bound memory (default: 1000).
            seed: Random seed for deterministic patch subsampling when N > max_patches (default: 42).
        """
        self.tile_size = tile_size
        self.tissue_threshold = tissue_threshold
        self.max_patches = max_patches
        self.seed = seed

    @staticmethod
    def compute_otsu_tissue_fraction(image_tile: Image.Image) -> float:
        """
        Computes the fraction of tissue (non-background) in a tile.
        In histology whole-slide imaging, glass background is bright white (grayscale > 220).
        Stained tissue has absorption with lower luminance and non-zero color saturation.
        
        Returns:
            Float in [0.0, 1.0] representing tissue coverage fraction.
        """
        # Convert to grayscale array
        gray = np.array(image_tile.convert("L"))
        
        # Pixels with grayscale < 220 are tissue pixels
        # For stained tissue, pixels are well below 220
        # If the tile has significant contrast (both tissue and glass present),
        # Otsu thresholding can refine the boundary
        is_tissue_basic = (gray < 220)
        basic_fraction = float(np.mean(is_tissue_basic))

        # If clearly almost all tissue or almost all glass, return directly
        if basic_fraction > 0.85 or basic_fraction < 0.15:
            return basic_fraction

        # Use Otsu on the non-saturated pixels to find exact tissue edge
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        total = gray.size

        current_max = 0.0
        threshold = 215
        weight_bg = 0
        sum_bg = 0
        sum_total = np.dot(np.arange(256), hist)

        for t in range(256):
            weight_bg += hist[t]
            if weight_bg == 0:
                continue
            weight_fg = total - weight_bg
            if weight_fg == 0:
                break
            sum_bg += t * hist[t]
            m_bg = sum_bg / weight_bg
            m_fg = (sum_total - sum_bg) / weight_fg
            between_var = weight_bg * weight_fg * ((m_bg - m_fg) ** 2)
            if between_var > current_max:
                current_max = between_var
                threshold = t

        # Ensure threshold does not exceed glass boundary
        effective_threshold = min(threshold, 220)
        tissue_mask = (gray < effective_threshold)
        return float(np.mean(tissue_mask))

    @staticmethod
    def apply_reinhard_normalization(
        image_tile: Image.Image,
        target_mean: Optional[np.ndarray] = None,
        target_std: Optional[np.ndarray] = None,
    ) -> Image.Image:
        """
        Applies Reinhard stain normalization against a calibrated target reference in Lab space.
        If no target statistics are provided, returns the original tile unchanged to prevent
        unverified color alteration (preserving original Kaggle preprocessing).
        """
        if target_mean is None or target_std is None:
            return image_tile

        # Convert RGB to Lab space
        rgb_arr = np.array(image_tile).astype(np.float32)
        # Fast perceptual RGB -> Lab approximation
        # Standard OpenCV or skimage equivalent
        try:
            import cv2
            lab = cv2.cvtColor(rgb_arr.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        except ImportError:
            # Fallback: if cv2 not present, return original
            return image_tile

        src_mean = np.mean(lab, axis=(0, 1))
        src_std = np.std(lab, axis=(0, 1)) + 1e-6

        # Scale and shift to target
        norm_lab = (lab - src_mean) * (target_std / src_std) + target_mean
        norm_lab = np.clip(norm_lab, 0, 255).astype(np.uint8)

        norm_rgb = cv2.cvtColor(norm_lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(norm_rgb)

    def process_single_tile(
        self,
        tile: Image.Image,
        reinhard_target_mean: Optional[np.ndarray] = None,
        reinhard_target_std: Optional[np.ndarray] = None,
    ) -> PatchCollection:
        """
        Processes an already-cropped 224x224 histological patch.
        
        Args:
            tile: 224x224 PIL Image.
            reinhard_target_mean: Optional target mean for stain normalization.
            reinhard_target_std: Optional target std for stain normalization.
            
        Returns:
            PatchCollection containing the single patch and metadata.
        """
        w, h = tile.size
        if (w, h) != (self.tile_size, self.tile_size):
            # If slightly different, bicubic resize to exact target tile size
            tile = tile.resize((self.tile_size, self.tile_size), Image.Resampling.BICUBIC)

        tissue_frac = self.compute_otsu_tissue_fraction(tile)
        if tissue_frac < self.tissue_threshold:
            raise ImagePreprocessingError(
                f"Single image tile has insufficient tissue content ({tissue_frac:.1%} < {self.tissue_threshold:.1%}). "
                f"At least {self.tissue_threshold:.0%} non-background tissue is required."
            )

        processed_tile = self.apply_reinhard_normalization(
            tile, reinhard_target_mean, reinhard_target_std
        )

        return PatchCollection(
            patches=[processed_tile],
            coordinates=[(0, 0)],
            metadata={
                "source_type": "single_tile",
                "original_dimensions": (w, h),
                "total_extracted": 1,
                "kept_patches": 1,
                "tissue_fraction": tissue_frac,
            },
        )

    def tile_large_image(
        self,
        large_image: Image.Image,
        reinhard_target_mean: Optional[np.ndarray] = None,
        reinhard_target_std: Optional[np.ndarray] = None,
    ) -> PatchCollection:
        """
        Tiles a large histological scan/micrograph into 224x224 patches, filters background
        via Otsu thresholding, and applies deterministic subsampling if patch count > max_patches.
        
        Args:
            large_image: High-resolution PIL Image.
            reinhard_target_mean: Optional stain normalization target mean.
            reinhard_target_std: Optional stain normalization target std.
            
        Returns:
            PatchCollection containing list of patches and their (x, y) top-left coordinates.
        """
        w, h = large_image.size
        stride = self.tile_size

        extracted_patches: List[Image.Image] = []
        coordinates: List[Tuple[int, int]] = []
        total_grid_tiles = 0

        # Bounded sliding window tiling
        for y in range(0, h - self.tile_size + 1, stride):
            for x in range(0, w - self.tile_size + 1, stride):
                total_grid_tiles += 1
                crop_box = (x, y, x + self.tile_size, y + self.tile_size)
                tile = large_image.crop(crop_box)

                # Otsu foreground tissue check
                tissue_frac = self.compute_otsu_tissue_fraction(tile)
                if tissue_frac >= self.tissue_threshold:
                    norm_tile = self.apply_reinhard_normalization(
                        tile, reinhard_target_mean, reinhard_target_std
                    )
                    extracted_patches.append(norm_tile)
                    coordinates.append((x, y))

        if len(extracted_patches) == 0:
            raise ImagePreprocessingError(
                f"No valid histological tissue patches could be extracted from image ({w}x{h}). "
                f"All {total_grid_tiles} grid tiles failed the {self.tissue_threshold:.0%} tissue threshold."
            )

        # Deterministic patch selection if patch count exceeds max_patches (capped at 1000)
        kept_patches = extracted_patches
        kept_coords = coordinates
        was_subsampled = False

        if len(extracted_patches) > self.max_patches:
            generator = torch.Generator().manual_seed(self.seed)
            perm = torch.randperm(len(extracted_patches), generator=generator)[: self.max_patches].tolist()
            kept_patches = [extracted_patches[i] for i in perm]
            kept_coords = [coordinates[i] for i in perm]
            was_subsampled = True

        return PatchCollection(
            patches=kept_patches,
            coordinates=kept_coords,
            metadata={
                "source_type": "whole_slide_grid",
                "original_dimensions": (w, h),
                "grid_tiles_evaluated": total_grid_tiles,
                "tissue_tiles_extracted": len(extracted_patches),
                "retained_patches": len(kept_patches),
                "subsampled": was_subsampled,
                "max_patch_cap": self.max_patches,
                "seed": self.seed,
            },
        )

    def preprocess(
        self,
        image: Image.Image,
        reinhard_target_mean: Optional[np.ndarray] = None,
        reinhard_target_std: Optional[np.ndarray] = None,
    ) -> PatchCollection:
        """
        Dispatches image preprocessing based on input dimensions.
        If the image is already 224x224, processes as a single tile;
        otherwise tiles it into a collection of 224x224 patches.
        """
        w, h = image.size
        if w == self.tile_size and h == self.tile_size:
            return self.process_single_tile(image, reinhard_target_mean, reinhard_target_std)
        else:
            return self.tile_large_image(image, reinhard_target_mean, reinhard_target_std)
