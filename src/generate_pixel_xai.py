import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import logging
from pathlib import Path

def set_deterministic_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_deterministic_seed(42)

def get_interactive_input():
    print("\n" + "=" * 50)
    print("ADVANCED MULTIMODAL XAI ENGINE v2.0 (Microscopic)")
    print("=" * 50)
    print("Select the tissue class folder:")
    print("  [1] MSI (MSIMUT_JPEG)")
    print("  [2] MSS (MSS_JPEG)")
    folder_choice = input("\nEnter 1 or 2: ").strip()
    if folder_choice == '1':
        target_folder = "../data/MSIMUT_JPEG"
    elif folder_choice == '2':
        target_folder = "../data/MSS_JPEG"
    else:
        target_folder = "MSS_JPEG"
    print(f"\nSelected Folder: {target_folder}")
    img_filename = input("Enter the exact image filename (e.g., TCGA-CM-6674_patch_45.jpg): ").strip()
    full_path = os.path.join(target_folder, img_filename)
    return full_path

def _get_tissue_mask(arr):
    """Robust tissue vs white background mask for 224x224 H&E patch.
    Returns bool mask True = tissue (not white glass/background).
    Uses saturation + brightness + morphology — no Phikon needed.
    """
    # HSV saturation: tissue ~30-255, white ~0-25
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # Otsu on saturation
    s_uint8 = s.astype(np.uint8)
    _, s_otsu = cv2.threshold(s_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tissue_s = s_otsu > 0
    # White glass: high brightness + low saturation
    white_mask = (gray > 210) & (s < 25) & (v > 200)
    tissue = tissue_s & (~white_mask)
    # Additional: very bright but low sat = glass even if Otsu said tissue
    # Morph clean
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tissue_u8 = tissue.astype(np.uint8) * 255
    tissue_u8 = cv2.morphologyEx(tissue_u8, cv2.MORPH_OPEN, kernel)
    tissue_u8 = cv2.morphologyEx(tissue_u8, cv2.MORPH_CLOSE, kernel)
    tissue = tissue_u8 > 0
    # If mask is tiny (<5% of patch) or huge (>95%), fallback to simple brightness gate
    coverage = tissue.mean()
    if coverage < 0.05 or coverage > 0.95:
        # fallback: tissue = not white
        tissue = ~(white_mask)
        tissue_u8 = tissue.astype(np.uint8) * 255
        tissue_u8 = cv2.morphologyEx(tissue_u8, cv2.MORPH_OPEN, kernel)
        tissue = tissue_u8 > 0
    return tissue

def _hematoxylin_density(arr):
    """Approximate hematoxylin (nuclei) density for fallback.
    Returns float map 0-1 where high = nuclei / glandular border / immune dense.
    Combines saturation, blue dominance, darkness — white background suppressed.
    """
    arr_f = arr.astype(np.float32) / 255.0
    # Avoid log(0)
    arr_f = np.clip(arr_f, 1e-4, 1.0)
    # Optical density
    od = -np.log(arr_f)
    # Ruifrok & Johnston H&E matrix (approx)
    # H: [0.65, 0.70, 0.29], E: [0.07, 0.99, 0.11]
    h_coeff = np.array([0.65, 0.70, 0.29])
    e_coeff = np.array([0.07, 0.99, 0.11])
    # Project (approx without deconvolution inverse, just dot)
    h_od = np.dot(od, h_coeff)  # hematoxylin OD — high in nuclei (blue-purple)
    e_od = np.dot(od, e_coeff)  # eosin OD — high in cytoplasm
    # Normalize each to 0-1
    h_od = (h_od - h_od.min()) / (h_od.max() - h_od.min() + 1e-8)
    e_od = (e_od - e_od.min()) / (e_od.max() - e_od.min() + 1e-8)
    # Nuclei score: high H, low E, high saturation, low brightness
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    # Blue dominance (B - R) normalized
    b = arr[:, :, 2].astype(np.float32) / 255.0
    r = arr[:, :, 0].astype(np.float32) / 255.0
    blue_dom = np.clip((b - r + 0.3), 0, 1)
    # Combine: hematoxylin OD * saturation * blue * darkness
    darkness = 1.0 - gray
    # Tissue-aware: white has low s and high gray → darkness low
    nuclei = h_od * (0.7 + 0.3 * s) * (0.6 + 0.4 * blue_dom) * (0.5 + 0.5 * darkness)
    # Boost where H > E (nuclei vs cytoplasm)
    he_ratio = np.clip(h_od - e_od * 0.5, 0, 1)
    nuclei = nuclei * (0.7 + 0.3 * he_ratio) + he_ratio * 0.3
    # Normalize
    nuclei = (nuclei - nuclei.min()) / (nuclei.max() - nuclei.min() + 1e-8)
    # Slight gamma to boost high nuclei, suppress mid stroma
    nuclei = np.power(nuclei, 0.85)
    return nuclei

def _dark_fallback_heatmap(img_path, raw_image, out_path):
    """Dark fallback when Phikon unavailable — CORRECTED: highlights nuclei, suppresses white."""
    print("Using CORRECTED DARK fallback heatmap (tissue-aware, hematoxylin)")
    arr = np.array(raw_image)
    tissue_mask = _get_tissue_mask(arr)
    # Hematoxylin-based pseudo-attention (0-1, high = nuclei)
    nuclei = _hematoxylin_density(arr)
    # Suppress white background: force 0 where not tissue
    nuclei[~tissue_mask] = 0
    # Smooth with small then larger blur to simulate 14x14 -> 224
    # First, downsample to 14x14 via area, then upsample with cubic to simulate ViT grid
    small = cv2.resize(nuclei, (14, 14), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (3, 3), 0)
    small = (small - small.min()) / (small.max() - small.min() + 1e-8)
    # Back to 224
    pseudo = cv2.resize(small, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_CUBIC)
    pseudo = cv2.GaussianBlur(pseudo, (7, 7), 0)
    pseudo = (pseudo - pseudo.min()) / (pseudo.max() - pseudo.min() + 1e-8)
    # Re-apply tissue mask after resize (interpolation can bleed into white)
    pseudo[~tissue_mask] = 0
    # Normalize again after masking
    if pseudo.max() > 0:
        pseudo = (pseudo - pseudo.min()) / (pseudo.max() - pseudo.min() + 1e-8)
    # Dark base
    dark_base = (arr.astype(np.float32) * 0.45).astype(np.uint8)
    # Jet via OpenCV (report spec) — but darkened: use jet then blend with dark_base at high alpha
    # Jet in OpenCV is BGR, convert
    heatmap_for_overlay = pseudo.copy()
    # Dynamic threshold: keep top 28-32% of TISSUE only, not including white zeros
    tissue_vals = pseudo[tissue_mask]
    if len(tissue_vals) > 0:
        dyn_thr = np.percentile(tissue_vals, 68)
    else:
        dyn_thr = np.percentile(pseudo, 68)
    heatmap_for_overlay[heatmap_for_overlay < dyn_thr] = 0
    # Also zero where not tissue
    heatmap_for_overlay[~tissue_mask] = 0
    heat_uint8 = (np.clip(heatmap_for_overlay, 0, 1) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    # Darken the heat_color itself slightly (multiply by 0.92) to avoid neon white blowing out dark theme
    heat_color = (heat_color.astype(np.float32) * 0.92).astype(np.uint8)
    # Dark overlay: blend dark_base (0.42) with heat (0.68) — high heat alpha but background suppressed so white not glowing
    overlay = cv2.addWeighted(dark_base, 0.48, heat_color, 0.68, 0)
    # Ensure white regions stay dark in overlay: force overlay where not tissue to be dark_base
    overlay[~tissue_mask] = dark_base[~tissue_mask]
    # Contours: top 20% of tissue
    if len(tissue_vals) > 0:
        thr = np.percentile(tissue_vals, 80)
    else:
        thr = np.percentile(pseudo, 80)
    mask = ((pseudo >= thr) & tissue_mask).astype(np.uint8) * 255
    blurred_mask = cv2.GaussianBlur(mask, (13, 13), 0)
    _, binary = cv2.threshold(blurred_mask, 127, 1, cv2.THRESH_BINARY)
    contour_img = dark_base.copy()
    # Slight brighten for contour visibility but keep dark
    contour_img = cv2.addWeighted(contour_img, 0.92, np.zeros_like(contour_img), 0, 10)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_img, contours, -1, (255, 225, 0), 2, lineType=cv2.LINE_AA)

    # Render 3-panel DARK but CORRECT (tissue-focused)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0B0B0F")
    filename_only = os.path.basename(img_path)
    fig.suptitle(f"Microscopic Attention — Dark Tissue-Aware\n{filename_only} (Fallback — Phikon offline, hematoxylin)", fontsize=12, fontweight='bold', color="#E2E8F0", y=0.98)
    axes[0].imshow(dark_base)
    axes[0].set_title("Darkened H&E Patch (0.45×)", fontweight='bold', pad=12, color="#CBD5E1")
    axes[0].axis('off')
    axes[1].imshow(contour_img)
    axes[1].set_title("ROI Contours — Top 20% Tissue (Dark)", fontweight='bold', pad=12, color="#FDE68A")
    axes[1].axis('off')
    axes[2].imshow(overlay)
    axes[2].set_title("Dark Jet Tissue Overlay α=0.68", fontweight='bold', color="#F87171", pad=12)
    axes[2].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor="#0B0B0F")
    plt.close()
    print(f"SUCCESS (corrected dark fallback) -> {out_path}  tissue_coverage={tissue_mask.mean():.2%}  nuclei_max={pseudo.max():.3f}")
    return out_path

def _attention_rollout_product(attentions):
    """Correct rollout: recursive product of last 6 layers (report §4.9.2), not mean.
    attentions: list of [B, H, N, N] tensors from model (24 layers). We take last 6.
    Returns rollout matrix [B, N, N] averaged over heads but multiplied across layers.
    """
    # Stack last 6: [6, B, H, N, N]
    last_six = torch.stack(attentions[-6:])  # [6, B, H, N, N]
    # Average heads first? Report says product of self-attention matrices; typical rollout: mean heads then product
    # Here we mean heads then product across layers
    # last_six mean over heads: [6, B, N, N]
    avg_heads = last_six.mean(dim=2)  # [6, B, N, N]
    # Add identity for residual (standard rollout)
    # Each attention already includes residual implicitly in ViT? We add I for numerical stability as per Abnar & Zuidema
    B, N, _ = avg_heads.shape[1], avg_heads.shape[2], avg_heads.shape[3]
    eye = torch.eye(N, device=avg_heads.device).unsqueeze(0)  # [1, N, N]
    avg_heads = avg_heads + eye.unsqueeze(0) * 0.0  # keep as is, but normalize
    # Normalize rows to sum to 1
    avg_heads = avg_heads / avg_heads.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    # Product across layers: rollout = A0 * A1 * ... * A5
    rollout = avg_heads[0]  # [B, N, N]
    for i in range(1, 6):
        rollout = torch.bmm(rollout, avg_heads[i])
    return rollout  # [B, N, N]

def generate_smooth_vit_heatmap(img_path, output_path=None):
    print(f"\nInitializing Phikon-v2 Engine for: {img_path}")
    try:
        raw_image = Image.open(img_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Could not find image at {img_path}")
        return None
    filename_only = os.path.basename(img_path)
    if output_path is None:
        try:
            output_path = str(Path("/tmp") / f"Interactive_XAI_{filename_only}.png")
            Path("/tmp").mkdir(parents=True, exist_ok=True)
        except Exception:
            output_path = f"Interactive_XAI_{filename_only}.png"
    # Try Phikon path with CORRECTED rollout + tissue masking
    try:
        from transformers import AutoImageProcessor, AutoModel
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        print(f"Executing on device: {device}")
        logging.getLogger("transformers").setLevel(logging.ERROR)
        processor = None
        model = None
        try:
            processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2", local_files_only=True, trust_remote_code=True)
            model = AutoModel.from_pretrained("owkin/phikon-v2", output_attentions=True, local_files_only=True, trust_remote_code=True).to(device)
        except Exception as e_local:
            print(f"Local Phikon not found, trying online download: {e_local}")
            old_off = os.environ.get("TRANSFORMERS_OFFLINE")
            old_hf = os.environ.get("HF_HUB_OFFLINE")
            if "TRANSFORMERS_OFFLINE" in os.environ:
                del os.environ["TRANSFORMERS_OFFLINE"]
            if "HF_HUB_OFFLINE" in os.environ:
                del os.environ["HF_HUB_OFFLINE"]
            try:
                processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2", trust_remote_code=True)
                model = AutoModel.from_pretrained("owkin/phikon-v2", output_attentions=True, trust_remote_code=True).to(device)
            finally:
                if old_off is not None:
                    os.environ["TRANSFORMERS_OFFLINE"] = old_off
                if old_hf is not None:
                    os.environ["HF_HUB_OFFLINE"] = old_hf
        if processor is None or model is None:
            raise RuntimeError("Phikon processor/model failed to load")
        model.eval()
        inputs = processor(images=raw_image, return_tensors="pt").to(device)
        print("Extracting deep attention matrices (rollout product)...")
        with torch.no_grad():
            outputs = model(**inputs)
        # CORRECTED: rollout product, not mean
        rollout = _attention_rollout_product(outputs.attentions)  # [B, N, N]
        # CLS token attention to patches: rollout[0, 0, 1:]
        cls_attention = rollout[0, 0, 1:].cpu().numpy()  # [196]
        grid_size = int(np.sqrt(cls_attention.shape[0]))
        attention_grid = cls_attention.reshape(grid_size, grid_size)
        # Normalize
        attention_grid = (attention_grid - attention_grid.min()) / (attention_grid.max() - attention_grid.min() + 1e-8)
        smoothed_grid = cv2.GaussianBlur(attention_grid, (5, 5), 0)
        smoothed_grid = (smoothed_grid - smoothed_grid.min()) / (smoothed_grid.max() - smoothed_grid.min() + 1e-8)
        # Tissue mask for suppression
        arr = np.array(raw_image)
        tissue_mask = _get_tissue_mask(arr)
        dark_base = (arr.astype(np.float32) * 0.45).astype(np.uint8)
        # Resize smoothed_grid to 224 with tissue masking
        heatmap_small = smoothed_grid.copy()
        # Dynamic threshold on tissue values only
        # First resize to full for threshold calc to avoid grid bias
        full_heat = cv2.resize(smoothed_grid, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_CUBIC)
        # Suppress white in full_heat before threshold
        full_heat[~tissue_mask] = 0
        tissue_vals = full_heat[tissue_mask] if tissue_mask.any() else full_heat.flatten()
        if len(tissue_vals) > 0 and tissue_vals.max() > 0:
            dyn_thr = np.percentile(tissue_vals[tissue_vals > 0], 68) if (tissue_vals > 0).any() else np.percentile(tissue_vals, 68)
        else:
            dyn_thr = np.percentile(full_heat, 68)
        heatmap_for_overlay = full_heat.copy()
        heatmap_for_overlay[heatmap_for_overlay < dyn_thr] = 0
        heatmap_for_overlay[~tissue_mask] = 0
        heat_uint8 = (np.clip(heatmap_for_overlay, 0, 1) * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
        heat_color = (heat_color.astype(np.float32) * 0.92).astype(np.uint8)
        overlay = cv2.addWeighted(dark_base, 0.48, heat_color, 0.68, 0)
        overlay[~tissue_mask] = dark_base[~tissue_mask]
        # Contours
        thr = np.percentile(tissue_vals, 80) if len(tissue_vals) > 0 else np.percentile(full_heat, 80)
        mask = ((full_heat >= thr) & tissue_mask).astype(np.uint8) * 255
        blurred_mask = cv2.GaussianBlur(mask, (13, 13), 0)
        _, binary = cv2.threshold(blurred_mask, 127, 1, cv2.THRESH_BINARY)
        contour_img = dark_base.copy()
        contour_img = cv2.addWeighted(contour_img, 0.92, np.zeros_like(contour_img), 0, 10)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_img, contours, -1, (255, 225, 0), 2, lineType=cv2.LINE_AA)

        print("Generating DARK tissue-aware visual output (rollout product)...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor("#0B0B0F")
        fig.suptitle(f"Microscopic Attention — Phikon-v2 Rollout Product\n{filename_only} (tissue-aware, dark Jet)", fontsize=12, fontweight='bold', y=0.98, color="#E2E8F0")
        axes[0].imshow(dark_base)
        axes[0].set_title("Darkened H&E Patch (0.45×)", fontweight='bold', pad=12, color="#CBD5E1")
        axes[0].axis('off')
        axes[1].imshow(contour_img)
        axes[1].set_title("ROI Contours — Top 20% Tissue (Dark)", fontweight='bold', pad=12, color="#FDE68A")
        axes[1].axis('off')
        axes[2].imshow(overlay)
        axes[2].set_title("Dark Jet Tissue Overlay α=0.68", fontweight='bold', color="#F87171", pad=12)
        axes[2].axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor="#0B0B0F")
        plt.close()
        print(f"SUCCESS! Dark tissue-aware Phikon heatmap saved -> {output_path}")
        return output_path
    except Exception as e:
        print(f"Phikon path failed ({e}), using corrected dark fallback...")
        import traceback
        traceback.print_exc()
        try:
            raw_image_fallback = Image.open(img_path).convert('RGB')
            return _dark_fallback_heatmap(img_path, raw_image_fallback, output_path)
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            traceback.print_exc()
            return None

if __name__ == "__main__":
    target = get_interactive_input()
    if target:
        generate_smooth_vit_heatmap(target)
