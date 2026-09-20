#!/usr/bin/env python3
"""
Professional AIML Full-Stack Project Diagrams
Colorectal Cancer MSI Detection via Multimodal Fusion
Generates 10 production-quality diagrams via matplotlib + graphviz-free matplotlib rendering
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Ellipse
import numpy as np
import textwrap
import os

OUT = "diagrams"
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────
# Style Config — Professional palette
# ─────────────────────────────────────────────
COLORS = {
    "primary": "#0B2C4A",      # Deep navy
    "accent": "#0EA5E9",       # Sky blue
    "accent2": "#F59E0B",      # Amber
    "success": "#10B981",      # Emerald
    "danger": "#EF4444",       # Red
    "purple": "#8B5CF6",
    "teal": "#14B8A6",
    "slate": "#64748B",
    "light": "#F1F5F9",
    "card": "#FFFFFF",
    "border": "#E2E8F0",
}
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
})

def save(fig, name, dpi=300):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ {path}")

def rounded_box(ax, xy, w, h, text, face="#FFFFFF", edge=COLORS["primary"], fontsize=8, textcolor="#0B2C4A", bold=False, alpha=1):
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.015,rounding_size=0.02", facecolor=face, edgecolor=edge, linewidth=1.4, alpha=alpha)
    ax.add_patch(box)
    cx, cy = xy[0]+w/2, xy[1]+h/2
    lines = text.split("\n")
    # auto wrap
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, color=textcolor,
            weight="bold" if bold else "normal", linespacing=1.35)
    return box

def arrow(ax, p1, p2, color=COLORS["primary"], width=1.5, style="-", label=None, label_bg="white", ls="-"):
    arr = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=10, color=color, linewidth=width, linestyle=ls, zorder=5)
    ax.add_patch(arr)
    if label:
        mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
        ax.text(mx, my, label, ha="center", va="center", fontsize=6.5, color=color, weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=label_bg, edgecolor=color, alpha=0.9))

# ═══════════════════════════════════════════════
# 1. SYSTEM ARCHITECTURE — 4-PHASE PIPELINE
# ═══════════════════════════════════════════════
def diagram_system_architecture():
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F8FAFC")

    # Title
    ax.text(8.5, 9.6, "SYSTEM ARCHITECTURE — 4-PHASE MULTIMODAL MSI DETECTION PIPELINE", ha="center", fontsize=14, weight="bold", color=COLORS["primary"])
    ax.text(8.5, 9.25, "TCGA-COAD • Phikon-v2 (ViT-L/16) • ClinicalBERT • SummaryToPatchCrossAttention • Streamlit + Ollama Qwen2.5", ha="center", fontsize=7.5, color=COLORS["slate"])
    ax.plot([0.3,16.7],[9.05,9.05], color=COLORS["border"], lw=1)

    # PHASE columns
    phases = [
        ("PHASE 1\nDATA CURATION", "#EFF6FF", "#3B82F6", 0.4, 1.2, [
            ("Kaggle Dataset\ntcga_coad_msi_mss_jpg\n326 Patients\n192k Patches (224×224)", 0.6, 7.9, 2.4, 0.95),
            ("Patient Registry\npatient_registry.json\nlabel • report_text\nimage_paths • patch_count", 0.6, 6.65, 2.4, 0.95),
            ("Preprocessing\nReinhard Stain Norm\nOtsu Foreground Mask\n10× Magnification", 0.6, 5.4, 2.4, 0.95),
        ]),
        ("PHASE 2\nFEATURE ENGINEERING\n(Frozen Foundation Encoders)", "#F0FDF4", "#10B981", 3.5, 1.2, [
            ("Vision Encoder\nPhikon-v2 (DINOv2 ViT-L/16)\n307M params • 460M histology images\nV ∈ ℝ^{N×1024}  [N≤1000]", 3.7, 7.9, 2.6, 0.95),
            ("Text Encoder\nClinicalBERT (110M)\n2M MIMIC-III notes\nT ∈ ℝ^{L×768}  [L≤512]\n[CLS] summary", 3.7, 6.65, 2.6, 0.95),
            ("Tensor Store\nclean_tensors/*.pt\n*_vision.pt  *_text.pt\n653 files • 326 patients ×2", 3.7, 5.4, 2.6, 0.95),
        ]),
        ("PHASE 3\nFUSION & LEARNING", "#FEF3C7", "#F59E0B", 6.9, 1.2, [
            ("Late Fusion (Baseline)\nAB-MIL Attention → [1×1024]\nText Proj 768→512\nConcat → MLP 1536→256→1", 7.1, 7.9, 2.6, 0.95),
            ("★ Cross-Attention (Primary)\nSummaryToPatchCrossAttention\nText 768→1024 Proj\nQ=text, K/V=patches\n8-head MHA + LayerNorm", 7.1, 6.65, 2.6, 0.95),
            ("Training\nWeighted BCE (pos=4.17)\nAdamW • lr=1e-4 • 30 epochs\nThreshold τ = 0.83", 7.1, 5.4, 2.6, 0.95),
        ]),
        ("PHASE 4\nINFERENCE & XAI\n(Clinical Deployment)", "#FDF2F8", "#EC4899", 10.2, 1.2, [
            ("Macroscopic XAI\nCross-Attention Weights\nTop-5 Patch Ranking\nRed-border gallery", 10.4, 7.9, 2.5, 0.95),
            ("Microscopic XAI\nPhikon-v2 Attention Rollout\nLast 6 layers avg • Gaussian blur\nJet overlay + ROI contours", 10.4, 6.65, 2.5, 0.95),
            ("LLM Scribe + UI\nQwen 2.5 via Ollama\n3-paragraph clinical report\nStreamlit Dashboard", 10.4, 5.4, 2.5, 0.95),
        ]),
    ]

    # Draw phase headers and boxes
    for title, bg, accent, x0, y0, boxes in phases:
        # Header
        hdr = FancyBboxPatch((x0, 8.55), 2.85, 0.45, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor=accent, edgecolor=accent)
        ax.add_patch(hdr)
        ax.text(x0+1.425, 8.77, title, ha="center", va="center", fontsize=6.5, weight="bold", color="white", linespacing=1.2)
        # Background panel
        panel = FancyBboxPatch((x0-0.05, 5.15), 2.95, 3.30, boxstyle="round,pad=0.02,rounding_size=0.04", facecolor=bg, edgecolor=accent, linewidth=1.2, alpha=0.5)
        ax.add_patch(panel)
        for txt, bx, by, bw, bh in boxes:
            rounded_box(ax, (bx, by), bw, bh, txt, face="white", edge=accent, fontsize=6.2)

    # Inter-phase arrows
    arrow(ax, (3.42, 7.05), (3.5, 7.05), color=COLORS["primary"], width=1.8)
    arrow(ax, (6.42, 7.05), (6.9, 7.05), color=COLORS["primary"], width=1.8)
    arrow(ax, (9.72, 7.05), (10.2, 7.05), color=COLORS["primary"], width=1.8)

    # Bottom bar: Data contract + metrics
    yb = 3.8
    # Data contract banner
    banner = FancyBboxPatch((0.4, yb+0.75), 16.2, 0.55, boxstyle="round,pad=0.015,rounding_size=0.03", facecolor=COLORS["primary"], edgecolor=COLORS["primary"])
    ax.add_patch(banner)
    ax.text(8.5, yb+1.02, "DATA CONTRACT & VALIDATION — actual_validation_keys.json  •  66 isolated patients (50 MSS / 16 MSI)  •  Zero data leakage  •  Stratified 80/20 split", ha="center", va="center", fontsize=7, weight="bold", color="white")

    # Metrics row
    metrics = [
        ("89.39%", "Accuracy", COLORS["primary"]),
        ("0.9150", "ROC-AUC", COLORS["accent"]),
        ("0.80 / 0.75", "MSI Prec / Rec", COLORS["success"]),
        ("0.77", "MSI F1", COLORS["purple"]),
        ("30 epochs", "Training", COLORS["accent2"]),
        ("τ = 0.83", "Decision Threshold", COLORS["danger"]),
    ]
    x = 0.6
    for val, lab, col in metrics:
        b = FancyBboxPatch((x, yb+0.05), 2.45, 0.55, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor="white", edgecolor=col, linewidth=1.3)
        ax.add_patch(b)
        ax.text(x+1.225, yb+0.36, val, ha="center", fontsize=8, weight="bold", color=col)
        ax.text(x+1.225, yb+0.16, lab, ha="center", fontsize=6, color=COLORS["slate"], weight="bold")
        x += 2.65

    # Technology stack footer
    ax.text(8.5, 3.55, "STACK:  PyTorch 2.9  •  HuggingFace Transformers (Phikon-v2 / ClinicalBERT)  •  Streamlit 1.52  •  Ollama + Qwen 2.5-Coder:7B  •  OpenCV  •  TCGA-COAD", ha="center", fontsize=6.5, color=COLORS["slate"], style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["light"], edgecolor=COLORS["border"]))

    # Infra note
    ax.text(0.4, 0.15, "
    ax.text(16.7, 0.15, "Architecture v1.2 — Production Ready", fontsize=5.5, color=COLORS["slate"], ha="right")

    save(fig, "01_system_architecture.png")

# ═══════════════════════════════════════════════
# 2. DATA FLOW DIAGRAM (DFD Level 0 + Level 1)
# ═══════════════════════════════════════════════
def diagram_data_flow():
    fig, axes = plt.subplots(1,2, figsize=(18, 9))
    fig.patch.set_facecolor("white")
    fig.suptitle("DATA FLOW DIAGRAM — Multimodal Colorectal Cancer MSI Detection", fontsize=13, weight="bold", color=COLORS["primary"], y=0.98)

    # ── LEFT: DFD Level 0 (Context) ──
    ax = axes[0]
    ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off")
    ax.set_title("DFD Level 0 — Context Diagram", fontsize=10, weight="bold", color=COLORS["primary"], pad=12)
    ax.set_facecolor("#F8FAFC")

    # Central process
    rounded_box(ax, (3.2, 4.3), 3.6, 1.6, "0\nMULTIMODAL MSI\nDETECTION SYSTEM\n(Phikon-v2 + ClinicalBERT\n+ Cross-Attention)", face=COLORS["primary"], edge=COLORS["primary"], fontsize=7, textcolor="white", bold=True)

    # External entities
    entities = [
        ("TCGA / Kaggle\ntcga_coad_msi_mss_jpg\n(H&E patches + pathology\nreports)", 0.3, 7.0, 2.4, 1.2, "#EFF6FF", "#3B82F6"),
        ("Pathologist /\nClinician\n(Streamlit UI)", 7.3, 7.0, 2.4, 1.2, "#FEF3C7", "#F59E0B"),
        ("Model Registry\n(weights: fusion_*.pth\npatient_registry.json)", 0.3, 1.0, 2.4, 1.2, "#F0FDF4", "#10B981"),
        ("Ollama LLM\nQwen 2.5-Coder:7B\n(Report Scribe)", 7.3, 1.0, 2.4, 1.2, "#FDF2F8", "#EC4899"),
    ]
    for txt, x, y, w, h, fc, ec in entities:
        rounded_box(ax, (x,y), w, h, txt, face=fc, edge=ec, fontsize=6.2, bold=True)

    # Flows
    arrow(ax, (2.7, 7.6), (3.2, 5.8), color="#3B82F6", label="D1: Raw JPEG + Reports", label_bg="#EFF6FF")
    arrow(ax, (5.0, 6.8), (7.3, 7.6), color="#F59E0B", ls="--", label="R1: Diagnosis + XAI", label_bg="#FEF3C7")
    arrow(ax, (7.3, 7.3), (6.8, 5.9), color="#F59E0B", label="Q1: Patient ID + Text", label_bg="#FEF3C7")
    arrow(ax, (2.7, 1.6), (3.2, 4.5), color="#10B981", ls="--", label="M1: Model Weights", label_bg="#F0FDF4")
    arrow(ax, (5.0, 4.3), (7.3, 1.8), color="#EC4899", label="C1: Context Package", label_bg="#FDF2F8")
    arrow(ax, (7.3, 1.5), (6.4, 4.3), color="#EC4899", ls="--", label="S1: Clinical Summary", label_bg="#FDF2F8")
    arrow(ax, (5.0, 4.9), (2.7, 1.9), color="#10B981", label="L1: Eval Metrics", label_bg="#F0FDF4")

    ax.text(5, 0.3, "External Entities: 4  •  Data Flows: 7  •  Trust Boundary: Hospital Network", ha="center", fontsize=6, color=COLORS["slate"], style="italic")

    # ── RIGHT: DFD Level 1 (Decomposed) ──
    ax = axes[1]
    ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off")
    ax.set_title("DFD Level 1 — Decomposed Processes & Data Stores", fontsize=10, weight="bold", color=COLORS["primary"], pad=12)
    ax.set_facecolor("#F8FAFC")

    # Processes (circles/rounded)
    procs = [
        ("1.0\nINGEST &\nPREPROCESS", 1.0, 8.0, "#EFF6FF", "#3B82F6"),
        ("2.0\nENCODE\nVISION", 1.0, 5.8, "#F0FDF4", "#10B981"),
        ("2.1\nENCODE\nTEXT", 1.0, 3.6, "#FDF2F8", "#EC4899"),
        ("3.0\nFUSE &\nCLASSIFY", 5.0, 5.8, "#FEF3C7", "#F59E0B"),
        ("4.0\nEXPLAIN\n(XAI)", 5.0, 3.2, "#EDE9FE", "#8B5CF6"),
        ("5.0\nSERVE\n(Streamlit)", 8.2, 5.8, "#FFF7ED", "#F97316"),
    ]
    for txt, x, y, fc, ec in procs:
        # ellipse-like via rounded box
        rounded_box(ax, (x, y), 1.7, 0.95, txt, face=fc, edge=ec, fontsize=5.8, bold=True)

    # Data stores (open rectangles)
    stores = [
        ("D1: patient_registry.json\n+ image_paths", 3.6, 8.6),
        ("D2: clean_tensors\n(*_vision.pt, *_text.pt)", 3.6, 7.0),
        ("D3: Model Weights\nfusion_model_v1_1.pth", 3.6, 4.2),
        ("D4: Validation Keys\nactual_validation_keys.json", 3.6, 2.5),
        ("D5: XAI Outputs\nTop-5 + Heatmaps", 6.9, 2.5),
    ]
    for txt, x, y in stores:
        # data store: parallel lines look via two rectangles
        r = FancyBboxPatch((x, y), 1.45, 0.55, boxstyle="round,pad=0.01,rounding_size=0.015", facecolor="white", edgecolor=COLORS["slate"], linewidth=1.1)
        ax.add_patch(r)
        # left thick bar
        ax.add_patch(Rectangle((x, y), 0.08, 0.55, facecolor=COLORS["slate"]))
        ax.text(x+0.78, y+0.275, txt, ha="center", va="center", fontsize=5, color=COLORS["primary"], weight="bold", linespacing=1.2)

    # Flow arrows L1
    ax.annotate("", xy=(1.85, 6.6), xytext=(3.6, 8.9), arrowprops=dict(arrowstyle="->", color="#3B82F6", lw=1.2))
    ax.text(2.5, 7.9, "patches\n+ reports", fontsize=4.5, color="#3B82F6", ha="center", weight="bold")
    ax.annotate("", xy=(3.6, 7.3), xytext=(2.7, 6.25), arrowprops=dict(arrowstyle="->", color="#10B981", lw=1.2))
    ax.text(2.9, 6.85, "224×224\nnormalized", fontsize=4.5, color="#10B981", ha="center", weight="bold")
    ax.annotate("", xy=(3.6, 7.25), xytext=(2.7, 4.05), arrowprops=dict(arrowstyle="->", color="#EC4899", lw=1.2))
    ax.text(3.0, 5.7, "tokenized\n512 tokens", fontsize=4.5, color="#EC4899", ha="center", weight="bold")
    # encodes to D2
    ax.annotate("", xy=(3.6, 7.0), xytext=(2.7, 5.9), arrowprops=dict(arrowstyle="->", color="#10B981", lw=1.2, ls="dashed"))
    ax.annotate("", xy=(3.6, 7.0), xytext=(2.7, 3.7), arrowprops=dict(arrowstyle="->", color="#EC4899", lw=1.2, ls="dashed"))
    ax.text(4.4, 7.55, "V: [N×1024]\nT: [L×768]", fontsize=4.5, color=COLORS["primary"], ha="center", weight="bold", bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=COLORS["border"]))
    # D2 to fuse
    ax.annotate("", xy=(5.0, 6.25), xytext=(5.05, 7.0), arrowprops=dict(arrowstyle="->", color=COLORS["primary"], lw=1.4))
    ax.text(4.8, 6.65, "tensors", fontsize=5, color=COLORS["primary"], ha="center", weight="bold")
    ax.annotate("", xy=(6.7, 5.9), xytext=(5.05, 4.2), arrowprops=dict(arrowstyle="->", color="#F59E0B", lw=1.1, ls="dashed"))
    ax.text(6.15, 5.15, "weights", fontsize=4.5, color="#F59E0B", ha="center")
    # fuse to XAI and serve
    ax.annotate("", xy=(5.0, 3.7), xytext=(5.85, 5.8), arrowprops=dict(arrowstyle="->", color="#8B5CF6", lw=1.4))
    ax.text(5.9, 4.9, "prob + attn\nweights", fontsize=4.5, color="#8B5CF6", ha="center", weight="bold", rotation=45)
    ax.annotate("", xy=(8.2, 6.25), xytext=(6.7, 5.9), arrowprops=dict(arrowstyle="->", color="#F97316", lw=1.4))
    ax.text(7.45, 6.45, "diagnosis", fontsize=5, color="#F97316", weight="bold", ha="center")
    # XAI to store and serve
    ax.annotate("", xy=(6.9, 2.8), xytext=(5.85, 3.3), arrowprops=dict(arrowstyle="->", color="#8B5CF6", lw=1.1, ls="dashed"))
    ax.annotate("", xy=(8.2, 6.0), xytext=(7.6, 2.8), arrowprops=dict(arrowstyle="->", color="#8B5CF6", lw=1.1, ls="dashed"))
    ax.text(8.3, 4.3, "heatmaps\nTop-5", fontsize=4.5, color="#8B5CF6", ha="center")

    ax.text(5, 0.3, "Processes: 6  •  Data Stores: 5  •  Batch Size = 1 (variable N handling)  •  Device agnostic: CUDA / MPS / CPU", ha="center", fontsize=6, color=COLORS["slate"], style="italic")

    plt.tight_layout()
    save(fig, "02_data_flow_diagram.png")

# ═══════════════════════════════════════════════
# 3. MODEL ARCHITECTURE COMPARISON
# ═══════════════════════════════════════════════
def diagram_model_architecture():
    fig, axes = plt.subplots(1,2, figsize=(18, 9.5))
    fig.patch.set_facecolor("white")
    fig.suptitle("MODEL ARCHITECTURE — Late Fusion (Baseline) vs. SummaryToPatchCrossAttention (Primary)", fontsize=12, weight="bold", color=COLORS["primary"], y=0.98)

    # ── LEFT: Late Fusion ──
    ax = axes[0]
    ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off")
    ax.set_facecolor("#FFF7ED")
    ax.set_title("A. Late Fusion Baseline  —  AB-MIL + Concat + MLP", fontsize=9, weight="bold", color="#9A3412", pad=10)
    # badge
    ax.text(8.9, 9.3, "BASELINE", ha="center", fontsize=6, weight="bold", color="white", bbox=dict(boxstyle="round,pad=0.3", facecolor="#F97316"))

    # Inputs
    rounded_box(ax, (0.4, 8.2), 2.2, 0.9, "INPUT A: Vision\nV ∈ ℝ^{1×N×1024}\nN patches (var., ≤1000)", face="#EFF6FF", edge="#3B82F6", fontsize=6, bold=True)
    rounded_box(ax, (7.4, 8.2), 2.2, 0.9, "INPUT B: Text\nT ∈ ℝ^{1×768}\n[CLS] vector", face="#FDF2F8", edge="#EC4899", fontsize=6, bold=True)

    # Vision AB-MIL branch
    rounded_box(ax, (0.4, 6.85), 2.2, 0.9, "AB-MIL Attention\nLinear(1024→512) → Tanh\nLinear(512→1) → Softmax\nA ∈ ℝ^{N×1}, ΣA=1.0", face="white", edge="#3B82F6", fontsize=5.5)
    arrow(ax, (1.5, 8.2), (1.5, 7.75), color="#3B82F6")
    ax.annotate("", xy=(1.5, 6.3), xytext=(1.5, 6.85), arrowprops=dict(arrowstyle="->", color="#3B82F6", lw=1.3))
    # Weighted sum
    rounded_box(ax, (0.4, 5.6), 2.2, 0.65, "Weighted Sum\nvision_summary = Aᵀ·V\n→ [1×1024]", face="#DBEAFE", edge="#3B82F6", fontsize=6, bold=True)
    # Text projector
    rounded_box(ax, (7.4, 6.85), 2.2, 0.9, "Text Projector\nLinear(768→512) → ReLU\nDropout(0.3)\ntext_summary [1×512]", face="white", edge="#EC4899", fontsize=5.5)
    arrow(ax, (8.5, 8.2), (8.5, 7.75), color="#EC4899")
    rounded_box(ax, (7.4, 5.6), 2.2, 0.65, "text_summary\n[1×512]", face="#FCE7F3", edge="#EC4899", fontsize=6, bold=True)
    ax.annotate("", xy=(8.5, 5.6), xytext=(8.5, 6.85), arrowprops=dict(arrowstyle="->", color="#EC4899", lw=1.3))

    # Concat
    arrow(ax, (1.5, 5.6), (3.6, 4.9), color="#3B82F6")
    arrow(ax, (8.5, 5.6), (6.4, 4.9), color="#EC4899")
    rounded_box(ax, (3.6, 4.45), 2.8, 0.9, "CONCATENATE\nfused = [vision_summary ; text_summary]\n→ [1×1536]  (1024+512)", face="#FEF3C7", edge="#F59E0B", fontsize=6, bold=True)
    ax.text(5, 4.0, "Static fusion — no inter-modal interaction", ha="center", fontsize=5.5, color=COLORS["slate"], style="italic")

    # Classifier
    arrow(ax, (5, 4.45), (5, 3.7), color="#F59E0B")
    rounded_box(ax, (3.0, 2.3), 4.0, 1.15, "CLASSIFIER HEAD\nLinear(1536→256) → ReLU → Dropout(0.3)\nLinear(256→1) → Sigmoid → prob ∈ [0,1]\nThreshold τ = 0.70 → MSI / MSS", face="white", edge="#F59E0B", fontsize=6, bold=True)

    # XAI note
    ax.text(5, 1.55, "XAI: Returns prob + AB-MIL weights A (patch ranking)", ha="center", fontsize=5.5, color=COLORS["slate"],
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFBEB", edgecolor="#F59E0B"))
    # Perf badge
    perf = FancyBboxPatch((1.5, 0.4), 7.0, 0.65, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor="#FFF7ED", edgecolor="#F97316", linewidth=1.2)
    ax.add_patch(perf)
    ax.text(5, 0.72, "PERFORMANCE:  Acc 81.82%  •  AUC 0.895  •  MSI F1 0.50 (Recall 0.38 — MISSES 62% of MSI)", ha="center", fontsize=6, weight="bold", color="#9A3412")

    # Limitation callout
    ax.text(0.4, 9.55, "⚠  ISOLATED BRANCHES — text cannot guide vision", fontsize=5, color="#DC2626", weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEE2E2", edgecolor="#FECACA"))

    # ── RIGHT: Cross-Attention ──
    ax = axes[1]
    ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off")
    ax.set_facecolor("#F0FDF4")
    ax.set_title("B. SummaryToPatchCrossAttention  —  Intermediate Dynamic Fusion ★ PRIMARY", fontsize=9, weight="bold", color="#065F46", pad=10)
    ax.text(8.9, 9.3, "PRIMARY ★", ha="center", fontsize=6, weight="bold", color="white", bbox=dict(boxstyle="round,pad=0.3", facecolor="#10B981"))

    # Inputs
    rounded_box(ax, (0.4, 8.2), 2.2, 0.9, "INPUT A: Vision\nV ∈ ℝ^{B×N×1024}\nN patches (uncompressed)", face="#EFF6FF", edge="#3B82F6", fontsize=6, bold=True)
    rounded_box(ax, (7.4, 8.2), 2.2, 0.9, "INPUT B: Text\nT ∈ ℝ^{B×768}\n[CLS] → unsqueeze [B×1×768]", face="#FDF2F8", edge="#EC4899", fontsize=6, bold=True)

    # Projection
    arrow(ax, (8.5, 8.2), (8.5, 7.9), color="#EC4899")
    rounded_box(ax, (6.8, 6.95), 3.4, 0.75, "① TEXT PROJECTION\nLinear(768→1024) — Xavier Init\nT' = T · W_proj  →  [B×1×1024]\nShared latent space alignment", face="white", edge="#EC4899", fontsize=5.5)
    # Shape handling
    rounded_box(ax, (0.4, 6.95), 3.4, 0.75, "SHAPE ALIGNMENT\nif dim==2: unsqueeze(0) → [1×N×1024]\nBatch-first, no padding", face="white", edge="#3B82F6", fontsize=5.5)

    # Central Cross-Attention
    arrow(ax, (2.1, 6.95), (3.8, 6.1), color="#3B82F6")
    arrow(ax, (8.5, 6.95), (6.2, 6.1), color="#EC4899")
    # Big cross-attention box
    outer = FancyBboxPatch((2.2, 4.3), 5.6, 2.0, boxstyle="round,pad=0.02,rounding_size=0.05", facecolor=COLORS["primary"], edgecolor=COLORS["primary"], linewidth=1.5)
    ax.add_patch(outer)
    ax.text(5, 6.05, "② MULTI-HEAD CROSS-ATTENTION  (nn.MultiheadAttention • 8 heads • batch_first=True)", ha="center", fontsize=6, weight="bold", color="white")
    # Q K V inside
    # Q
    qb = FancyBboxPatch((2.5, 5.35), 1.5, 0.45, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor="#FCE7F3", edgecolor="white", linewidth=1)
    ax.add_patch(qb)
    ax.text(3.25, 5.57, "Q = T'  [B×1×1024]", ha="center", fontsize=5, weight="bold", color="#9D174D")
    # K V
    kb = FancyBboxPatch((4.25, 5.35), 1.5, 0.45, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor="#DBEAFE", edgecolor="white", linewidth=1)
    ax.add_patch(kb)
    ax.text(5.0, 5.57, "K = V  [B×N×1024]", ha="center", fontsize=5, weight="bold", color="#1E40AF")
    vb = FancyBboxPatch((6.0, 5.35), 1.5, 0.45, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor="#DBEAFE", edgecolor="white", linewidth=1)
    ax.add_patch(vb)
    ax.text(6.75, 5.57, "V = V  [B×N×1024]", ha="center", fontsize=5, weight="bold", color="#1E40AF")
    # Formula
    ax.text(5, 4.95, "Attn(Q,K,V) = softmax( QKᵀ / √1024 ) · V", ha="center", fontsize=6.5, weight="bold", color="#FDE68A", style="italic",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#1E293B", edgecolor="#334155"))
    ax.text(5, 4.58, "→ attn_output [B×1×1024]   +   attn_weights [B×1×N]  (interpretable!)", ha="center", fontsize=5.2, color="white")
    ax.text(5, 4.35, "Text query searches WSI patches for structural evidence", ha="center", fontsize=5, color="#93C5FD", style="italic")

    # Residual + LayerNorm
    arrow(ax, (5, 4.3), (5, 3.85), color="white")
    rounded_box(ax, (2.2, 3.15), 5.6, 0.55, "③ RESIDUAL + LAYERNORM  —  fused = LayerNorm( T' + attn_output )  →  [B×1×1024]", face="#FEF3C7", edge="#F59E0B", fontsize=5.5, bold=True)
    arrow(ax, (5, 3.15), (5, 2.85), color="#F59E0B")

    # Pool + Classifier
    rounded_box(ax, (2.2, 1.75), 5.6, 0.95, "④ SQUEEZE + CLASSIFIER HEAD  —  pooled [B×1024] → Linear(1024→256) → ReLU → Dropout(0.3) → Linear(256→1) → Sigmoid", face="white", edge="#10B981", fontsize=5.5, bold=True)
    ax.text(5, 1.45, "Output:  prob ∈ [0,1]  +  attn_weights  •  Decision: MSI if prob ≥ 0.83 else MSS", ha="center", fontsize=5.5, color=COLORS["slate"],
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#ECFDF5", edgecolor="#6EE7B7"))

    # Perf
    perf2 = FancyBboxPatch((1.2, 0.35), 7.6, 0.65, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor="#ECFDF5", edgecolor="#10B981", linewidth=1.2)
    ax.add_patch(perf2)
    ax.text(5, 0.67, "PERFORMANCE:  Acc 89.39%  •  AUC 0.9150  •  MSI F1 0.77 (Recall 0.75 — catches 2× more MSI!)", ha="center", fontsize=6, weight="bold", color="#065F46")
    ax.text(0.4, 9.55, "✓  DYNAMIC TEXT-GUIDED VISUAL SEARCH — perturbation-verified", fontsize=5, color="#065F46", weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#D1FAE5", edgecolor="#A7F3D0"))

    plt.tight_layout()
    save(fig, "03_model_architecture_comparison.png")

# ═══════════════════════════════════════════════
# 4. SEQUENCE DIAGRAM (Training + Inference)
# ═══════════════════════════════════════════════
def diagram_sequence():
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F8FAFC")
    ax.text(9, 9.7, "SEQUENCE DIAGRAMS — Training Pipeline (Top) & Inference Pipeline (Bottom)", ha="center", fontsize=12, weight="bold", color=COLORS["primary"])
    ax.plot([0.3,17.7],[9.4,9.4], color=COLORS["border"], lw=1)

    # ── TOP: Training Sequence ──
    ax.text(0.5, 9.0, "A. TRAINING PHASE — 30 Epochs • Weighted BCE • AdamW", fontsize=7.5, weight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["primary"]))

    actors = ["DataLoader\n(Batch=1)", "Multimodal\nDataset", "Phikon-v2\n(Frozen)", "ClinicalBERT\n(Frozen)", "Cross-Attention\nModel", "Weighted\nBCE Loss", "AdamW\nOptimizer"]
    xs = [1, 3.5, 6, 8.3, 10.8, 13.3, 15.5]
    # lifelines
    for x, name in zip(xs, actors):
        # actor box
        rounded_box(ax, (x-0.65, 8.05), 1.3, 0.55, name, face="white", edge=COLORS["primary"], fontsize=5, bold=True)
        # lifeline
        ax.plot([x, x], [8.05, 5.25], color=COLORS["slate"], lw=0.8, linestyle="--", alpha=0.6)

    # Messages
    # 1. Load
    ax.annotate("", xy=(3.5, 7.7), xytext=(1, 7.7), arrowprops=dict(arrowstyle="->", color=COLORS["primary"], lw=1.2))
    ax.text(2.25, 7.85, "1. __getitem__(idx) → {vision [N×1024], text [768], label}", ha="center", fontsize=4.8, color=COLORS["primary"], weight="bold", bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=COLORS["border"]))
    # 2. dynamic cap
    ax.text(2.25, 7.45, "2. Dynamic cap: random sample 1000 if N>1000", ha="center", fontsize=4.5, color=COLORS["slate"], style="italic")
    ax.plot([1,3.5],[7.45,7.45], color=COLORS["slate"], lw=0.7, linestyle=":")
    # 3. forward vision text (merged for space)
    ax.annotate("", xy=(10.8, 7.2), xytext=(1, 7.2), arrowprops=dict(arrowstyle="->", color="#3B82F6", lw=1.2))
    ax.text(5.9, 7.35, "3. vision [1×N×1024] + text [1×768]  →  forward()", ha="center", fontsize=4.8, color="#3B82F6", weight="bold", bbox=dict(boxstyle="round,pad=0.15", facecolor="#EFF6FF", edgecolor="#93C5FD"))
    # 4. inside model: projection + cross-attn
    ax.text(10.8, 6.95, "4. Text Proj 768→1024 + 8-head Cross-Attn + LayerNorm + Classifier", ha="center", fontsize=4.5, color=COLORS["primary"], style="italic")
    # 5. prob output
    ax.annotate("", xy=(13.3, 6.7), xytext=(10.8, 6.7), arrowprops=dict(arrowstyle="->", color="#F59E0B", lw=1.2))
    ax.text(12.05, 6.85, "5. prob + attn_weights", ha="center", fontsize=4.8, color="#92400E", weight="bold", bbox=dict(boxstyle="round,pad=0.15", facecolor="#FEF3C7", edgecolor="#FCD34D"))
    # 6. loss
    ax.annotate("", xy=(13.3, 6.4), xytext=(10.8, 6.4), arrowprops=dict(arrowstyle="->", color=COLORS["danger"], lw=1.2))
    ax.text(12.05, 6.25, "6. L = -[4.17·y·log(p)+(1-y)·log(1-p)]", ha="center", fontsize=4.5, color=COLORS["danger"], weight="bold")
    # 7. backward
    ax.annotate("", xy=(15.5, 6.0), xytext=(13.3, 6.0), arrowprops=dict(arrowstyle="->", color="#10B981", lw=1.2))
    ax.text(14.4, 6.15, "7. loss.backward()", ha="center", fontsize=4.5, color="#065F46")
    ax.annotate("", xy=(10.8, 5.7), xytext=(15.5, 5.7), arrowprops=dict(arrowstyle="->", color="#10B981", lw=1.2))
    ax.text(13.15, 5.85, "8. optimizer.step() — update Proj+MHA+MLP (encoders frozen)", ha="center", fontsize=4.8, color="#065F46", weight="bold", bbox=dict(boxstyle="round,pad=0.15", facecolor="#ECFDF5", edgecolor="#6EE7B7"))
    # val loop
    ax.text(9, 5.35, "⟲  Loop: Training batches (261 patients)  →  Validation (65) — early stopping on best AUC  →  Save best_fusion_model.pth", ha="center", fontsize=5, color=COLORS["slate"], weight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=COLORS["light"], edgecolor=COLORS["border"]))

    # ── BOTTOM: Inference Sequence ──
    ax.text(0.5, 5.0, "B. INFERENCE PHASE — Streamlit → Model → XAI → LLM Scribe", fontsize=7.5, weight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#8B5CF6"))
    actors2 = ["Clinician\n(Browser)", "Streamlit\nUI (app.py)", "Cross-Attention\nModel", "Phikon-v2\nViT Rollout", "Ollama\nQwen 2.5", "File System\n(Assets)"]
    xs2 = [1.2, 4.0, 7.0, 10.0, 13.0, 16.0]
    for x, name in zip(xs2, actors2):
        rounded_box(ax, (x-0.65, 4.05), 1.3, 0.55, name, face="white", edge="#8B5CF6", fontsize=5, bold=True)
        ax.plot([x, x], [4.05, 0.5], color=COLORS["slate"], lw=0.8, linestyle="--", alpha=0.6)

    # Inf messages
    ax.annotate("", xy=(4.0, 3.75), xytext=(1.2, 3.75), arrowprops=dict(arrowstyle="->", color="#8B5CF6", lw=1.2))
    ax.text(2.6, 3.9, "1. Enter Patient ID + edit report", ha="center", fontsize=4.8, color="#8B5CF6", weight="bold")
    ax.annotate("", xy=(7.0, 3.5), xytext=(4.0, 3.5), arrowprops=dict(arrowstyle="->", color=COLORS["primary"], lw=1.2))
    ax.text(5.5, 3.65, "2. Load *_vision.pt + *_text.pt → forward()", ha="center", fontsize=4.8, color=COLORS["primary"], weight="bold", bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=COLORS["border"]))
    ax.annotate("", xy=(4.0, 3.25), xytext=(7.0, 3.25), arrowprops=dict(arrowstyle="->", color=COLORS["success"], lw=1.2))
    ax.text(5.5, 3.1, "3. Return prob=0.87 + attn_weights [1×N]", ha="center", fontsize=4.8, color=COLORS["success"], weight="bold")
    # branching
    ax.text(7.0, 2.85, "4a. argsort → Top-5 indices → Load JPEG patches → Display gallery", ha="center", fontsize=4.5, color="#3B82F6", style="italic")
    ax.annotate("", xy=(10.0, 2.6), xytext=(7.0, 2.6), arrowprops=dict(arrowstyle="->", color="#3B82F6", lw=1.1, ls="dashed"))
    ax.text(8.5, 2.45, "4b. Selected patch → Phikon-v2 attn rollout (6 layers avg)", ha="center", fontsize=4.5, color="#3B82F6", style="italic")
    ax.annotate("", xy=(10.0, 2.25), xytext=(10.0, 2.35), arrowprops=dict(arrowstyle="->", color=COLORS["primary"], lw=1.1))
    ax.text(10.0, 2.1, "Jet overlay + ROI contours", ha="center", fontsize=4.5, color=COLORS["primary"], weight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#EFF6FF", edgecolor="#93C5FD"))
    # LLM
    ax.annotate("", xy=(13.0, 2.0), xytext=(7.0, 2.0), arrowprops=dict(arrowstyle="->", color="#EC4899", lw=1.2))
    ax.text(10.0, 1.85, "5. Context pkg {prob, top-5, report_text} → POST /api/generate (Qwen2.5)", ha="center", fontsize=4.8, color="#EC4899", weight="bold", bbox=dict(boxstyle="round,pad=0.15", facecolor="#FDF2F8", edgecolor="#F9A8D4"))
    ax.annotate("", xy=(4.0, 1.6), xytext=(13.0, 1.6), arrowprops=dict(arrowstyle="->", color="#EC4899", lw=1.2, ls="dashed"))
    ax.text(8.5, 1.45, "6. 3-paragraph diagnostic summary (deterministic prompt, no hallucination)", ha="center", fontsize=4.5, color="#9D174D", style="italic")
    # Final render
    ax.annotate("", xy=(1.2, 1.25), xytext=(4.0, 1.25), arrowprops=dict(arrowstyle="->", color=COLORS["primary"], lw=1.4))
    ax.text(2.6, 1.1, "7. Render: Probability meter + Top-5 gallery + Heatmap + Scribe report", ha="center", fontsize=5, color=COLORS["primary"], weight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#EDE9FE", edgecolor="#C4B5FD"))

    ax.text(9, 0.2, "Threshold τ: 0.83 (Cross-Attn) vs 0.70 (Late Fusion)  •  Device: Auto-detect CUDA → MPS → CPU  •  Batch=1, variable N preserved", ha="center", fontsize=5.5, color=COLORS["slate"], style="italic")

    save(fig, "04_sequence_diagram.png")

# ═══════════════════════════════════════════════
# 5. DEPLOYMENT ARCHITECTURE
# ═══════════════════════════════════════════════
def diagram_deployment():
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.set_xlim(0,17); ax.set_ylim(0,10); ax.axis("off")
    fig.patch.set_facecolor("white"); ax.set_facecolor("#F8FAFC")
    ax.text(8.5, 9.7, "DEPLOYMENT ARCHITECTURE — Streamlit Clinical Deployment & Infrastructure", ha="center", fontsize=13, weight="bold", color=COLORS["primary"])
    ax.plot([0.3,16.7],[9.4,9.4], color=COLORS["border"], lw=1)

    # ── Client Tier ──
    rounded_box(ax, (0.5, 7.6), 3.2, 1.45, "CLIENT TIER\nClinician Workstation\nChrome / Safari / Edge\nNo install — Browser only\nPatient ID input + Report edit", face="#EFF6FF", edge="#3B82F6", fontsize=6.5, bold=True)
    ax.text(2.1, 7.35, "HTTPS :8501", ha="center", fontsize=5, color="#3B82F6", weight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#93C5FD"))
    # Arrow to app
    arrow(ax, (3.7, 8.3), (5.5, 8.3), color="#3B82F6", width=2)

    # ── App Tier (Streamlit) ──
    outer_app = FancyBboxPatch((5.5, 6.2), 6.5, 3.0, boxstyle="round,pad=0.02,rounding_size=0.05", facecolor="white", edgecolor=COLORS["primary"], linewidth=1.6)
    ax.add_patch(outer_app)
    ax.text(8.75, 8.95, "APPLICATION TIER — Streamlit Server (app.py)", ha="center", fontsize=7, weight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["primary"]))
    # Session state
    rounded_box(ax, (5.8, 8.15), 2.7, 0.65, "Session State\nanalysis_complete • prob\nattn_weights • scribe_report", face="#EDE9FE", edge="#8B5CF6", fontsize=5.5)
    rounded_box(ax, (8.9, 8.15), 3.6, 0.65, "Device Router\ncuda → mps → cpu auto-detect\ntorch.device dispatch", face="#FEF3C7", edge="#F59E0B", fontsize=5.5)
    # Inference
    rounded_box(ax, (5.8, 7.3), 6.0, 0.65, "Inference Engine: load_system() @st.cache_resource — SummaryToPatchCrossAttention + weights fusion_model_v1_1.pth", face="#ECFDF5", edge="#10B981", fontsize=5.5)
    # UI components
    comps = [
        ("Input Panel\nPatient ID • Slider N\nEditable text_area", 5.8, 6.55, 1.35, 0.5),
        ("Prob Meter\nMSI/MSS + Confidence\nTrue vs Predicted", 7.4, 6.55, 1.35, 0.5),
        ("Top-5 Gallery\n5-col grid • argsort\nattn_weights", 9.0, 6.55, 1.35, 0.5),
        ("Heatmap Tool\nPhikon-v2 rollout\n+ upload", 10.6, 6.55, 1.35, 0.5),
    ]
    for txt, x, y, w, h in comps:
        rounded_box(ax, (x,y), w, h, txt, face="white", edge="#8B5CF6", fontsize=4.8)

    # Arrow to data
    arrow(ax, (8.75, 6.2), (8.75, 5.7), color=COLORS["primary"], width=1.6)

    # ── Data Tier ──
    outer_data = FancyBboxPatch((3.0, 2.8), 11.0, 2.65, boxstyle="round,pad=0.02,rounding_size=0.05", facecolor="white", edgecolor="#64748B", linewidth=1.4, linestyle="--")
    ax.add_patch(outer_data)
    ax.text(8.5, 5.25, "DATA & MODEL TIER — Persistent Storage", ha="center", fontsize=7, weight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#64748B"))
    stores = [
        ("patient_registry.json\n326 patients\nlabel • report_text\nimage_paths[192k]", 3.4, 4.3, 2.2, 0.7, "#EFF6FF", "#3B82F6"),
        ("clean_tensors/\n653 × .pt files\nvision [N×1024]\ntext [768]", 6.0, 4.3, 2.2, 0.7, "#F0FDF4", "#10B981"),
        ("models/*.pth\nfusion_model_v1_1.pth\nlate_fusion_model.pth\n~5-20 MB each", 8.6, 4.3, 2.2, 0.7, "#FEF3C7", "#F59E0B"),
        ("assets / MSIMUT_JPEG\nMSS_JPEG\n224×224 patches\nJPEG", 11.2, 4.3, 2.2, 0.7, "#FDF2F8", "#EC4899"),
    ]
    for txt, x, y, w, h, fc, ec in stores:
        rounded_box(ax, (x,y), w, h, txt, face=fc, edge=ec, fontsize=5)

    # validation keys
    rounded_box(ax, (6.0, 3.25), 5.0, 0.45, "actual_validation_keys.json  •  66 isolated validation IDs  •  Enforces zero leakage contract", face=COLORS["light"], edge=COLORS["slate"], fontsize=5.5)

    # ── LLM Tier ──
    outer_llm = FancyBboxPatch((12.5, 6.2), 4.0, 3.0, boxstyle="round,pad=0.02,rounding_size=0.05", facecolor="#FDF2F8", edgecolor="#EC4899", linewidth=1.6)
    ax.add_patch(outer_llm)
    ax.text(14.5, 8.95, "LLM TIER — Ollama + Qwen2.5", ha="center", fontsize=7, weight="bold", color="white", bbox=dict(boxstyle="round,pad=0.3", facecolor="#EC4899"))
    rounded_box(ax, (12.8, 8.15), 3.4, 0.65, "Ollama Server\nlocalhost:11434\n/api/generate", face="white", edge="#EC4899", fontsize=5.5, bold=True)
    rounded_box(ax, (12.8, 7.35), 3.4, 0.65, "Qwen 2.5-Coder:7B\nDeterministic prompt\nNo hallucination", face="white", edge="#EC4899", fontsize=5.5, bold=True)
    rounded_box(ax, (12.8, 6.55), 3.4, 0.65, "3-Paragraph Output\nSummary • Findings\nRecommendation", face="white", edge="#EC4899", fontsize=5.5)
    # Arrows between app and LLM
    arrow(ax, (12.0, 7.6), (12.5, 7.6), color="#EC4899", label="POST prompt", label_bg="#FDF2F8")
    arrow(ax, (12.5, 7.3), (12.0, 7.3), color="#EC4899", ls="--", label="JSON response", label_bg="#FDF2F8")

    # ── Infra Footer ──
    infra_y = 1.7
    outer_infra = FancyBboxPatch((0.5, 0.4), 16.0, 1.9, boxstyle="round,pad=0.02,rounding_size=0.04", facecolor=COLORS["primary"], edgecolor=COLORS["primary"])
    ax.add_patch(outer_infra)
    ax.text(8.5, 2.05, "INFRASTRUCTURE & DEVOPS", ha="center", fontsize=7, weight="bold", color="#FDE68A")
    cols = [
        ("Runtime\nPython 3.10+\nPyTorch 2.9\nrun: streamlit run src/ui/app.py", 0.7, 1.3),
        ("Hardware\nCUDA / Apple MPS (M4)\nCPU Fallback\nMPS auto-detect", 3.5, 1.3),
        ("Dependencies\nrequirements.txt (75 pkgs)\ntimm • MONAI • smp\nstreamlit 1.52", 6.3, 1.3),
        ("Scaling Path\nDocker + Streamlit Cloud\nHF Spaces / EC2\nGPU inference", 9.1, 1.3),
        ("Monitoring\nEval reports/*.txt\nConfusion / ROC\nLoss trajectory", 11.9, 1.3),
        ("Security\nPHI stripped\nLocal-first (offline)\nNo cloud PHI leak", 14.2, 1.3),
    ]
    for txt, x, y in cols:
        # split title
        lines = txt.split("\n")
        ax.text(x, y+0.55, lines[0], fontsize=5.5, weight="bold", color="white", ha="left")
        ax.text(x, y+0.15, "\n".join(lines[1:]), fontsize=4.8, color="#CBD5E1", ha="left", linespacing=1.2)

    ax.text(0.5, 0.15, "Port: 8501  •  Start: streamlit run src/ui/app.py --server.port 8501  •  Ollama: ollama serve & ollama pull qwen2.5-coder:7b", fontsize=5, color="#94A3B8", style="italic")
    save(fig, "05_deployment_architecture.png")

# ═══════════════════════════════════════════════
# 6. COMPONENT / CLASS DIAGRAM
# ═══════════════════════════════════════════════
def diagram_component():
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.set_xlim(0,17); ax.set_ylim(0,10); ax.axis("off")
    fig.patch.set_facecolor("white"); ax.set_facecolor("#F8FAFC")
    ax.text(8.5, 9.7, "COMPONENT & CLASS DIAGRAM — Codebase Structure (src/)", ha="center", fontsize=12, weight="bold", color=COLORS["primary"])
    ax.plot([0.3,16.7],[9.4,9.4], color=COLORS["border"], lw=1)

    # Class boxes
    def class_box(x, y, w, h, title, attrs, methods, color):
        # header
        hdr = FancyBboxPatch((x, y+h-0.45), w, 0.45, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor=color, edgecolor=color)
        ax.add_patch(hdr)
        ax.text(x+w/2, y+h-0.22, title, ha="center", va="center", fontsize=6.5, weight="bold", color="white")
        # body
        body = FancyBboxPatch((x, y), w, h-0.45, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor="white", edgecolor=color, linewidth=1.3)
        ax.add_patch(body)
        # attrs
        ax.text(x+0.15, y+h-0.7, "Attributes:", fontsize=5, weight="bold", color=color)
        for i, a in enumerate(attrs):
            ax.text(x+0.3, y+h-0.95 - i*0.28, f"• {a}", fontsize=4.8, color=COLORS["primary"])
        # divider
        div_y = y+h-0.95 - len(attrs)*0.28 - 0.1
        ax.plot([x+0.1, x+w-0.1], [div_y, div_y], color=COLORS["border"], lw=0.6)
        ax.text(x+0.15, div_y-0.15, "Methods:", fontsize=5, weight="bold", color=color)
        for i, m in enumerate(methods):
            ax.text(x+0.3, div_y-0.4 - i*0.28, f"• {m}", fontsize=4.8, color=COLORS["slate"])

    # Dataset
    class_box(0.5, 5.2, 3.2, 3.6,
              "MultimodalTCGADataset\n(torch.utils.data.Dataset)",
              ["tensor_dir: str", "max_patches=1000: int", "valid_patients: List[Dict]", "registry: Dict[ID→meta]"],
              ["__init__(registry, tensor_dir)", "__len__() → int", "__getitem__(idx) → Dict", "collate: batch=1 (var N)"],
              "#3B82F6")
    # Late Fusion
    class_box(4.2, 5.2, 3.2, 3.6,
              "MultimodalFusionNetwork\n(nn.Module)",
              ["vision_attention: Sequential", "text_projector: Sequential", "classifier: Sequential", "hidden_dim=512"],
              ["forward(V[1×N×1024], T[1×768])", "→ prob, A[N×1]", "AB-MIL + concat 1536→1"],
              "#F59E0B")
    # Cross Attention
    class_box(7.9, 5.2, 3.6, 3.6,
              "SummaryToPatchCrossAttention\n(nn.Module) ★",
              ["text_projection: Linear 768→1024", "cross_attn: MultiheadAttn 8h", "layer_norm: LayerNorm 1024", "classifier: Seq 1024→256→1"],
              ["forward(V[B×N×1024], T[B×768])", "→ prob[B×1], attn[ B×1×N]", "Q=T', K/V=V, softmax/√1024"],
              "#10B981")
    # UI
    class_box(12.0, 5.2, 3.4, 3.6,
              "Streamlit App\n(app.py)",
              ["MODEL_WEIGHTS: str", "REGISTRY_PATH: str", "device: cuda/mps/cpu", "session_state: Dict"],
              ["load_system() @cache_resource", "generate_clinical_report()", "Run Diagnostic Pipeline", "Heatmap + Scribe UI"],
              "#8B5CF6")

    # Lower components
    class_box(0.5, 1.4, 3.2, 3.0,
              "Training Scripts\n(cross_attention_train.py)",
              ["REGISTRY, TENSORS, EPOCHS=30", "LR=1e-4, weight_decay=1e-4", "pos_weight=4.17"],
              ["set_seed(42)", "weighted_bce_loss()", "train_model() — save best AUC"],
              "#64748B")
    class_box(4.2, 1.4, 3.2, 3.0,
              "Evaluation Scripts\n(cross_attention_eval.py)",
              ["MODEL_WEIGHTS", "actual_validation_keys.json", "threshold=0.83"],
              ["evaluate_model()", "plot_cross_attention_matrix()", "Confusion + ROC"],
              "#64748B")
    class_box(7.9, 1.4, 3.6, 3.0,
              "XAI Engine\n(generate_pixel_xai.py)",
              ["Phikon-v2: 24 layers", "grid_size = √N", "GaussianBlur (5×5 & 15×15)"],
              ["generate_smooth_vit_heatmap()", "Layer avg last 6 → head avg", "Jet overlay + contours"],
              "#EC4899")
    class_box(12.0, 1.4, 3.4, 3.0,
              "Verification Suite\n(eval/)",
              ["verify_cross_attention.py", "verify_late_attention.py", "tensor_dim_check.py"],
              ["run_perturbation_test()", "Jensen-Shannon Divergence", "Real vs Fake text attn shift"],
              "#0EA5E9")

    # Relationships (arrows)
    arrow(ax, (3.7, 7.0), (4.2, 7.0), color=COLORS["slate"], width=1.1)
    ax.text(3.95, 7.15, "provides\ntensors", ha="center", fontsize=4.5, color=COLORS["slate"], weight="bold")
    arrow(ax, (7.4, 6.5), (7.9, 6.5), color=COLORS["slate"], width=1.1, ls="--")
    ax.text(7.65, 6.65, "evolves\nto", ha="center", fontsize=4.5, color=COLORS["slate"], weight="bold")
    arrow(ax, (11.5, 7.0), (12.0, 7.0), color="#8B5CF6", width=1.3)
    ax.text(11.75, 7.35, "inference", ha="center", fontsize=5, color="#8B5CF6", weight="bold", bbox=dict(boxstyle="round,pad=0.15", facecolor="#EDE9FE", edgecolor="#C4B5FD"))
    # vertical down
    arrow(ax, (2.1, 5.2), (2.1, 4.4), color=COLORS["slate"], width=1)
    arrow(ax, (5.8, 5.2), (5.8, 4.4), color=COLORS["slate"])
    ax.text(4.0, 4.75, "trained / evaluated by", ha="center", fontsize=5, color=COLORS["slate"], style="italic")
    arrow(ax, (9.7, 5.2), (9.7, 4.4), color="#EC4899")
    ax.text(8.5, 4.75, "explained by", ha="center", fontsize=5, color="#EC4899", style="italic")
    arrow(ax, (13.7, 5.2), (13.7, 4.4), color="#0EA5E9", ls="--")
    ax.text(14.5, 4.75, "verified", ha="center", fontsize=5, color="#0EA5E9", style="italic")

    # bottom note
    ax.text(8.5, 0.6, "Inheritance: nn.Module → MultimodalFusionNetwork, SummaryToPatchCrossAttention  •  Composition: Dataset → DataLoader(batch=1)  •  Encoders: FROZEN (no grad)  •  Trainable: Projection + MHA + LayerNorm + MLP Head", ha="center", fontsize=6, color=COLORS["slate"], style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLORS["border"]))
    save(fig, "06_component_class_diagram.png")

# ═══════════════════════════════════════════════
# 7. TRAINING & EVALUATION PIPELINE
# ═══════════════════════════════════════════════
def diagram_training_evaluation():
    fig, axes = plt.subplots(2,1, figsize=(17, 11))
    fig.patch.set_facecolor("white")
    fig.suptitle("TRAINING & EVALUATION PIPELINE — Optimization, Metrics & Loss Dynamics", fontsize=12, weight="bold", color=COLORS["primary"], y=0.98)

    # ── TOP: Training Flow ──
    ax = axes[0]
    ax.set_xlim(0,17); ax.set_ylim(0,5.2); ax.axis("off"); ax.set_facecolor("#F8FAFC")
    ax.text(0.3, 4.9, "TRAINING LOOP — 30 Epochs • 80/20 Split • Weighted BCE Convergence to 4.17", fontsize=7, weight="bold", color="white", bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["primary"]))

    steps = [
        ("1. Seed\nset_seed(42)\nrandom+np+torch\n+mps manual", 0.4, 3.0, 1.6, 1.3, "#EFF6FF", "#3B82F6"),
        ("2. Dataset:\n326 patients\nCounter MSS:263\nMSI:63 → w=4.17\nVerify .pt exist", 2.3, 3.0, 1.6, 1.3, "#F0FDF4", "#10B981"),
        ("3. Split:\n80% Train (261)\n20% Val (65)\nSave actual_\nvalidation_keys.json", 4.2, 3.0, 1.6, 1.3, "#FEF3C7", "#F59E0B"),
        ("4. DataLoader:\nbatch=1, shuffle\nmax_patches=1000\nrandperm sample", 6.1, 3.0, 1.6, 1.3, "#EDE9FE", "#8B5CF6"),
        ("5. Model:\nSummaryToPatch\n8-head MHA\nLayerNorm + MLP\n768→1024 proj", 8.0, 3.0, 1.6, 1.3, "#ECFDF5", "#10B981"),
        ("6. Forward:\nprob, attn\nClamp 1e-7\nWeighted BCE\n- [4.17·y·logp]", 9.9, 3.0, 1.6, 1.3, "#FEF3C7", "#F59E0B"),
        ("7. Backward:\nloss.backward()\nAdamW step\nlr=1e-4 • wd 1e-4\nDropout 0.3", 11.8, 3.0, 1.6, 1.3, "#FDF2F8", "#EC4899"),
        ("8. Validate:\nval_loader loop\nROC-AUC • Acc\nSave best AUC\n→ .pth checkpoint", 13.7, 3.0, 1.6, 1.3, "#FFF7ED", "#F97316"),
        ("9. Plot:\nLoss Trajectory\nTrain vs Val\nWeighted BCE\nplates at ~4.17", 15.3, 3.0, 1.4, 1.3, "#EFF6FF", "#3B82F6"),
    ]
    for txt, x, y, w, h, fc, ec in steps:
        rounded_box(ax, (x,y), w, h, txt, face=fc, edge=ec, fontsize=5)
        # arrow to next
        if x < 15.3:
            arrow(ax, (x+w, y+h/2), (x+w+0.15, y+h/2), color=COLORS["slate"], width=1.2)

    # Loss formula banner
    banner = FancyBboxPatch((0.4, 1.7), 16.2, 0.8, boxstyle="round,pad=0.02,rounding_size=0.04", facecolor="white", edgecolor=COLORS["danger"], linewidth=1.4)
    ax.add_patch(banner)
    ax.text(8.5, 2.25, "WEIGHTED BCE LOSS:  L = − [ w⁺·y·log(ŷ) + (1−y)·log(1−ŷ) ]   where  w⁺ = N_MSS/N_MSI = 263/63 ≈ 4.17", ha="center", fontsize=7, weight="bold", color=COLORS["danger"], style="italic")
    ax.text(8.5, 1.85, "Clamped ŷ ∈ [1e-7, 1−1e-7]  •  Penalizes missed MSI 4.17× harder  •  Equilibrium loss ≈ 4.17 (not 0) — mathematically expected plateau due to scaled positive class", ha="center", fontsize=5.5, color=COLORS["slate"])

    # threshold banner
    banner2 = FancyBboxPatch((0.4, 0.7), 16.2, 0.7, boxstyle="round,pad=0.015,rounding_size=0.03", facecolor="#1E293B", edgecolor="#1E293B")
    ax.add_patch(banner2)
    ax.text(8.5, 1.05, "DECISION THRESHOLDS:   Late Fusion τ=0.70  •  Cross-Attention τ=0.83 (conservative, high MSI precision)   •   MSI if prob ≥ τ else MSS", ha="center", fontsize=6, weight="bold", color="white")
    ax.text(8.5, 0.75, "Threshold sweep 0.30→0.90 — chosen for max MSI Precision while preserving Recall within clinical limits", ha="center", fontsize=5, color="#94A3B8")

    # ── BOTTOM: Evaluation Metrics ──
    ax = axes[1]
    ax.set_xlim(0,17); ax.set_ylim(0,5.2); ax.axis("off"); ax.set_facecolor("#F8FAFC")
    ax.text(0.3, 4.9, "EVALUATION — Isolated Validation (66 patients) • No Leakage • Confusion Matrix + ROC + Reports", fontsize=7, weight="bold", color="white", bbox=dict(boxstyle="round,pad=0.3", facecolor="#0EA5E9"))

    # Metrics comparison table as boxes
    # Header
    hdr = ["Metric", "Late Fusion (τ=0.70)", "Cross-Attention (τ=0.83) ★", "Δ Improvement"]
    xs_tbl = [0.5, 4.5, 8.5, 13.0]
    ws_tbl = [3.6, 3.8, 3.8, 3.5]
    for x, w, h in zip(xs_tbl, ws_tbl, hdr):
        b = FancyBboxPatch((x, 4.05), w, 0.5, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor=COLORS["primary"], edgecolor=COLORS["primary"])
        ax.add_patch(b)
        ax.text(x+w/2, 4.3, h, ha="center", fontsize=6, weight="bold", color="white")

    rows = [
        ("Accuracy", "81.82%  (54/66)", "89.39%  (59/66)", "+7.57 pp", "#10B981"),
        ("ROC-AUC", "0.8950", "0.9150", "+0.020", "#0EA5E9"),
        ("MSS Precision / Recall", "0.83 / 0.96", "0.92 / 0.94", "+0.09 / -0.02", "#3B82F6"),
        ("MSI Precision / Recall", "0.78 / 0.44", "0.80 / 0.75", "+0.02 / +0.31 ★", "#10B981"),
        ("MSI F1-Score", "0.50", "0.77", "+0.27 (+54%)", "#10B981"),
        ("Macro F1", "0.69", "0.85", "+0.16", "#8B5CF6"),
    ]
    y = 3.55
    for metric, late, cross, delta, dcol in rows:
        bg = "#F8FAFC" if rows.index((metric,late,cross,delta,dcol))%2==0 else "white"
        for x, w, txt, col, bold in [(xs_tbl[0], ws_tbl[0], metric, COLORS["primary"], True), (xs_tbl[1], ws_tbl[1], late, "#9A3412", False), (xs_tbl[2], ws_tbl[2], cross, "#065F46", True), (xs_tbl[3], ws_tbl[3], delta, dcol, True)]:
            b = FancyBboxPatch((x, y), w, 0.38, boxstyle="round,pad=0.005,rounding_size=0.01", facecolor=bg, edgecolor=COLORS["border"], linewidth=0.7)
            ax.add_patch(b)
            ax.text(x+w/2, y+0.19, txt, ha="center", fontsize=5.5, weight="bold" if bold else "normal", color=col)
        y -= 0.42

    # Confusion visuals
    ax.text(0.5, 0.95, "Confusion Insight:", fontsize=6, weight="bold", color=COLORS["primary"])
    cm_text = "Late Fusion: misses 9/16 MSI (56% FN) due to weak recall  •  Cross-Attn: catches 12/16 MSI (only 4 FN) — text-guided search doubles MSI detection while maintaining 0.80 precision"
    ax.text(0.5, 0.65, cm_text, fontsize=5.5, color=COLORS["slate"], style="italic", bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFBEB", edgecolor="#FDE68A"))
    ax.text(0.5, 0.25, "Validation Contract: random_split seed=42 • actual_validation_keys.json locks 66 patients • eval filters strictly to those keys — guarantees zero overlap with training", fontsize=5, color=COLORS["slate"], style="italic")

    plt.tight_layout()
    save(fig, "07_training_evaluation_pipeline.png")

# ═══════════════════════════════════════════════
# 8. EXPLAINABLE AI STACK
# ═══════════════════════════════════════════════
def diagram_xai():
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.set_xlim(0,17); ax.set_ylim(0,10); ax.axis("off")
    fig.patch.set_facecolor("white"); ax.set_facecolor("#F8FAFC")
    ax.text(8.5, 9.7, "EXPLAINABLE AI (XAI) STACK — 3-Tier Interpretability System", ha="center", fontsize=13, weight="bold", color=COLORS["primary"])
    ax.text(8.5, 9.3, "Macroscopic (patch ranking) → Microscopic (cellular heatmap) → Linguistic (LLM scribe) — no black box, clinician-auditable", ha="center", fontsize=7, color=COLORS["slate"], style="italic")
    ax.plot([0.3,16.7],[9.05,9.05], color=COLORS["border"], lw=1)

    # Tier 1
    outer1 = FancyBboxPatch((0.4, 5.9), 5.3, 2.9, boxstyle="round,pad=0.02,rounding_size=0.05", facecolor="white", edgecolor="#3B82F6", linewidth=1.5)
    ax.add_patch(outer1)
    ax.text(3.05, 8.55, "TIER 1 — MACROSCOPIC XAI", ha="center", fontsize=7.5, weight="bold", color="white", bbox=dict(boxstyle="round,pad=0.3", facecolor="#3B82F6"))
    ax.text(3.05, 8.2, "Q: Where did the model look?", ha="center", fontsize=6, color="#3B82F6", weight="bold", style="italic")
    steps1 = [
        "① Intercept attn_weights\n[B×1×N] before aggregation",
        "② Average across 8 heads\n→ scalar per patch",
        "③ argsort descending\n→ rank 1…N",
        "④ Extract Top-5 indices\nHighest influence on MSI/MSS",
        "⑤ Gallery: 5-col grid\nRed border • Score label\nFilename truncated",
    ]
    y = 7.85
    for s in steps1:
        rounded_box(ax, (0.7, y), 4.7, 0.32, s, face="#EFF6FF", edge="#93C5FD", fontsize=5)
        y -= 0.38
    ax.text(3.05, 5.95, "Output: Ranked patch list + filenames → Streamlit gallery", ha="center", fontsize=5, color=COLORS["slate"], weight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor=COLORS["light"], edgecolor=COLORS["border"]))

    # Arrow
    arrow(ax, (5.7, 7.3), (6.3, 7.3), color=COLORS["slate"], width=1.5)

    # Tier 2
    outer2 = FancyBboxPatch((6.3, 5.9), 5.3, 2.9, boxstyle="round,pad=0.02,rounding_size=0.05", facecolor="white", edgecolor="#EC4899", linewidth=1.5)
    ax.add_patch(outer2)
    ax.text(8.95, 8.55, "TIER 2 — MICROSCOPIC XAI", ha="center", fontsize=7.5, weight="bold", color="white", bbox=dict(boxstyle="round,pad=0.3", facecolor="#EC4899"))
    ax.text(8.95, 8.2, "Q: What is inside the patch that triggered the signal?", ha="center", fontsize=6, color="#EC4899", weight="bold", style="italic")
    steps2 = [
        "① Select top-ranked patch\nRe-route to Phikon-v2 ViT",
        "② Attention Rollout: stack\nlast 6 layers (of 24) → avg",
        "③ Head-average → CLS attn\n→ grid_size = √196 = 14",
        "④ GaussianBlur (5×5)→ smooth\nThreshold 70th & 80th %ile",
        "⑤ Jet colormap (jet, α=0.45)\n+ Yellow ROI contours (AA)",
    ]
    y = 7.85
    for s in steps2:
        rounded_box(ax, (6.6, y), 4.7, 0.32, s, face="#FDF2F8", edge="#F9A8D4", fontsize=5)
        y -= 0.38
    ax.text(8.95, 5.95, "Output: 3-panel figure — Original | ROI Contours | Jet Heatmap Overlay", ha="center", fontsize=5, color=COLORS["slate"], weight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor=COLORS["light"], edgecolor=COLORS["border"]))

    arrow(ax, (11.6, 7.3), (12.2, 7.3), color=COLORS["slate"], width=1.5)

    # Tier 3
    outer3 = FancyBboxPatch((12.2, 5.9), 4.4, 2.9, boxstyle="round,pad=0.02,rounding_size=0.05", facecolor="white", edgecolor="#8B5CF6", linewidth=1.5)
    ax.add_patch(outer3)
    ax.text(14.4, 8.55, "TIER 3 — LINGUISTIC XAI", ha="center", fontsize=7.5, weight="bold", color="white", bbox=dict(boxstyle="round,pad=0.3", facecolor="#8B5CF6"))
    ax.text(14.4, 8.2, "Q: What does it mean clinically?", ha="center", fontsize=6, color="#8B5CF6", weight="bold", style="italic")
    steps3 = [
        "① Assemble context pkg\n{prob, top-5, report_text}",
        "② Deterministic prompt\n'You are pathology scribe…'",
        "③ POST localhost:11434\nmodel: qwen2.5-coder:7b",
        "④ Stream=False → JSON\nExtract 'response'",
        "⑤ 3 paragraphs: Summary\nFindings • Recommendation",
    ]
    y = 7.85
    for s in steps3:
        rounded_box(ax, (12.5, y), 3.8, 0.32, s, face="#EDE9FE", edge="#C4B5FD", fontsize=5)
        y -= 0.38
    ax.text(14.4, 5.95, "Output: Clinician-readable report", ha="center", fontsize=5, color=COLORS["slate"], weight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor=COLORS["light"], edgecolor=COLORS["border"]))

    # Bottom: Verification + Flow
    # Verification banner
    outer_v = FancyBboxPatch((0.4, 2.2), 16.2, 3.2, boxstyle="round,pad=0.02,rounding_size=0.05", facecolor="white", edgecolor=COLORS["primary"], linewidth=1.5)
    ax.add_patch(outer_v)
    ax.text(8.5, 5.15, "VERIFICATION: PERTURBATION TEST — Proving Cross-Attention is Not Late Fusion", ha="center", fontsize=7, weight="bold", color="white", bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["primary"]))
    # Two columns
    rounded_box(ax, (0.7, 3.6), 7.6, 1.2, "LATE FUSION BASELINE\nReal text vs. Fake noise text → Attention profile UNCHANGED\n→ Attention weights static, ignoring text semantics\n→ Proves isolated branches (no text-guided search)", face="#FEE2E2", edge="#FCA5A5", fontsize=5.5)
    rounded_box(ax, (9.0, 3.6), 7.3, 1.2, "CROSS-ATTENTION PRIMARY ★\nReal text vs. Fake noise → Top-5 ranking PHYSICALLY SHIFTS\n→ Jensen-Shannon Divergence > 0, profile diverges\n→ Proves dynamic text-guided visual search ✓", face="#D1FAE5", edge="#6EE7B7", fontsize=5.5)
    ax.text(8.5, 3.35, "Procedure: Record baseline attn → synthesize random noise text (same shape) → second forward → compare Top-5 + divergence → plot profile shift (verify_cross_attention.py)", ha="center", fontsize=5, color=COLORS["slate"], style="italic",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=COLORS["light"], edgecolor=COLORS["border"]))
    # Color legend bottom
    ax.text(2.5, 2.55, "JET Heatmap Legend:  [Red] = high attention (nuclei clusters, glandular borders, immune infiltrate)    [Blue] = low (background / stroma)    [Yellow] contours = Top 20% ROI", ha="center", fontsize=5, color=COLORS["slate"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLORS["border"]))
    ax.text(8.5, 2.0, "FILES:  src/generate_pixel_xai.py  •  src/generate_xai.py  •  src/eval/verify_cross_attention.py  •  src/ui/app.py (integrated)", ha="center", fontsize=5, color=COLORS["slate"], style="italic")

    save(fig, "08_xai_stack.png")

# ═══════════════════════════════════════════════
# 9. USE CASE + ER DIAGRAM
# ═══════════════════════════════════════════════
def diagram_usecase_er():
    fig, axes = plt.subplots(1,2, figsize=(18, 10))
    fig.patch.set_facecolor("white")
    fig.suptitle("USE CASE DIAGRAM & ENTITY-RELATIONSHIP DIAGRAM", fontsize=12, weight="bold", color=COLORS["primary"], y=0.98)

    # ── LEFT: Use Case ──
    ax = axes[0]
    ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off"); ax.set_facecolor("#F8FAFC")
    ax.set_title("Use Case Diagram — Actors & System Interactions", fontsize=9, weight="bold", color=COLORS["primary"], pad=10)

    # System boundary
    sys_box = FancyBboxPatch((2.2, 1.0), 5.6, 7.8, boxstyle="round,pad=0.02,rounding_size=0.05", facecolor="white", edgecolor=COLORS["primary"], linewidth=1.5)
    ax.add_patch(sys_box)
    ax.text(5.0, 8.5, "MSI Detection System", ha="center", fontsize=7, weight="bold", color=COLORS["primary"], style="italic")

    # Actors
    actors = [
        ("Pathologist", 0.6, 7.5, "#EFF6FF", "#3B82F6"),
        ("ML Engineer", 0.6, 5.8, "#F0FDF4", "#10B981"),
        ("Clinician", 8.5, 7.5, "#FEF3C7", "#F59E0B"),
        ("System\n(Phikon-v2,\nClinicalBERT,\nQwen LLM)", 8.5, 3.5, "#EDE9FE", "#8B5CF6"),
    ]
    for txt, x, y, fc, ec in actors:
        # stick figure approx via ellipse + line (simple box)
        rounded_box(ax, (x-0.7, y-0.3), 1.4, 0.7, txt, face=fc, edge=ec, fontsize=5.5, bold=True)

    # Use cases inside
    ucs = [
        ("UC1: Upload\nPatient Data", 3.2, 7.5, 1.6, 0.6),
        ("UC2: Run\nDiagnosis", 5.6, 7.5, 1.6, 0.6),
        ("UC3: View Top-5\nXAI Gallery", 3.2, 6.6, 1.6, 0.6),
        ("UC4: Generate\nCellular Heatmap", 5.6, 6.6, 1.6, 0.6),
        ("UC5: Read LLM\nClinical Report", 3.2, 5.7, 1.6, 0.6),
        ("UC6: Train /\nRetrain Model", 3.2, 4.6, 1.6, 0.6),
        ("UC7: Evaluate\n(ROC/CM)", 5.6, 4.6, 1.6, 0.6),
        ("UC8: Verify\nPerturbation Test", 3.2, 3.7, 1.6, 0.6),
        ("UC9: Manage\nDataset Registry", 5.6, 3.7, 1.6, 0.6),
        ("UC10: Monitor\nLoss Trajectory", 4.4, 2.7, 1.6, 0.6),
    ]
    for txt, x, y, w, h in ucs:
        e = Ellipse((x, y), w, h, facecolor="white", edgecolor=COLORS["primary"], linewidth=1.1)
        ax.add_patch(e)
        ax.text(x, y, txt, ha="center", va="center", fontsize=4.5, color=COLORS["primary"], weight="bold", linespacing=1.2)

    # Connections (simplified straight lines)
    # Pathologist to UC1,2,3,4,5
    for y_uc in [7.5, 6.6, 5.7]:
        ax.plot([1.3, 3.0], [7.5, y_uc], color=COLORS["slate"], lw=0.7)
    ax.plot([1.3, 5.6], [7.3, 6.6], color=COLORS["slate"], lw=0.7)
    # ML Engineer to UC6,7,8...
    for yy, xx in [(4.6,3.2),(4.6,5.6),(3.7,3.2),(2.7,4.4)]:
        ax.plot([1.3, xx], [5.8, yy], color=COLORS["slate"], lw=0.7, ls="--")
    # Clinician to UC5,2,3
    ax.plot([8.2, 4.0], [7.5, 5.7], color=COLORS["slate"], lw=0.7)
    ax.plot([8.2, 5.6], [7.5, 7.5], color=COLORS["slate"], lw=0.7)
    # System to core UCs
    for xx, yy in [(5.6,7.5),(3.2,6.6),(3.2,5.7)]:
        ax.plot([8.0, xx+0.5], [3.5, yy], color="#8B5CF6", lw=0.7, ls=":")

    # Include/extend
    ax.text(5.0, 1.3, "<<include>>: UC2 includes UC3 & UC5  •  <<extend>>: UC4 extends UC3", ha="center", fontsize=5, color=COLORS["slate"], style="italic",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=COLORS["light"], edgecolor=COLORS["border"]))

    # ── RIGHT: ER Diagram ──
    ax = axes[1]
    ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off"); ax.set_facecolor("#F8FAFC")
    ax.set_title("Entity-Relationship Diagram — Data Model", fontsize=9, weight="bold", color=COLORS["primary"], pad=10)

    def entity_box(x, y, w, h, title, attrs, pk_idx=0, color="#3B82F6"):
        # header
        hdr = FancyBboxPatch((x, y+h-0.4), w, 0.4, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor=color, edgecolor=color)
        ax.add_patch(hdr)
        ax.text(x+w/2, y+h-0.2, title, ha="center", fontsize=6, weight="bold", color="white")
        body = FancyBboxPatch((x, y), w, h-0.4, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor="white", edgecolor=color, linewidth=1.3)
        ax.add_patch(body)
        for i, (attr, pk) in enumerate(attrs):
            yy = y+h-0.65 - i*0.28
            marker = "PK:" if pk else "  "
            ax.text(x+0.15, yy, f"{marker} {attr}", fontsize=5, color=COLORS["primary"] if pk else COLORS["slate"], weight="bold" if pk else "normal")

    entity_box(0.4, 6.2, 2.8, 2.6, "Patient", [
        ("patient_id (TCGA-*)", True),
        ("label: 0=MSS, 1=MSI", False),
        ("class_name: MSI/MSS", False),
        ("report_text: long str", False),
        ("image_paths: List[str]", False),
        ("patch_count: int", False),
    ], color="#3B82F6")

    entity_box(3.8, 6.2, 2.8, 2.6, "TensorStore", [
        ("patient_id (FK)", True),
        ("vision_path: *_vision.pt", False),
        ("text_path: *_text.pt", False),
        ("vision_dim: 1024", False),
        ("text_dim: 768", False),
        ("tensor_type: frozen", False),
    ], color="#10B981")

    entity_box(7.0, 6.2, 2.6, 2.6, "ModelWeight", [
        ("model_name (PK)", True),
        ("arch: Late/CrossAttn", False),
        ("file: *.pth", False),
        ("threshold: 0.70/0.83", False),
        ("best_auc: float", False),
        ("epochs: 30", False),
    ], color="#F59E0B")

    entity_box(0.4, 2.8, 2.8, 2.4, "ValidationSplit", [
        ("split_id (PK)", True),
        ("patient_id (FK)", False),
        ("split: train/val", False),
        ("source: actual_keys.json", False),
        ("size: 66 val", False),
    ], color="#8B5CF6")

    entity_box(3.8, 2.8, 2.8, 2.4, "InferenceResult", [
        ("patient_id (FK)", True),
        ("prob: float [0,1]", False),
        ("pred_label: MSI/MSS", False),
        ("attn_weights: [N]", False),
        ("top5_indices: List", False),
    ], color="#EC4899")

    entity_box(7.0, 2.8, 2.6, 2.4, "XAIOutput", [
        ("patient_id (FK)", True),
        ("top5_patches: List[JPG]", False),
        ("heatmap_path: PNG", False),
        ("scribe_report: text", False),
        ("XAI_tier: macro/micro", False),
    ], color="#0EA5E9")

    # Relationships
    # Patient 1—2 TensorStore (1 to 2? actually 1-1 per patient has 2 tensors)
    arrow(ax, (3.2, 7.4), (3.8, 7.4), color=COLORS["slate"], width=1.2)
    ax.text(3.5, 7.55, "1 : 2", ha="center", fontsize=5, weight="bold", color=COLORS["slate"], bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=COLORS["border"]))
    ax.text(3.5, 7.15, "has", ha="center", fontsize=4.5, color=COLORS["slate"], style="italic")
    # Patient to ValidationSplit 1:N
    arrow(ax, (1.8, 6.2), (1.8, 5.2), color=COLORS["slate"], width=1.1)
    ax.text(2.15, 5.7, "1 : N", fontsize=5, weight="bold", color=COLORS["slate"], bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=COLORS["border"]))
    # TensorStore to Inference
    arrow(ax, (5.2, 6.2), (5.2, 5.2), color=COLORS["slate"], width=1.1)
    ax.text(5.55, 5.7, "N : 1", ha="center", fontsize=5, weight="bold", color=COLORS["slate"], bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=COLORS["border"]))
    # ModelWeight to Inference
    arrow(ax, (7.0, 5.2), (6.6, 5.2), color=COLORS["slate"], width=1.1)
    ax.text(6.8, 5.4, "1 : N", ha="center", fontsize=5, weight="bold", color=COLORS["slate"], bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=COLORS["border"]))
    # Inference to XAI
    arrow(ax, (6.6, 3.7), (7.0, 3.7), color=COLORS["slate"], width=1.1)
    ax.text(6.8, 3.85, "1 : 1", ha="center", fontsize=5, weight="bold", color=COLORS["slate"], bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=COLORS["border"]))

    ax.text(5, 0.6, "Cardinalities: Patient 1:2 Tensors  •  Patient 1:N Validation membership  •  (Tensor+Model) N:1 Inference  •  Inference 1:1 XAI  •  PK = patient_id string (TCGA-*)", ha="center", fontsize=5, color=COLORS["slate"], style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLORS["border"]))

    plt.tight_layout()
    save(fig, "09_usecase_er_diagram.png")

# ═══════════════════════════════════════════════
# 10. STATE & FLOWCHART
# ═══════════════════════════════════════════════
def diagram_state_flow():
    fig, axes = plt.subplots(1,2, figsize=(18, 10))
    fig.patch.set_facecolor("white")
    fig.suptitle("STATE TRANSITION & OPERATIONAL FLOWCHART", fontsize=12, weight="bold", color=COLORS["primary"], y=0.98)

    # ── LEFT: State Machine (Inference lifecycle) ──
    ax = axes[0]
    ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off"); ax.set_facecolor("#F8FAFC")
    ax.set_title("State Diagram — Clinical Inference Lifecycle", fontsize=9, weight="bold", color=COLORS["primary"], pad=10)

    states = [
        ("IDLE IDLE\nAwaiting\nInput", 5.0, 8.5, 1.6, 0.7, COLORS["slate"], "white"),
        ("LOADING\nLoad tensors\n+ weights", 1.5, 7.0, 1.6, 0.7, "#3B82F6", "white"),
        ("INFERENCING\nCross-Attn\nforward()", 5.0, 7.0, 1.6, 0.7, "#F59E0B", "white"),
        ("XAI READY\nTop-5 ranked\n+ prob", 8.5, 7.0, 1.6, 0.7, "#10B981", "white"),
        ("HEATMAP\nGENERATING\nPhikon-v2 rollout", 1.5, 5.2, 1.6, 0.7, "#EC4899", "white"),
        ("SCRIBE\nGENERATING\nQwen 2.5", 8.5, 5.2, 1.6, 0.7, "#8B5CF6", "white"),
        ("DISPLAY\nAll panels\nrendered", 5.0, 5.2, 1.6, 0.7, COLORS["primary"], "white"),
        ("ERROR\nFileNotFound\n/ Ollama down", 5.0, 3.5, 1.6, 0.7, COLORS["danger"], "white"),
    ]
    for txt, x, y, w, h, bg, fg in states:
        # pill
        b = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.02,rounding_size=0.2", facecolor=bg, edgecolor=bg, linewidth=1.2)
        ax.add_patch(b)
        ax.text(x, y, txt, ha="center", va="center", fontsize=5, weight="bold", color=fg, linespacing=1.2)

    # Transitions
    trans = [
        ((5.0, 8.15), (5.0, 7.35), "Run Pipeline", COLORS["primary"]),
        ((4.2, 8.3), (2.3, 7.35), "Patient found", "#3B82F6"),
        ((6.7, 7.0), (5.8, 7.0), "prob+attn", "#F59E0B"),
        ((7.7, 7.0), (8.5, 5.9), "Generate Heatmap", "#EC4899"),
        ((7.7, 6.8), (8.5, 5.9), "", "#8B5CF6"),  # will overlap, offset
        ((5.8, 5.2), (2.3, 5.2), "← back", COLORS["slate"]),
        ((4.2, 5.2), (5.0, 5.5), "heatmap done →", "#EC4899"),
        ((6.0, 5.2), (5.8, 5.2), "", "#8B5CF6"),
    ]
    # Simplified arrows
    arrow(ax, (5.0, 8.15), (3.1, 7.35), color=COLORS["primary"], width=1.1, label="Click Run", label_bg="white")
    arrow(ax, (2.3, 6.65), (4.2, 6.65), color="#3B82F6", width=1.1)
    ax.text(3.25, 6.8, "tensors OK", ha="center", fontsize=4.5, color="#3B82F6", weight="bold")
    arrow(ax, (6.6, 7.0), (7.7, 7.0), color="#F59E0B", width=1.1)
    arrow(ax, (8.5, 6.65), (8.5, 5.55), color="#10B981", width=1.1)
    # heatmap loop
    arrow(ax, (5.8, 5.55), (2.3, 5.55), color="#EC4899", width=1, ls="--")
    ax.text(4.0, 5.7, "Select patch → rollout", ha="center", fontsize=4.5, color="#EC4899", style="italic")
    arrow(ax, (2.3, 5.2), (4.2, 5.2), color="#EC4899", width=1, ls="--")
    # scribe
    arrow(ax, (7.7, 6.7), (8.5, 5.55), color="#8B5CF6", width=1.1)
    ax.text(8.2, 6.3, "LLM", fontsize=4.5, color="#8B5CF6", weight="bold")
    arrow(ax, (7.7, 5.2), (5.8, 5.2), color="#8B5CF6", width=1, ls="--")
    ax.text(6.7, 5.35, "scribe done", ha="center", fontsize=4.5, color="#8B5CF6", style="italic")
    # error paths (dotted red)
    arrow(ax, (1.5, 6.65), (4.2, 3.85), color=COLORS["danger"], width=0.9, ls="--")
    ax.text(2.5, 4.8, "FileNotFound", ha="center", fontsize=4.5, color=COLORS["danger"], weight="bold")
    arrow(ax, (8.5, 4.85), (5.8, 3.85), color=COLORS["danger"], width=0.9, ls="--")
    ax.text(7.8, 4.5, "Ollama error", ha="center", fontsize=4.5, color=COLORS["danger"], weight="bold")
    arrow(ax, (5.0, 3.15), (5.0, 4.85), color=COLORS["danger"], width=0.9, ls="--")
    ax.text(5.6, 4.0, "retry", ha="center", fontsize=4.5, color=COLORS["danger"], style="italic")

    ax.text(5, 0.5, "States: 8  •  Transitions: 12  •  Terminal: DISPLAY (success) or ERROR (recoverable)  •  Session-persistent via st.session_state", ha="center", fontsize=5, color=COLORS["slate"], style="italic")

    # ── RIGHT: Flowchart ──
    ax = axes[1]
    ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off"); ax.set_facecolor("#F8FAFC")
    ax.set_title("Flowchart — End-to-End Operational Pipeline", fontsize=9, weight="bold", color=COLORS["primary"], pad=10)

    def flow_box(x, y, w, h, txt, shape="rect", color=COLORS["primary"], face="white", fontsize=5):
        if shape == "terminator":
            # pill
            b = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.02,rounding_size=0.3", facecolor=color, edgecolor=color)
            ax.add_patch(b)
            ax.text(x, y, txt, ha="center", va="center", fontsize=fontsize, weight="bold", color="white")
        elif shape == "diamond":
            # diamond via polygon
            import matplotlib.patches as patches
            pts = [(x, y+h/2), (x+w/2, y), (x, y-h/2), (x-w/2, y)]
            poly = patches.Polygon(pts, closed=True, facecolor="#FEF3C7", edgecolor="#F59E0B", linewidth=1.3)
            ax.add_patch(poly)
            ax.text(x, y, txt, ha="center", va="center", fontsize=fontsize, weight="bold", color="#92400E", linespacing=1.1)
        elif shape == "io":
            # parallelogram
            import matplotlib.patches as patches
            dx = 0.25
            pts = [(x-w/2+dx, y+h/2), (x+w/2+dx, y+h/2), (x+w/2-dx, y-h/2), (x-w/2-dx, y-h/2)]
            poly = patches.Polygon(pts, closed=True, facecolor=face, edgecolor=color, linewidth=1.2)
            ax.add_patch(poly)
            ax.text(x, y, txt, ha="center", va="center", fontsize=fontsize, weight="bold", color=color)
        else:
            rounded_box(ax, (x-w/2, y-h/2), w, h, txt, face=face, edge=color, fontsize=fontsize, bold=False)

    # Flow steps vertical
    flow_box(5, 9.1, 2.2, 0.5, "START", shape="terminator", color=COLORS["primary"])
    arrow(ax, (5, 8.85), (5, 8.5), color=COLORS["primary"])
    flow_box(5, 8.15, 3.6, 0.5, "Load patient_registry.json + tensors", shape="io", color="#3B82F6", face="#EFF6FF")
    arrow(ax, (5, 7.9), (5, 7.55), color=COLORS["primary"])
    flow_box(5, 7.2, 2.0, 0.5, "Patient ID\nin registry?", shape="diamond", fontsize=4.8)
    # No branch
    arrow(ax, (4.0, 7.2), (1.5, 7.2), color=COLORS["danger"])
    ax.text(2.7, 7.35, "No", ha="center", fontsize=5, weight="bold", color=COLORS["danger"])
    flow_box(1.5, 6.5, 2.0, 0.55, "Show ERROR\n'Patient not found'", shape="rect", color=COLORS["danger"], face="#FEE2E2")
    arrow(ax, (1.5, 6.22), (1.5, 5.6), color=COLORS["danger"], ls="--")
    ax.text(1.5, 5.0, "⟲ Await new input", ha="center", fontsize=5, color=COLORS["danger"], style="italic")
    # Yes branch
    arrow(ax, (5, 6.95), (5, 6.6), color=COLORS["success"])
    ax.text(5.25, 6.75, "Yes", ha="center", fontsize=5, weight="bold", color=COLORS["success"])
    flow_box(5, 6.25, 3.6, 0.5, "Load vision [N×1024] + text [768]\n→ device (cuda/mps/cpu)", shape="rect", color=COLORS["primary"])
    arrow(ax, (5, 6.0), (5, 5.65), color=COLORS["primary"])
    flow_box(5, 5.3, 3.8, 0.5, "Forward: SummaryToPatchCrossAttention\nprob, attn_weights = model(V,T)", shape="rect", color="#F59E0B", face="#FFFBEB")
    arrow(ax, (5, 5.05), (5, 4.7), color=COLORS["primary"])
    flow_box(5, 4.35, 2.2, 0.5, "prob ≥ 0.83 ?", shape="diamond", fontsize=5)
    # branches
    arrow(ax, (3.9, 4.35), (1.8, 4.35), color=COLORS["success"])
    ax.text(2.8, 4.5, "Yes → MSI", ha="center", fontsize=5, weight="bold", color=COLORS["success"])
    arrow(ax, (6.1, 4.35), (8.2, 4.35), color=COLORS["slate"])
    ax.text(7.1, 4.5, "No → MSS", ha="center", fontsize=5, weight="bold", color=COLORS["slate"])
    # Merge
    flow_box(5, 3.6, 3.8, 0.5, "Compute confidence • argsort attn → Top-5\nGenerate gallery + LLM scribe", shape="rect", color="#8B5CF6", face="#EDE9FE")
    # from both branches to merge
    ax.plot([1.8, 1.8, 5], [4.35, 3.6, 3.6], color=COLORS["slate"], lw=1, linestyle="--")
    ax.plot([8.2, 8.2, 5], [4.35, 3.6, 3.6], color=COLORS["slate"], lw=1, linestyle="--")
    arrow(ax, (5, 3.35), (5, 3.0), color=COLORS["primary"])
    flow_box(5, 2.65, 3.8, 0.5, "Render: Streamlit dashboard\nMeter + Gallery + Heatmap + Report", shape="io", color="#8B5CF6", face="#EDE9FE")
    arrow(ax, (5, 2.4), (5, 2.05), color=COLORS["primary"])
    # Manual heatmap decision
    flow_box(5, 1.7, 2.2, 0.5, "User requests\nheatmap?", shape="diamond", fontsize=4.8)
    arrow(ax, (3.9, 1.7), (1.5, 1.7), color="#EC4899")
    ax.text(2.7, 1.85, "Yes", ha="center", fontsize=5, weight="bold", color="#EC4899")
    flow_box(1.5, 1.0, 2.0, 0.5, "Phikon-v2 rollout\nJet + Contours", shape="rect", color="#EC4899", face="#FDF2F8")
    ax.plot([1.5, 1.5, 5], [0.75, 0.55, 0.55], color=COLORS["slate"], lw=1, ls="--")
    arrow(ax, (5, 1.45), (5, 1.0), color=COLORS["slate"])
    ax.text(5.35, 1.3, "No", ha="center", fontsize=5, color=COLORS["slate"])
    flow_box(5, 0.45, 2.0, 0.35, "END", shape="terminator", color=COLORS["primary"], fontsize=5)
    arrow(ax, (5, 0.72), (5, 0.62), color=COLORS["primary"])

    plt.tight_layout()
    save(fig, "10_state_flowchart.png")

if __name__ == "__main__":
    diagram_system_architecture()
    diagram_data_flow()
    diagram_model_architecture()
    diagram_sequence()
    diagram_deployment()
    diagram_component()
    diagram_training_evaluation()
    diagram_xai()
    diagram_usecase_er()
    diagram_state_flow()
    print("\n🎉 All 10 diagrams generated successfully!")
