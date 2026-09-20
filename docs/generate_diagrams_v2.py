#!/usr/bin/env python3
"""
Professional Diagram Generator v2 — Precision Orchestrated (SVG + PNG)
Replaces matplotlib FancyBbox raster with SVG vector (svgwrite + cairosvg)
Style: Corporate, Gemini/Claude-like — Inter, 8pt grid, rx12, subtle shadow, consistent palette
"""
import pathlib, os
import svgwrite
import cairosvg

OUT = pathlib.Path(__file__).parent / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)

# Tokens
COLORS = {
    "primary": "#0B2C4A",
    "accent": "#0EA5E9",
    "accent2": "#F59E0B",
    "success": "#10B981",
    "danger": "#EF4444",
    "purple": "#8B5CF6",
    "teal": "#14B8A6",
    "slate": "#64748B",
    "slate_light": "#F1F5F9",
    "border": "#E2E8F0",
    "bg": "#F8FAFC",
    "card": "#FFFFFF",
}
FONT = "Inter, Helvetica, Arial, sans-serif"

def grid(v): return round(v/8)*8

def add_shadow_filter(dwg):
    # No-op shadow for compatibility — subtle shadow via stroke instead
    return None
    f = dwg.defs.add(dwg.filter(id="shadow", x="-20%", y="-20%", width="140%", height="140%"))
    f.feDropShadow(dx=0, dy=2, stdDeviation=3, flood_color="#0B2C4A", flood_opacity=0.08)
    return f

def rounded_rect(dwg, pos, size, rx=12, fill="white", stroke="#0B2C4A", stroke_width=1.4, shadow=False, opacity=1):
    g = dwg.g()
    rect = dwg.rect(insert=pos, size=size, rx=rx, ry=rx, fill=fill, stroke=stroke, stroke_width=stroke_width, opacity=opacity)
    if False and shadow:
        rect.update({"filter": "url(#shadow)"})
    g.add(rect)
    dwg.add(g)
    return g

def text(dwg, pos, txt, size=9, weight=400, color="#0B2C4A", anchor="middle", family=FONT, line_spacing=1.35):
    # Multi-line support via tspan
    lines = txt.split("\n")
    g = dwg.g(font_family=family, font_size=size, font_weight=weight, fill=color, text_anchor=anchor, dominant_baseline="middle")
    x, y = pos
    line_h = size * line_spacing
    start_y = y - (len(lines)-1)*line_h/2
    for i, line in enumerate(lines):
        g.add(dwg.text(line, insert=(x, start_y + i*line_h)))
    dwg.add(g)
    return g

def arrow(dwg, start, end, color="#0B2C4A", width=1.5, dashed=False, marker=True):
    # Define marker once
    if marker and "arrowhead" not in dwg.defs.__dict__:
        try:
            marker = dwg.marker(id="arrowhead", viewBox="0 0 10 10", refX=8, refY=5, markerWidth=6, markerHeight=6, orient="auto")
            marker.add(dwg.path(d="M 0 0 L 10 5 L 0 10 z", fill=color))
            dwg.defs.add(marker)
        except:
            pass
    line = dwg.line(start=start, end=end, stroke=color, stroke_width=width, stroke_linecap="round")
    if dashed:
        line.update({"stroke_dasharray": "6,4"})
    if marker:
        line.update({"marker_end": "url(#arrowhead)"})
    dwg.add(line)
    return line

def label_pill(dwg, pos, txt, bg="white", border="#0B2C4A", color="#0B2C4A", size=6.5):
    w = len(txt)*4.2 + 16
    h = 14
    g = dwg.g()
    g.add(dwg.rect(insert=(pos[0]-w/2, pos[1]-h/2), size=(w,h), rx=7, fill=bg, stroke=border, stroke_width=1))
    g.add(dwg.text(txt, insert=(pos[0], pos[1]+0.5), text_anchor="middle", dominant_baseline="middle", font_family=FONT, font_size=size, font_weight=600, fill=color))
    dwg.add(g)
    return g

def save(dwg, name):
    svg_path = OUT / f"{name}.svg"
    png_path = OUT / f"{name}.png"
    dwg.saveas(str(svg_path))
    # PNG via cairosvg
    try:
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), scale=2, background_color="white")
    except Exception as e:
        print(f"cairosvg fail {name}: {e}")
        # fallback: keep SVG only
    print(f"✅ {svg_path.name} + {png_path.name if png_path.exists() else 'SVG only'}")

# ──────────────────────────────────────────────────
# 01 System Architecture — 4 columns, 8pt grid, rx12, shadow
# ──────────────────────────────────────────────────
def diagram_01():
    W, H = 1700, 1000
    dwg = svgwrite.Drawing(size=(W, H), viewBox=f"0 0 {W} {H}")
    dwg.add(dwg.rect(insert=(0,0), size=(W,H), fill="white"))
    add_shadow_filter(dwg)
    # Title
    text(dwg, (W/2, 48), "SYSTEM ARCHITECTURE — 4-PHASE MULTIMODAL MSI DETECTION PIPELINE", 16, 700, COLORS["primary"])
    text(dwg, (W/2, 76), "TCGA-COAD  •  Phikon-v2 (ViT-L/16)  •  ClinicalBERT  •  SummaryToPatchCrossAttention  •  Streamlit + Qwen 2.5", 9, 400, COLORS["slate"])
    dwg.add(dwg.line(start=(32, 96), end=(W-32, 96), stroke=COLORS["border"], stroke_width=1))
    # Columns
    cols = [
        ("PHASE 1 — DATA CURATION", "#3B82F6", "#EFF6FF", [
            "Kaggle Dataset\ntcga_coad_msi_mss_jpg\n326 Patients • 192k Patches\n224×224 @10×, Reinhard norm",
            "Patient Registry\npatient_registry.json\nlabel • report_text\nimage_paths • patch_count",
            "Preprocessing\nReinhard Stain Norm\nOtsu Foreground Mask\n10× Magnification",
        ]),
        ("PHASE 2 — FEATURE ENGINEERING\n(Frozen Foundation Encoders)", "#10B981", "#F0FDF4", [
            "Vision Encoder\nPhikon-v2 (DINOv2 ViT-L/16)\n307M params • 460M tiles\nV ∈ ℝ^{N×1024}  [N≤1000]",
            "Text Encoder\nClinicalBERT (110M)\n2M MIMIC-III notes\nT ∈ ℝ^{1×768}  →  [L×768]",
            "Tensor Store\nclean_tensors/*.pt\n*_vision.pt  *_text.pt\n653 files • 326×2",
        ]),
        ("PHASE 3 — FUSION & LEARNING", "#F59E0B", "#FEF3C7", [
            "Late Fusion (Baseline)\nAB-MIL Attention → [1×1024]\nText 768→512 • Concat 1536\nMLP 1536→256→1  τ=0.70",
            "★ Cross-Attention (Primary)\nSummaryToPatchCrossAttention\n768→1024 Proj • Q=text\n8-head MHA + LayerNorm",
            "Training\nWeighted BCE pos=4.17\nAdamW  lr1e-4  30 epochs\nThreshold τ = 0.83",
        ]),
        ("PHASE 4 — INFERENCE & XAI\n(Clinical Deployment)", "#EC4899", "#FDF2F8", [
            "Macroscopic XAI\nCross-Attention Weights\nTop-5 Patch Ranking\nRed-border gallery",
            "Microscopic XAI\nPhikon-v2 Rollout Product\nHematoxylin-aware mask\nJet dark overlay  α0.68",
            "LLM Scribe + UI\nQwen 2.5 via Ollama\n3-paragraph report\nStreamlit Dashboard",
        ]),
    ]
    col_w = 390
    col_gap = 24
    start_x = 32
    for idx, (title, accent, bg, boxes) in enumerate(cols):
        x = start_x + idx*(col_w+col_gap)
        # Column background
        rounded_rect(dwg, (x, 120), (col_w, 620), rx=16, fill=bg, stroke=accent, stroke_width=1.2, shadow=False, opacity=0.7)
        # Header
        rounded_rect(dwg, (x, 120), (col_w, 52), rx=12, fill=accent, stroke=accent, shadow=True)
        text(dwg, (x+col_w/2, 146), title, 8, 700, "white")
        # Boxes
        for i, txt in enumerate(boxes):
            y = 192 + i*148
            rounded_rect(dwg, (x+12, y), (col_w-24, 132), rx=12, fill="white", stroke=accent, stroke_width=1.3, shadow=True)
            text(dwg, (x+col_w/2, y+66), txt, 8, 400, COLORS["primary"])
        # Arrow to next column (center)
        if idx < 3:
            ax1 = x+col_w
            ax2 = x+col_w+col_gap
            cy = 460
            arrow(dwg, (ax1, cy), (ax2, cy), color=COLORS["primary"], width=1.6)
    # Banner
    rounded_rect(dwg, (32, 770), (W-64, 48), rx=12, fill=COLORS["primary"], stroke=COLORS["primary"], shadow=True)
    text(dwg, (W/2, 794), "DATA CONTRACT & VALIDATION — actual_validation_keys.json  •  66 isolated patients (50 MSS / 16 MSI)  •  Zero data leakage  •  Stratified 80/20 split", 8.5, 700, "white")
    # Metrics row
    metrics = [("89.39%", "Accuracy", COLORS["primary"]), ("0.9150", "ROC-AUC", COLORS["accent"]), ("0.80 / 0.75", "MSI Prec / Rec", COLORS["success"]), ("0.77", "MSI F1", COLORS["purple"]), ("30 epochs", "Training", COLORS["accent2"]), ("τ = 0.83", "Threshold", COLORS["danger"])]
    mx = 48
    for val, lab, col in metrics:
        rounded_rect(dwg, (mx, 832), (250, 64), rx=12, fill="white", stroke=col, stroke_width=1.4, shadow=True)
        text(dwg, (mx+125, 852), val, 11, 700, col)
        text(dwg, (mx+125, 876), lab, 7, 600, COLORS["slate"])
        mx += 268
    # Footer
    rounded_rect(dwg, (32, 912), (W-64, 36), rx=10, fill=COLORS["slate_light"], stroke=COLORS["border"])
    text(dwg, (W/2, 930), "STACK:  PyTorch 2.9  •  HuggingFace Transformers (Phikon-v2 / ClinicalBERT)  •  Streamlit 1.52  •  Ollama + Qwen 2.5  •  OpenCV  •  TCGA-COAD", 7, 400, COLORS["slate"])
    text(dwg, (32, 970), "PB05 • June 2026 — Dark Tissue-Aware XAI (Jet, hematoxylin, tissue mask 0.42)", 6, 400, COLORS["slate"], anchor="start")
    text(dwg, (W-32, 970), "Architecture v2.0 — Vector SVG", 6, 400, COLORS["slate"], anchor="end")
    save(dwg, "01_system_architecture")

# 02 Data Flow — 2 columns, Graphviz-like orthogonal
def diagram_02():
    W, H = 1800, 1000
    dwg = svgwrite.Drawing(size=(W, H), viewBox=f"0 0 {W} {H}")
    dwg.add(dwg.rect(insert=(0,0), size=(W,H), fill="white"))
    add_shadow_filter(dwg)
    text(dwg, (W/2, 48), "DATA FLOW DIAGRAM — Multimodal Colorectal Cancer MSI Detection", 16, 700, COLORS["primary"])
    dwg.add(dwg.line(start=(32, 72), end=(W-32, 72), stroke=COLORS["border"], stroke_width=1))
    # Left: DFD 0
    text(dwg, (W/4, 104), "DFD Level 0 — Context Diagram", 10, 700, COLORS["primary"])
    # Central process
    rounded_rect(dwg, (W/4-140, 280), (280, 140), rx=20, fill=COLORS["primary"], stroke=COLORS["primary"], shadow=True)
    text(dwg, (W/4, 350), "0\nMULTIMODAL MSI\nDETECTION SYSTEM\nPhikon-v2 + ClinicalBERT\n+ Cross-Attention", 9, 700, "white")
    # Entities
    entities = [
        ("TCGA / Kaggle\ntcga_coad_msi_mss_jpg\nH&E patches + pathology\nreports", 40, 180, "#EFF6FF", "#3B82F6"),
        ("Pathologist /\nClinician\n(Streamlit UI)", W/2-80, 180, "#FEF3C7", "#F59E0B"),
        ("Model Registry\nfusion_model_v1_1.pth\npatient_registry.json", 40, 520, "#F0FDF4", "#10B981"),
        ("Ollama LLM\nQwen 2.5-Coder:7B\nReport Scribe", W/2-80, 520, "#FDF2F8", "#EC4899"),
    ]
    for txt, x, y, bg, col in entities:
        rounded_rect(dwg, (x, y), (180, 96), rx=12, fill=bg, stroke=col, stroke_width=1.3, shadow=True)
        text(dwg, (x+90, y+48), txt, 7, 700, COLORS["primary"])
    # Compact flows (orthogonal)
    # TCGA -> System
    dwg.add(dwg.line(start=(220, 228), end=(W/4-140, 320), stroke="#3B82F6", stroke_width=1.6))
    label_pill(dwg, (W/4-80, 260), "D1: Raw JPEG + Reports", bg="#EFF6FF", border="#3B82F6", color="#3B82F6")
    # Clinician -> System
    dwg.add(dwg.line(start=(W/2-80, 228), end=(W/4+140, 280), stroke="#F59E0B", stroke_width=1.4))
    label_pill(dwg, (W/4+60, 240), "Q1: Patient ID + Text", bg="#FEF3C7", border="#F59E0B", color="#92400E")
    # Right: DFD 1
    text(dwg, (W*3/4, 104), "DFD Level 1 — Decomposed Processes & Data Stores", 10, 700, COLORS["primary"])
    procs = [
        ("1.0\nINGEST &\nPREPROCESS", W/2+40, 180, "#EFF6FF", "#3B82F6"),
        ("2.0\nENCODE\nVISION", W/2+40, 300, "#F0FDF4", "#10B981"),
        ("2.1\nENCODE\nTEXT", W/2+40, 420, "#FDF2F8", "#EC4899"),
        ("3.0\nFUSE &\nCLASSIFY", W/2+280, 300, "#FEF3C7", "#F59E0B"),
        ("4.0\nEXPLAIN\n(XAI)", W/2+280, 450, "#EDE9FE", "#8B5CF6"),
        ("5.0\nSERVE\n(Streamlit)", W/2+520, 300, "#FFF7ED", "#F97316"),
    ]
    for txt, x, y, bg, col in procs:
        rounded_rect(dwg, (x, y), (120, 80), rx=12, fill=bg, stroke=col, stroke_width=1.3, shadow=True)
        text(dwg, (x+60, y+40), txt, 7, 700, COLORS["primary"])
    # Data stores
    stores = [
        ("D1: patient_registry.json", W/2+200, 160),
        ("D2: clean_tensors/*.pt", W/2+200, 240),
        ("D3: Model Weights", W/2+200, 380),
        ("D4: Validation Keys", W/2+200, 500),
        ("D5: XAI Outputs", W/2+440, 500),
    ]
    for txt, x, y in stores:
        g = dwg.g()
        g.add(dwg.rect(insert=(x, y), size=(140, 36), rx=6, fill="white", stroke=COLORS["slate"], stroke_width=1.1))
        # left bar
        g.add(dwg.rect(insert=(x, y), size=(6, 36), rx=2, fill=COLORS["slate"]))
        g.add(dwg.text(txt, insert=(x+73, y+19), text_anchor="middle", dominant_baseline="middle", font_family=FONT, font_size=6, font_weight=600, fill=COLORS["primary"]))
        dwg.add(g)
    save(dwg, "02_data_flow_diagram")

# 03 Model comparison
def diagram_03():
    W, H = 1800, 1000
    dwg = svgwrite.Drawing(size=(W, H), viewBox=f"0 0 {W} {H}")
    dwg.add(dwg.rect(insert=(0,0), size=(W,H), fill="white"))
    add_shadow_filter(dwg)
    text(dwg, (W/2, 40), "MODEL ARCHITECTURE — Late Fusion (Baseline) vs. SummaryToPatchCrossAttention (Primary)", 13, 700, COLORS["primary"])
    dwg.add(dwg.line(start=(32, 60), end=(W-32, 60), stroke=COLORS["border"], stroke_width=1))
    # Left
    text(dwg, (W/4, 84), "A. Late Fusion Baseline — AB-MIL + Concat + MLP", 9, 700, "#9A3412")
    label_pill(dwg, (W/2-120, 84), "BASELINE", bg="#F97316", border="#F97316", color="white")
    # Inputs
    rounded_rect(dwg, (32, 100), (220, 72), rx=12, fill="#EFF6FF", stroke="#3B82F6", shadow=True)
    text(dwg, (142, 136), "INPUT A: Vision\nV ∈ ℝ^{1×N×1024}\nN patches (var., ≤1000)", 7, 700, COLORS["primary"])
    rounded_rect(dwg, (W/2-252, 100), (220, 72), rx=12, fill="#FDF2F8", stroke="#EC4899", shadow=True)
    text(dwg, (W/2-142, 136), "INPUT B: Text\nT ∈ ℝ^{1×768}\n[CLS] vector", 7, 700, COLORS["primary"])
    # AB-MIL
    rounded_rect(dwg, (32, 192), (220, 84), rx=12, fill="white", stroke="#3B82F6", shadow=True)
    text(dwg, (142, 234), "AB-MIL Attention\nLinear(1024→512) → Tanh\nLinear(512→1) → Softmax\nA ∈ ℝ^{N×1}, ΣA=1.0", 6, 400, COLORS["primary"])
    rounded_rect(dwg, (32, 292), (220, 48), rx=12, fill="#DBEAFE", stroke="#3B82F6", shadow=True)
    text(dwg, (142, 316), "Weighted Sum\nvision_summary = Aᵀ·V → [1×1024]", 7, 700, COLORS["primary"])
    # Text proj
    rounded_rect(dwg, (W/2-252, 192), (220, 84), rx=12, fill="white", stroke="#EC4899", shadow=True)
    text(dwg, (W/2-142, 234), "Text Projector\nLinear(768→512) → ReLU\nDropout(0.3)\n[1×512]", 6, 400, COLORS["primary"])
    rounded_rect(dwg, (W/2-252, 292), (220, 48), rx=12, fill="#FCE7F3", stroke="#EC4899", shadow=True)
    text(dwg, (W/2-142, 316), "text_summary [1×512]", 7, 700, COLORS["primary"])
    # Concat
    rounded_rect(dwg, (W/4-140, 380), (280, 72), rx=12, fill="#FEF3C7", stroke="#F59E0B", shadow=True)
    text(dwg, (W/4, 416), "CONCATENATE\nfused = [vision; text] → [1×1536]", 8, 700, COLORS["primary"])
    text(dwg, (W/4, 466), "Static fusion — no inter-modal interaction", 6, 400, COLORS["slate"])
    # Classifier
    rounded_rect(dwg, (W/4-160, 492), (320, 84), rx=12, fill="white", stroke="#F59E0B", shadow=True)
    text(dwg, (W/4, 534), "CLASSIFIER HEAD\nLinear(1536→256) → ReLU → Dropout(0.3)\nLinear(256→1) → Sigmoid → prob", 7, 700, COLORS["primary"])
    rounded_rect(dwg, (W/4-180, 596), (360, 44), rx=10, fill="#FFFBEB", stroke="#F59E0B")
    text(dwg, (W/4, 618), "XAI: prob + AB-MIL weights A (patch ranking)", 6, 600, COLORS["slate"])
    rounded_rect(dwg, (40, 656), (W/2-80, 48), rx=12, fill="#FFF7ED", stroke="#F97316", shadow=True)
    text(dwg, (W/4, 680), "PERF: Acc 81.82% • AUC 0.895 • MSI F1 0.50 (Recall 0.38)", 7, 700, "#9A3412")
    text(dwg, (32, 90), "⚠ ISOLATED BRANCHES — text cannot guide vision", 5, 700, "#DC2626")

    # Right
    text(dwg, (W*3/4, 84), "B. SummaryToPatchCrossAttention — Intermediate Dynamic Fusion ★ PRIMARY", 9, 700, "#065F46")
    label_pill(dwg, (W-120, 84), "PRIMARY ★", bg="#10B981", border="#10B981", color="white")
    rounded_rect(dwg, (W/2+32, 100), (220, 72), rx=12, fill="#EFF6FF", stroke="#3B82F6", shadow=True)
    text(dwg, (W/2+142, 136), "Vision V [B×N×1024]\n(uncompressed)", 7, 700, COLORS["primary"])
    rounded_rect(dwg, (W-252, 100), (220, 72), rx=12, fill="#FDF2F8", stroke="#EC4899", shadow=True)
    text(dwg, (W-142, 136), "Text T [B×768]\n[CLS] → [B×1×768]", 7, 700, COLORS["primary"])
    rounded_rect(dwg, (W-280, 192), (260, 60), rx=12, fill="white", stroke="#EC4899", shadow=True)
    text(dwg, (W-150, 222), "① TEXT PROJECTION\nLinear(768→1024)  T' = T·W → [B×1×1024]", 6, 700, COLORS["primary"])
    # Central
    rounded_rect(dwg, (W/2+40, 280), (520, 140), rx=16, fill=COLORS["primary"], stroke=COLORS["primary"], shadow=True)
    text(dwg, (W/2+300, 304), "② MULTI-HEAD CROSS-ATTENTION  (8 heads • batch_first=True)", 7, 700, "white")
    # Q K V pills
    for txt, x in [("Q = T'  [B×1×1024]", W/2+110), ("K = V  [B×N×1024]", W/2+300), ("V = V  [B×N×1024]", W/2+490)]:
        col = "#FCE7F3" if "Q" in txt else "#DBEAFE"
        c = "#9D174D" if "Q" in txt else "#1E40AF"
        dwg.add(dwg.rect(insert=(x-70, 324), size=(140, 28), rx=8, fill=col, stroke="white"))
        text(dwg, (x, 338), txt, 6, 700, c)
    # Formula
    dwg.add(dwg.rect(insert=(W/2+90, 364), size=(420, 28), rx=8, fill="#1E293B", stroke="#334155"))
    text(dwg, (W/2+300, 378), "Attn(Q,K,V) = softmax( QKᵀ / √1024 ) · V", 8, 700, "#FDE68A")
    text(dwg, (W/2+300, 400), "→ attn_output [B×1×1024]  +  attn_weights [B×1×N]  (interpretable!)", 6, 400, "white")
    rounded_rect(dwg, (W/2+40, 436), (520, 40), rx=12, fill="#FEF3C7", stroke="#F59E0B", shadow=True)
    text(dwg, (W/2+300, 456), "③ RESIDUAL + LAYERNORM  —  fused = LayerNorm(T' + attn_output)", 7, 700, COLORS["primary"])
    rounded_rect(dwg, (W/2+40, 492), (520, 64), rx=12, fill="white", stroke="#10B981", shadow=True)
    text(dwg, (W/2+300, 524), "④ SQUEEZE + CLASSIFIER  —  pooled [B×1024] → Linear(1024→256) → ReLU → Dropout → Linear→Sigmoid", 6, 700, COLORS["primary"])
    text(dwg, (W/2+300, 576), "prob ∈ [0,1] + attn_weights  •  Decision: MSI if prob ≥ 0.83 else MSS", 6, 600, COLORS["slate"])
    rounded_rect(dwg, (W/2+60, 596), (480, 48), rx=12, fill="#ECFDF5", stroke="#10B981", shadow=True)
    text(dwg, (W/2+300, 620), "PERF: Acc 89.39% • AUC 0.9150 • MSI F1 0.77 (Recall 0.75)", 7, 700, "#065F46")
    text(dwg, (W/2+32, 90), "✓ DYNAMIC TEXT-GUIDED VISUAL SEARCH — rollout product, tissue mask", 5, 700, "#065F46")
    save(dwg, "03_model_architecture_comparison")
def diagram_04():
    W, H = 1800, 1000
    dwg = svgwrite.Drawing(size=(W, H), viewBox=f"0 0 {W} {H}")
    dwg.add(dwg.rect(insert=(0,0), size=(W,H), fill="white"))
    text(dwg, (W/2, 48), "SEQUENCE DIAGRAMS — Training Pipeline (Top) & Inference Pipeline (Bottom)", 14, 700, COLORS["primary"])
    dwg.add(dwg.line(start=(32, 72), end=(W-32, 72), stroke=COLORS["border"], stroke_width=1))
    # Top title
    dwg.add(dwg.rect(insert=(32, 84), size=(520, 28), rx=8, fill=COLORS["primary"]))
    text(dwg, (292, 98), "A. TRAINING PHASE — 30 Epochs • Weighted BCE • AdamW", 7, 700, "white")
    actors = ["DataLoader\n(Batch=1)", "Multimodal\nDataset", "Phikon-v2\n(Frozen)", "ClinicalBERT\n(Frozen)", "Cross-Attention\nModel", "Weighted\nBCE Loss", "AdamW\nOptimizer"]
    xs = [120, 340, 560, 760, 1000, 1240, 1480]
    for x, name in zip(xs, actors):
        rounded_rect(dwg, (x-65, 132), (130, 48), rx=12, fill="white", stroke=COLORS["primary"], shadow=False)
        text(dwg, (x, 156), name, 7, 700, COLORS["primary"])
        dwg.add(dwg.line(start=(x, 180), end=(x, 340), stroke=COLORS["slate"], stroke_width=0.9))
        dwg.add(dwg.line(start=(x, 180), end=(x, 340), stroke=COLORS["slate"], stroke_width=0.9, stroke_dasharray="6,4"))
    # Messages
    dwg.add(dwg.line(start=(120, 210), end=(340, 210), stroke=COLORS["primary"], stroke_width=1.4))
    dwg.add(dwg.polygon(points=[(340,210),(332,206),(332,214)], fill=COLORS["primary"]))
    text(dwg, (230, 202), "1. __getitem__(idx) → {vision [N×1024], text [768], label}", 6, 700, COLORS["primary"])
    text(dwg, (230, 232), "2. Dynamic cap: randperm 1000 if N>1000", 6, 400, COLORS["slate"])
    dwg.add(dwg.line(start=(120, 260), end=(1000, 260), stroke="#3B82F6", stroke_width=1.4))
    dwg.add(dwg.polygon(points=[(1000,260),(992,256),(992,264)], fill="#3B82F6"))
    text(dwg, (560, 252), "3. vision [1×N×1024] + text [1×768] → forward()", 6, 700, "#3B82F6")
    text(dwg, (1000, 286), "4. Text Proj 768→1024 + 8-head Cross-Attn + LayerNorm", 6, 400, COLORS["primary"], anchor="middle")
    dwg.add(dwg.line(start=(1000, 310), end=(1240, 310), stroke="#F59E0B", stroke_width=1.4))
    dwg.add(dwg.polygon(points=[(1240,310),(1232,306),(1232,314)], fill="#F59E0B"))
    text(dwg, (1120, 302), "5. prob + attn_weights", 6, 700, "#92400E")
    dwg.add(dwg.line(start=(1000, 330), end=(1240, 330), stroke=COLORS["danger"], stroke_width=1.4))
    dwg.add(dwg.polygon(points=[(1240,330),(1232,326),(1232,334)], fill=COLORS["danger"]))
    text(dwg, (1120, 322), "6. L = -[4.17·y·log(p)+(1-y)·log(1-p)]", 6, 700, COLORS["danger"])
    rounded_rect(dwg, (440, 352), (920, 28), rx=8, fill=COLORS["slate_light"], stroke=COLORS["border"])
    text(dwg, (900, 366), "Loop: Train 261 → Val 65 — early stopping on best AUC → Save best_fusion_model.pth", 6, 400, COLORS["slate"])
    # Bottom
    dwg.add(dwg.rect(insert=(32, 400), size=(540, 28), rx=8, fill=COLORS["purple"]))
    text(dwg, (302, 414), "B. INFERENCE PHASE — Streamlit → Model → XAI → LLM Scribe", 7, 700, "white")
    actors2 = ["Clinician\n(Browser)", "Streamlit\nUI (app.py)", "Cross-Attention\nModel", "Phikon-v2\nViT Rollout", "Ollama\nQwen 2.5", "File System\n(Assets)"]
    xs2 = [120, 360, 600, 860, 1120, 1420]
    for x, name in zip(xs2, actors2):
        rounded_rect(dwg, (x-65, 448), (130, 48), rx=12, fill="white", stroke=COLORS["purple"], shadow=False)
        text(dwg, (x, 472), name, 7, 700, COLORS["primary"])
        dwg.add(dwg.line(start=(x, 496), end=(x, 700), stroke=COLORS["slate"], stroke_width=0.9, stroke_dasharray="6,4"))
    dwg.add(dwg.line(start=(120, 540), end=(360, 540), stroke=COLORS["purple"], stroke_width=1.4))
    dwg.add(dwg.polygon(points=[(360,540),(352,536),(352,544)], fill=COLORS["purple"]))
    text(dwg, (240, 532), "1. Enter Patient ID + edit report", 6, 700, COLORS["purple"])
    dwg.add(dwg.line(start=(360, 580), end=(600, 580), stroke=COLORS["primary"], stroke_width=1.4))
    dwg.add(dwg.polygon(points=[(600,580),(592,576),(592,584)], fill=COLORS["primary"]))
    text(dwg, (480, 572), "2. Load *_vision.pt + *_text.pt → forward()", 6, 700, COLORS["primary"])
    dwg.add(dwg.line(start=(600, 620), end=(360, 620), stroke=COLORS["success"], stroke_width=1.4))
    dwg.add(dwg.polygon(points=[(360,620),(368,616),(368,624)], fill=COLORS["success"]))
    text(dwg, (480, 612), "3. Return prob=0.87 + attn_weights [1×N]", 6, 700, COLORS["success"])
    text(dwg, (600, 656), "4a. argsort → Top-5 → Gallery   |   4b. Phikon rollout (6 layers, Jet dark) • 5. Ollama POST → 6. 3-paragraph report", 6, 400, COLORS["slate"])
    rounded_rect(dwg, (440, 680), (920, 28), rx=8, fill="#EDE9FE", stroke="#C4B5FD")
    text(dwg, (900, 694), "7. Render: Probability meter + Top-5 gallery + Dark heatmap + Scribe report (tissue-aware)", 7, 700, COLORS["primary"])
    save(dwg, "04_sequence_diagram")

# 05 Deployment
def diagram_05():
    W, H = 1700, 1000
    dwg = svgwrite.Drawing(size=(W, H), viewBox=f"0 0 {W} {H}")
    dwg.add(dwg.rect(insert=(0,0), size=(W,H), fill="white"))
    text(dwg, (W/2, 48), "DEPLOYMENT ARCHITECTURE — Streamlit Clinical Deployment & Infrastructure", 14, 700, COLORS["primary"])
    dwg.add(dwg.line(start=(32, 72), end=(W-32, 72), stroke=COLORS["border"], stroke_width=1))
    # Client
    rounded_rect(dwg, (32, 100), (280, 120), rx=12, fill="#EFF6FF", stroke="#3B82F6", shadow=False)
    text(dwg, (172, 160), "CLIENT TIER\nClinician Workstation\nChrome / Safari / Edge\nBrowser only — HTTPS :8501", 8, 700, COLORS["primary"])
    # App
    rounded_rect(dwg, (360, 100), (620, 260), rx=16, fill="white", stroke=COLORS["primary"], shadow=False)
    dwg.add(dwg.rect(insert=(360, 100), size=(620, 36), rx=12, fill=COLORS["primary"]))
    text(dwg, (670, 118), "APPLICATION TIER — Streamlit Server (src/ui/app.py)", 8, 700, "white")
    rounded_rect(dwg, (376, 152), (280, 48), rx=12, fill="#EDE9FE", stroke="#8B5CF6")
    text(dwg, (516, 176), "Session State\nanalysis_complete • prob • attn", 7, 400, COLORS["primary"])
    rounded_rect(dwg, (680, 152), (280, 48), rx=12, fill="#FEF3C7", stroke="#F59E0B")
    text(dwg, (820, 176), "Device Router\ncuda → mps → cpu", 7, 400, COLORS["primary"])
    rounded_rect(dwg, (376, 216), (600, 40), rx=12, fill="#ECFDF5", stroke="#10B981")
    text(dwg, (676, 236), "Inference Engine: @st.cache_resource — CrossAttention + fusion_model_v1_1.pth", 7, 400, COLORS["primary"])
    for txt, x in [("Input Panel", 396), ("Prob Meter", 536), ("Top-5 Gallery", 676), ("Dark Heatmap", 816)]:
        rounded_rect(dwg, (x, 272), (120, 56), rx=12, fill="white", stroke="#8B5CF6")
        text(dwg, (x+60, 300), txt, 6, 700, COLORS["primary"])
    # Data tier
    rounded_rect(dwg, (200, 400), (1060, 200), rx=16, fill="white", stroke=COLORS["slate"])
    dwg.add(dwg.rect(insert=(200, 400), size=(1060, 32), rx=12, fill=COLORS["slate"]))
    text(dwg, (730, 416), "DATA & MODEL TIER — Persistent Storage", 8, 700, "white")
    for txt, x, bg, col in [("patient_registry.json\n326 patients", 220, "#EFF6FF", "#3B82F6"), ("clean_tensors/\n653 × .pt", 460, "#F0FDF4", "#10B981"), ("models/*.pth\nfusion_model_v1_1.pth", 700, "#FEF3C7", "#F59E0B"), ("archive/*.JPG\n192k patches", 940, "#FDF2F8", "#EC4899")]:
        rounded_rect(dwg, (x, 452), (200, 72), rx=12, fill=bg, stroke=col)
        text(dwg, (x+100, 488), txt, 7, 700, COLORS["primary"])
    rounded_rect(dwg, (460, 544), (480, 32), rx=8, fill=COLORS["slate_light"], stroke=COLORS["border"])
    text(dwg, (700, 560), "actual_validation_keys.json • 66 isolated • Zero leakage", 6, 600, COLORS["slate"])
    # LLM
    rounded_rect(dwg, (1060, 100), (360, 260), rx=16, fill="#FDF2F8", stroke="#EC4899")
    dwg.add(dwg.rect(insert=(1060, 100), size=(360, 36), rx=12, fill="#EC4899"))
    text(dwg, (1240, 118), "LLM TIER — Ollama + Qwen 2.5", 8, 700, "white")
    for txt, y in [("Ollama :11434 /api/generate", 160), ("Qwen 2.5-Coder:7B", 210), ("3-paragraph report", 270)]:
        rounded_rect(dwg, (1080, y), (320, 40), rx=12, fill="white", stroke="#EC4899")
        text(dwg, (1240, y+20), txt, 7, 700, COLORS["primary"])
    # Infra footer
    rounded_rect(dwg, (32, 640), (W-64, 160), rx=16, fill=COLORS["primary"], stroke=COLORS["primary"])
    text(dwg, (W/2, 672), "INFRASTRUCTURE & DEVOPS", 9, 700, "#FDE68A")
    for txt, x in [("Runtime\nPython 3.10+\nstreamlit run", 60), ("Hardware\nCUDA / MPS / CPU", 340), ("Dependencies\n75 pkgs • timm", 620), ("Scaling\nDocker • Cloud", 880), ("Monitoring\nCM / ROC / Loss", 1140), ("Security\nPHI stripped", 1420)]:
        text(dwg, (x, 710), txt, 7, 700, "white", anchor="start")
        text(dwg, (x, 740), txt.split("\n",1)[1] if "\n" in txt else "", 6, 400, "#CBD5E1", anchor="start")
    save(dwg, "05_deployment_architecture")

# 06 Component
def diagram_06():
    W, H = 1700, 1000
    dwg = svgwrite.Drawing(size=(W, H), viewBox=f"0 0 {W} {H}")
    dwg.add(dwg.rect(insert=(0,0), size=(W,H), fill="white"))
    text(dwg, (W/2, 48), "COMPONENT & CLASS DIAGRAM — Codebase Structure (src/)", 14, 700, COLORS["primary"])
    dwg.add(dwg.line(start=(32, 72), end=(W-32, 72), stroke=COLORS["border"], stroke_width=1))
    def class_box(x, y, w, h, title, attrs, methods, color):
        dwg.add(dwg.rect(insert=(x, y+h-36), size=(w, 36), rx=12, fill=color))
        text(dwg, (x+w/2, y+h-18), title, 7, 700, "white")
        rounded_rect(dwg, (x, y), (w, h-36), rx=12, fill="white", stroke=color, shadow=False)
        text(dwg, (x+12, y+20), "Attributes:", 6, 700, color, anchor="start")
        for i, a in enumerate(attrs):
            text(dwg, (x+18, y+38+i*16), "• " + a, 6, 400, COLORS["primary"], anchor="start")
        dwg.add(dwg.line(start=(x+8, y+ 38+len(attrs)*16+4), end=(x+w-8, y+38+len(attrs)*16+4), stroke=COLORS["border"], stroke_width=0.8))
        text(dwg, (x+12, y+60+len(attrs)*16), "Methods:", 6, 700, color, anchor="start")
        for i,m in enumerate(methods):
            text(dwg, (x+18, y+78+len(attrs)*16+i*16), "• " + m, 6, 400, COLORS["slate"], anchor="start")
    class_box(32, 100, 380, 280, "MultimodalTCGADataset (Dataset)", ["tensor_dir: str", "max_patches=1000", "valid_patients: List"], ["__init__(registry, dir)", "__len__()", "__getitem__(idx)"], "#3B82F6")
    class_box(452, 100, 380, 280, "MultimodalFusionNetwork", ["vision_attention: Seq", "text_projector: Seq", "hidden_dim=512"], ["forward(V[1×N×1024], T[1×768])", "→ prob, A[N×1]"], "#F59E0B")
    class_box(872, 100, 400, 280, "SummaryToPatchCrossAttention ★", ["text_proj: Linear 768→1024", "cross_attn: MHA 8h", "layer_norm: 1024"], ["forward(V[B×N×1024], T[B×768])", "→ prob, attn[B×1×N]"], "#10B981")
    class_box(1312, 100, 360, 280, "Streamlit App (app.py)", ["MODEL_WEIGHTS", "device: cuda/mps/cpu", "session_state"], ["load_system()", "generate_clinical_report()", "Run Pipeline"], "#8B5CF6")
    class_box(32, 420, 380, 240, "Training Scripts", ["EPOCHS=30, LR=1e-4", "pos_weight=4.17"], ["set_seed(42)", "weighted_bce_loss()", "train_model()"], "#64748B")
    class_box(452, 420, 380, 240, "Evaluation Scripts", ["threshold=0.83", "actual_validation_keys.json"], ["evaluate_model()", "plot_cross_attention()"], "#64748B")
    class_box(872, 420, 400, 240, "XAI Engine", ["Phikon-v2: 24 layers", "tissue_mask: HSV+Otsu", "hematoxylin OD"], ["generate_smooth_vit_heatmap()", "rollout product", "dark Jet overlay"], "#EC4899")
    class_box(1312, 420, 360, 240, "Verification Suite", ["verify_cross_attention.py", "Jensen-Shannon"], ["run_perturbation_test()", "Real vs Fake shift"], "#0EA5E9")
    save(dwg, "06_component_class_diagram")

# 07 Training
def diagram_07():
    W, H = 1700, 1100
    dwg = svgwrite.Drawing(size=(W, H), viewBox=f"0 0 {W} {H}")
    dwg.add(dwg.rect(insert=(0,0), size=(W,H), fill="white"))
    text(dwg, (W/2, 48), "TRAINING & EVALUATION PIPELINE — Optimization, Metrics & Loss Dynamics", 14, 700, COLORS["primary"])
    dwg.add(dwg.line(start=(32, 72), end=(W-32, 72), stroke=COLORS["border"], stroke_width=1))
    dwg.add(dwg.rect(insert=(32, 84), size=(520, 28), rx=8, fill=COLORS["primary"]))
    text(dwg, (292, 98), "TRAINING LOOP — 30 Epochs • 80/20 Split • Weighted BCE 4.17", 7, 700, "white")
    steps = [
        ("1. Seed\nset_seed(42)", "#EFF6FF", "#3B82F6"),
        ("2. Dataset:\n326 patients\n63 MSI / 263 MSS", "#F0FDF4", "#10B981"),
        ("3. Split:\n80% Train 261\n20% Val 65", "#FEF3C7", "#F59E0B"),
        ("4. DataLoader:\nbatch=1\nmax_patches=1000", "#EDE9FE", "#8B5CF6"),
        ("5. Model:\nCrossAttention\n8-head MHA", "#ECFDF5", "#10B981"),
        ("6. Forward:\nprob, attn\nWeighted BCE", "#FEF3C7", "#F59E0B"),
        ("7. Backward:\nAdamW 1e-4", "#FDF2F8", "#EC4899"),
        ("8. Validate:\nROC-AUC\nSave best", "#FFF7ED", "#F97316"),
        ("9. Plot:\nLoss 4.17", "#EFF6FF", "#3B82F6"),
    ]
    x = 32
    for txt, bg, col in steps:
        rounded_rect(dwg, (x, 132), (172, 96), rx=12, fill=bg, stroke=col, shadow=False)
        text(dwg, (x+86, 180), txt, 6, 700, COLORS["primary"])
        x += 188
    rounded_rect(dwg, (32, 250), (W-64, 48), rx=12, fill="white", stroke=COLORS["danger"])
    text(dwg, (W/2, 274), "WEIGHTED BCE:  L = −[ w⁺·y·log(ŷ) + (1−y)·log(1−ŷ) ]   w⁺ = 263/63 ≈ 4.17  (clamped ŷ∈[1e-7,1-1e-7])", 8, 700, COLORS["danger"])
    rounded_rect(dwg, (32, 310), (W-64, 32), rx=8, fill="#1E293B", stroke="#1E293B")
    text(dwg, (W/2, 326), "DECISION:  Late Fusion τ=0.70  •  Cross-Attention τ=0.83 (conservative)  •  MSI if prob ≥ τ", 7, 700, "white")
    # Metrics table
    dwg.add(dwg.rect(insert=(32, 368), size=(W-64, 28), rx=8, fill=COLORS["primary"]))
    text(dwg, (W/2, 382), "EVALUATION — Isolated Validation (66 patients) • No Leakage • Confusion Matrix + ROC", 7, 700, "white")
    headers = ["Metric", "Late Fusion (τ=0.70)", "Cross-Attention (τ=0.83) ★", "Δ Improvement"]
    xs = [32, 432, 832, 1232]
    ws = [400, 400, 400, 368]
    for hx, w, h in zip(xs, ws, headers):
        dwg.add(dwg.rect(insert=(hx, 412), size=(w, 32), rx=8, fill=COLORS["primary"]))
        text(dwg, (hx+w/2, 428), h, 7, 700, "white")
    rows = [
        ("Accuracy", "81.82%", "89.39%", "+7.57 pp", "#10B981"),
        ("ROC-AUC", "0.8950", "0.9150", "+0.020", "#0EA5E9"),
        ("MSI Precision / Recall", "0.78 / 0.44", "0.80 / 0.75", "+0.02 / +0.31 ★", "#10B981"),
        ("MSI F1", "0.50", "0.77", "+0.27 (+54%)", "#10B981"),
        ("Macro F1", "0.69", "0.85", "+0.16", "#8B5CF6"),
    ]
    y = 460
    for metric, late, cross, delta, col in rows:
        bg = "#F8FAFC" if rows.index((metric, late, cross, delta, col))%2==0 else "white"
        for hx, w, txt, c in [(xs[0], ws[0], metric, COLORS["primary"]), (xs[1], ws[1], late, "#9A3412"), (xs[2], ws[2], cross, "#065F46"), (xs[3], ws[3], delta, col)]:
            dwg.add(dwg.rect(insert=(hx, y), size=(w, 28), rx=6, fill=bg, stroke=COLORS["border"], stroke_width=0.8))
            text(dwg, (hx+w/2, y+14), txt, 7, 700, c)
        y += 32
    rounded_rect(dwg, (32, 640), (W-64, 48), rx=12, fill="#FFFBEB", stroke="#FDE68A")
    text(dwg, (W/2, 664), "Late Fusion misses 9/16 MSI (56% FN)  •  Cross-Attn catches 12/16 MSI (only 4 FN) — text-guided search doubles MSI detection", 7, 700, COLORS["primary"])
    save(dwg, "07_training_evaluation_pipeline")

# 08 XAI
def diagram_08():
    W, H = 1700, 1000
    dwg = svgwrite.Drawing(size=(W, H), viewBox=f"0 0 {W} {H}")
    dwg.add(dwg.rect(insert=(0,0), size=(W,H), fill="white"))
    text(dwg, (W/2, 48), "EXPLAINABLE AI (XAI) STACK — 3-Tier Interpretability System (Dark Tissue-Aware)", 14, 700, COLORS["primary"])
    text(dwg, (W/2, 72), "Macroscopic (patch ranking) → Microscopic (hematoxylin-aware Jet) → Linguistic (LLM scribe) — tissue-aware, not white", 8, 400, COLORS["slate"])
    dwg.add(dwg.line(start=(32, 88), end=(W-32, 88), stroke=COLORS["border"], stroke_width=1))
    # Tier 1
    rounded_rect(dwg, (32, 104), (520, 280), rx=16, fill="white", stroke="#3B82F6")
    dwg.add(dwg.rect(insert=(32, 104), size=(520, 36), rx=12, fill="#3B82F6"))
    text(dwg, (292, 122), "TIER 1 — MACROSCOPIC XAI", 8, 700, "white")
    text(dwg, (292, 150), "Q: Where did the model look?", 7, 700, "#3B82F6")
    for i, txt in enumerate(["① Intercept attn_weights [B×1×N]", "② Average 8 heads → scalar", "③ argsort descending → Top-5", "④ Gallery: 5-col, red border", "⑤ Hematoxylin-aware ranking"]):
        rounded_rect(dwg, (48, 168+i*36), (488, 28), rx=8, fill="#EFF6FF", stroke="#93C5FD")
        text(dwg, (292, 182+i*36), txt, 6, 600, COLORS["primary"])
    # Tier 2
    rounded_rect(dwg, (592, 104), (540, 280), rx=16, fill="white", stroke="#EC4899")
    dwg.add(dwg.rect(insert=(592, 104), size=(540, 36), rx=12, fill="#EC4899"))
    text(dwg, (862, 122), "TIER 2 — MICROSCOPIC XAI (Corrected)", 8, 700, "white")
    text(dwg, (862, 150), "Q: What is inside that patch? (tissue, not white)", 7, 700, "#EC4899")
    for i, txt in enumerate(["① Tissue mask: HSV Otsu + white <210", "② Hematoxylin OD = -log(I)·[0.65,0.70,0.29]", "③ Pseudo = H·sat·blue·darkness", "④ Jet α0.68 on tissue, white=0", "⑤ Rollout product (not mean)"]):
        rounded_rect(dwg, (608, 168+i*36), (508, 28), rx=8, fill="#FDF2F8", stroke="#F9A8D4")
        text(dwg, (862, 182+i*36), txt, 6, 600, COLORS["primary"])
    # Tier 3
    rounded_rect(dwg, (1172, 104), (496, 280), rx=16, fill="white", stroke="#8B5CF6")
    dwg.add(dwg.rect(insert=(1172, 104), size=(496, 36), rx=12, fill="#8B5CF6"))
    text(dwg, (1420, 122), "TIER 3 — LINGUISTIC XAI", 8, 700, "white")
    text(dwg, (1420, 150), "Q: What does it mean clinically?", 7, 700, "#8B5CF6")
    for i, txt in enumerate(["① Context pkg {prob, top5, report}", "② Deterministic prompt (no halluc)", "③ POST localhost:11434", "④ Qwen 2.5 → 3 paragraphs", "⑤ Fallback template if offline"]):
        rounded_rect(dwg, (1188, 168+i*36), (464, 28), rx=8, fill="#EDE9FE", stroke="#C4B5FD")
        text(dwg, (1420, 182+i*36), txt, 6, 600, COLORS["primary"])
    # Verification
    rounded_rect(dwg, (32, 420), (W-64, 240), rx=16, fill="white", stroke=COLORS["primary"])
    dwg.add(dwg.rect(insert=(32, 420), size=(W-64, 36), rx=12, fill=COLORS["primary"]))
    text(dwg, (W/2, 438), "VERIFICATION — Perturbation Test + Tissue Mask Inversion Check", 8, 700, "white")
    rounded_rect(dwg, (48, 476), (760, 72), rx=12, fill="#FEE2E2", stroke="#FCA5A5")
    text(dwg, (428, 512), "BEFORE (inverted): pseudo = 0.6*blur(gray) → white high → Jet on glass", 7, 700, COLORS["danger"])
    rounded_rect(dwg, (848, 476), (760, 72), rx=12, fill="#D1FAE5", stroke="#6EE7B7")
    text(dwg, (1228, 512), "AFTER (corrected): tissue_mask → hema = H·sat·blue → white=0 → Jet on nuclei", 7, 700, "#065F46")
    rounded_rect(dwg, (48, 568), (W-96, 32), rx=8, fill=COLORS["slate_light"], stroke=COLORS["border"])
    text(dwg, (W/2, 584), "Test: 3 patches (17% / 13% / 94% tissue) — tissue_mean 0.325 > white_mean 0.269 ✓ not inverted • overlay mean ~103 dark ✓", 6, 600, COLORS["slate"])
    save(dwg, "08_xai_stack")

# 09 UseCase+ER
def diagram_09():
    W, H = 1800, 1000
    dwg = svgwrite.Drawing(size=(W, H), viewBox=f"0 0 {W} {H}")
    dwg.add(dwg.rect(insert=(0,0), size=(W,H), fill="white"))
    text(dwg, (W/2, 48), "USE CASE DIAGRAM & ENTITY-RELATIONSHIP DIAGRAM", 14, 700, COLORS["primary"])
    dwg.add(dwg.line(start=(32, 72), end=(W-32, 72), stroke=COLORS["border"], stroke_width=1))
    text(dwg, (W/4, 100), "Use Case — Actors & System Interactions", 9, 700, COLORS["primary"])
    dwg.add(dwg.rect(insert=(W/4-280, 120), size=(560, 680), rx=16, fill="white", stroke=COLORS["primary"], stroke_width=1.4))
    text(dwg, (W/4, 140), "MSI Detection System", 7, 700, COLORS["primary"])
    # Actors
    for txt, x, y, bg, col in [("Pathologist", 80, 200, "#EFF6FF", "#3B82F6"), ("ML Engineer", 80, 320, "#F0FDF4", "#10B981"), ("Clinician", 1600, 200, "#FEF3C7", "#F59E0B"), ("System", 1600, 460, "#EDE9FE", "#8B5CF6")]:
        rounded_rect(dwg, (x-60, y-20), (120, 40), rx=20, fill=bg, stroke=col)
        text(dwg, (x, y), txt, 7, 700, COLORS["primary"])
    # UC ellipses
    ucs = [("UC1: Upload\nPatient Data", W/4-100, 220), ("UC2: Run\nDiagnosis", W/4+100, 220), ("UC3: View Top-5", W/4-100, 320), ("UC4: Generate\nHeatmap", W/4+100, 320), ("UC5: Read LLM\nReport", W/4-100, 420), ("UC6: Train", W/4-100, 540), ("UC7: Evaluate", W/4+100, 540), ("UC8: Verify", W/4-100, 640), ("UC9: Manage\nRegistry", W/4+100, 640)]
    for txt, x, y in ucs:
        dwg.add(dwg.ellipse(center=(x, y), r=(80, 32), fill="white", stroke=COLORS["primary"], stroke_width=1.2))
        text(dwg, (x, y), txt, 6, 700, COLORS["primary"])
    # ER right
    text(dwg, (W*3/4, 100), "Entity-Relationship — Data Model", 9, 700, COLORS["primary"])
    def er_box(x, y, w, h, title, attrs, color):
        dwg.add(dwg.rect(insert=(x, y+h-32), size=(w, 32), rx=8, fill=color))
        text(dwg, (x+w/2, y+h-16), title, 7, 700, "white")
        rounded_rect(dwg, (x, y), (w, h-32), rx=12, fill="white", stroke=color)
        for i, a in enumerate(attrs):
            text(dwg, (x+12, y+20+i*18), a, 6, 600 if "PK" in a else 400, COLORS["primary"] if "PK" in a else COLORS["slate"], anchor="start")
    er_box(W/2+40, 140, 240, 160, "Patient", ["PK patient_id (TCGA-*)", "label: 0/1", "report_text", "image_paths[]"], "#3B82F6")
    er_box(W/2+340, 140, 240, 160, "TensorStore", ["PK patient_id (FK)", "vision [N×1024]", "text [768]"], "#10B981")
    er_box(W/2+640, 140, 240, 160, "ModelWeight", ["PK model_name", "arch: Late/Cross", "threshold 0.83"], "#F59E0B")
    er_box(W/2+40, 340, 240, 140, "ValidationSplit", ["PK split_id", "FK patient_id", "split: train/val"], "#8B5CF6")
    er_box(W/2+340, 340, 240, 140, "InferenceResult", ["PK patient_id (FK)", "prob, attn", "top5"], "#EC4899")
    er_box(W/2+640, 340, 240, 140, "XAIOutput", ["PK patient_id (FK)", "heatmaps", "scribe"], "#0EA5E9")
    save(dwg, "09_usecase_er_diagram")

# 10 State & Flowchart
def diagram_10():
    W, H = 1800, 1000
    dwg = svgwrite.Drawing(size=(W, H), viewBox=f"0 0 {W} {H}")
    dwg.add(dwg.rect(insert=(0,0), size=(W,H), fill="white"))
    text(dwg, (W/2, 48), "STATE TRANSITION & OPERATIONAL FLOWCHART", 14, 700, COLORS["primary"])
    dwg.add(dwg.line(start=(32, 72), end=(W-32, 72), stroke=COLORS["border"], stroke_width=1))
    text(dwg, (W/4, 100), "State Diagram — Clinical Inference Lifecycle", 9, 700, COLORS["primary"])
    states = [("IDLE\nAwaiting Input", W/4, 160, COLORS["slate"]), ("LOADING\nLoad tensors", 180, 260, "#3B82F6"), ("INFERENCING\nCross-Attn", W/4, 260, "#F59E0B"), ("XAI READY\nTop-5", 720, 260, "#10B981"), ("HEATMAP\nGENERATING", 180, 380, "#EC4899"), ("SCRIBE\nGENERATING", 720, 380, "#8B5CF6"), ("DISPLAY\nRendered", W/4, 380, COLORS["primary"]), ("ERROR\nRecoverable", W/4, 480, COLORS["danger"])]
    for txt, x, y, col in states:
        dwg.add(dwg.rect(insert=(x-70, y-28), size=(140, 56), rx=28, fill=col, stroke=col))
        text(dwg, (x, y), txt, 6, 700, "white")
    text(dwg, (W*3/4, 100), "Flowchart — End-to-End Operational Pipeline", 9, 700, COLORS["primary"])
    # Flowchart
    def flow_box(x, y, w, h, txt, shape="rect", col=COLORS["primary"], fill="white"):
        if shape=="terminator":
            dwg.add(dwg.rect(insert=(x-w/2, y-h/2), size=(w, h), rx=20, fill=col, stroke=col))
            text(dwg, (x, y), txt, 7, 700, "white")
        elif shape=="diamond":
            pts = [(x, y-h/2),(x+w/2, y),(x, y+h/2),(x-w/2, y)]
            dwg.add(dwg.polygon(points=pts, fill="#FEF3C7", stroke="#F59E0B", stroke_width=1.3))
            text(dwg, (x, y), txt, 6, 700, "#92400E")
        elif shape=="io":
            # parallelogram
            pts = [(x-w/2+12, y-h/2),(x+w/2+12, y-h/2),(x+w/2-12, y+h/2),(x-w/2-12, y+h/2)]
            dwg.add(dwg.polygon(points=pts, fill=fill, stroke=col, stroke_width=1.2))
            text(dwg, (x, y), txt, 7, 700, col)
        else:
            rounded_rect(dwg, (x-w/2, y-h/2), (w, h), rx=12, fill=fill, stroke=col)
            text(dwg, (x, y), txt, 6, 700, COLORS["primary"])
    flow_box(W*3/4, 150, 160, 32, "START", "terminator", COLORS["primary"])
    flow_box(W*3/4, 200, 280, 40, "Load registry + tensors", "io", "#3B82F6", "#EFF6FF")
    flow_box(W*3/4, 260, 160, 48, "Patient ID\nin registry?", "diamond")
    flow_box(W*3/4-160, 340, 160, 48, "Show ERROR\n'Not found'", shape="rect", col=COLORS["danger"], fill="#FEE2E2")
    flow_box(W*3/4, 340, 260, 40, "Load vision [N×1024]\n+ text [768] → device", shape="rect")
    flow_box(W*3/4, 400, 280, 40, "Forward: CrossAttention\nprob, attn_weights", shape="rect", col="#F59E0B", fill="#FFFBEB")
    flow_box(W*3/4, 460, 140, 48, "prob ≥ 0.83 ?", "diamond")
    flow_box(W*3/4, 520, 280, 40, "Compute Top-5 • Scribe", shape="rect", col="#8B5CF6", fill="#EDE9FE")
    flow_box(W*3/4, 580, 280, 40, "Render: Dashboard\nMeter + Gallery + Heatmap", shape="io", col="#8B5CF6", fill="#EDE9FE")
    flow_box(W*3/4, 640, 120, 40, "heatmap?", "diamond")
    flow_box(W*3/4-120, 700, 160, 40, "Phikon rollout\nJet dark tissue", shape="rect", col="#EC4899", fill="#FDF2F8")
    flow_box(W*3/4, 760, 120, 32, "END", "terminator", COLORS["primary"])
    save(dwg, "10_state_flowchart")

if __name__ == "__main__":
    diagram_01()
    diagram_02()
    diagram_03()
    diagram_04()
    diagram_05()
    diagram_06()
    diagram_07()
    diagram_08()
    diagram_09()
    diagram_10()
    print("✅ All 10 vector diagrams done")
