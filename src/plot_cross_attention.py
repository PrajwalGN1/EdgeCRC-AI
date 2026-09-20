import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch


def plot_cross_attention_matrix(attention_matrix, text_tokens, patch_labels,
                                save_path="../results/Cross_Attention_Matrix.png"):
    """
    Generates a publication-ready heatmap of the cross-attention weights.

    Args:
        attention_matrix: 2D numpy array of shape (len(text_tokens), len(patch_labels))
        text_tokens: List of strings (the words/features from the text)
        patch_labels: List of strings (the patch IDs from the image)
    """
    print("📊 Generating Cross-Attention Matrix diagram...")

    # 1. Figure Setup
    plt.figure(figsize=(12, 8), facecolor='white')
    sns.set_theme(style="white")

    # 2. Draw the Heatmap
    # 'mako' or 'rocket' are excellent, perceptually uniform colormaps for attention matrices
    ax = sns.heatmap(
        attention_matrix,
        cmap="mako",  # Deep blue/purple to bright yellow
        cbar_kws={'label': 'Attention Weight (Softmax Probability)'},
        linewidths=0.5,  # Adds a tiny grid line between cells
        linecolor='white',
        square=True,  # Forces cells to be perfectly square
        vmin=0.0,
        vmax=np.max(attention_matrix) * 1.2  # Give the hottest spot some headroom
    )

    # 3. Labeling & Styling
    ax.set_title("Summary-to-Patch Cross-Attention Matrix\n(Token vs. Patch Mapping)",
                 fontsize=16, fontweight='bold', pad=20)

    ax.set_xlabel("Vision Transformer Image Patches (N)", fontsize=13, fontweight='bold', labelpad=15)
    ax.set_ylabel("Clinical Summary Text Tokens (M)", fontsize=13, fontweight='bold', labelpad=15)

    # Set the tick labels using the words and patch IDs
    ax.set_xticklabels(patch_labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(text_tokens, rotation=0, fontsize=11, fontweight='medium')

    # 4. Save High-Res Output
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved publication-ready cross-attention matrix to '{save_path}'")


def run_synthetic_demo():
    """
    Runs the visualization using synthetic data so you can test the formatting immediately.
    """
    print("⚙️ Initializing synthetic attention generation...")

    # 1. Define M (Text Tokens) - A realistic slice of an LLM summary
    text_tokens = [
        "[CLS]", "patient", "exhibits", "high", "microsatellite",
        "instability", "with", "dense", "lymphocytic", "infiltrate"
    ]

    # 2. Define N (Image Patches) - A slice of patches from the ViT
    patch_labels = [f"Patch_{i}" for i in range(45, 60)]  # 15 patches

    # 3. Generate Mock Attention Matrix (M x N)
    # Give it a baseline low attention (noise)
    mock_attention = np.random.uniform(low=0.00, high=0.05, size=(len(text_tokens), len(patch_labels)))

    # Inject "hot spots" (simulating the model correctly matching text to specific visual patches)
    # E.g., The word "instability" (index 5) lights up patches 50, 51, 52
    mock_attention[5, 5:8] = np.random.uniform(0.6, 0.9, size=3)
    # E.g., The word "lymphocytic" (index 8) lights up patches 46, 58
    mock_attention[8, 1] = 0.85
    mock_attention[8, 13] = 0.75
    # The [CLS] token usually has broad, uniform attention
    mock_attention[0, :] = np.random.uniform(0.1, 0.2, size=len(patch_labels))

    # Normalize rows to somewhat simulate softmax behavior
    mock_attention = mock_attention / mock_attention.sum(axis=1, keepdims=True)

    # Run the plotter
    plot_cross_attention_matrix(mock_attention, text_tokens, patch_labels)


if __name__ == "__main__":
    run_synthetic_demo()