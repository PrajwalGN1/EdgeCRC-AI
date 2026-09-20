import torch
import torch.nn as nn

class SummaryToPatchCrossAttention(nn.Module):
    def __init__(self, vision_dim=1024, text_dim=768, num_heads=8):
        super().__init__()
        # 1. Project the 768-D text [CLS] token to match the 1024-D vision space
        self.text_projection = nn.Linear(text_dim, vision_dim)
        
        # 2. Cross Attention layer
        self.cross_attn = nn.MultiheadAttention(embed_dim=vision_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(vision_dim)
        
        # 3. Final Classification Head (Replaces the Late Fusion MLP)
        self.classifier = nn.Sequential(
            nn.Linear(vision_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()     # Sigmoid output for Binary MSI/MSS Classification
        )

    def forward(self, vision_seq, text_summary):
        """
        vision_seq: Shape (Batch, Num_Patches, 1024) - Uncompressed image patches (e.g., the 385 patches)
        text_summary: Shape (Batch, 768) - ClinicalBERT [CLS] vector
        """
        # Ensure text summary has a sequence dimension: (Batch, 1, 768)
        if len(text_summary.shape) == 2:
            text_summary = text_summary.unsqueeze(1)
            
        # Ensure vision has a batch dimension if it's missing: (1, Num_Patches, 1024)
        if len(vision_seq.shape) == 2:
            vision_seq = vision_seq.unsqueeze(0)
            
        # --- Step A: Project Text ---
        text_query = self.text_projection(text_summary) # Now shape (Batch, 1, 1024)
        
        # --- Step B: Cross Attention ---
        # The text query searches the WSI patches for relevant structural evidence
        attn_output, attn_weights = self.cross_attn(
            query=text_query, 
            key=vision_seq, 
            value=vision_seq
        )

        # (Assuming batch size is 1, so we take index 0, and detach from GPU/grad)
        real_matrix = attn_weights[0].detach().cpu().numpy()

        # --- Step C: Residual Connection & Normalization ---
        fused_features = self.layer_norm(text_query + attn_output) # Shape: (Batch, 1, 1024)
        
        # --- Step D: Squeeze and Classify ---
        pooled_features = fused_features.squeeze(1) # Shape: (Batch, 1024)
        probability = self.classifier(pooled_features)
        
        return probability, attn_weights
