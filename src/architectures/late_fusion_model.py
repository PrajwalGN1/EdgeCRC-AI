import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalFusionNetwork(nn.Module):
    def __init__(self, vision_dim=1024, text_dim=768, hidden_dim=512):
        super(MultimodalFusionNetwork, self).__init__()
        
        # ---------------------------------------------------------
        # 1. Vision Branch: Attention-Based Multiple Instance Learning
        # ---------------------------------------------------------
        # This network learns to score each patch from 0.0 to 1.0.
        # High score = "This patch looks like cancer"
        # Low score = "This patch looks like healthy tissue/background"
        self.vision_attention = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # ---------------------------------------------------------
        # 2. Text Branch: Semantic Refinement
        # ---------------------------------------------------------
        # We project the 768-D ClinicalBERT vector into the same 
        # dimensional space (512) to balance its weight against the images.
        self.text_projector = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # ---------------------------------------------------------
        # 3. Fusion & Classification Head
        # ---------------------------------------------------------
        # 1024 (Vision Summary) + 512 (Text Summary) = 1536
        fusion_dim = vision_dim + hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1) # Single output node for Binary Classification
        )

    def forward(self, vision_tensor, text_tensor):
        # vision_tensor shape: [1, N, 1024]
        # text_tensor shape: [1, 768]
        
        # Drop the artificial batch dimension for MIL processing
        H = vision_tensor.squeeze(0) # Shape becomes: [N, 1024]
        
        # --- AB-MIL Processing ---
        # Calculate raw attention scores for the N patches
        A_raw = self.vision_attention(H) # Shape: [N, 1]
        
        # Softmax ensures all patch attention scores sum up to exactly 1.0
        A = F.softmax(A_raw, dim=0) 
        
        # Matrix Multiplication: Multiply patches by their weights and sum them up
        # This collapses the variable [N, 1024] tensor into a fixed [1, 1024] patient summary
        vision_summary = torch.mm(A.t(), H) 
        
        # --- Text Processing ---
        text_summary = self.text_projector(text_tensor) # Shape: [1, 512]
        
        # --- Multimodal Fusion ---
        # Concatenate the vision and text summaries side-by-side
        fused_vector = torch.cat((vision_summary, text_summary), dim=1) # Shape: [1, 1536]
        
        # --- Final Prediction ---
        logits = self.classifier(fused_vector) # Shape: [1, 1]
        prob = torch.sigmoid(logits) # Squashes the output between 0.0 and 1.0
        
        # 🚨 XAI OUTPUT: We return both the probability AND the Attention weights (A)
        # We will need 'A' later to draw the Heatmaps over the top-scoring images.
        return prob, A

# --- Quick Diagnostic Test ---
if __name__ == "__main__":
    print("🚀 Initializing Multimodal Architecture Test...")
    model = MultimodalFusionNetwork()
    
    # Simulate the exact tensors our DataLoader just gave us
    dummy_vision = torch.randn(1, 1000, 1024)
    dummy_text = torch.randn(1, 768)
    
    prob, attention_weights = model(dummy_vision, dummy_text)
    
    print("\n--- Architecture Sanity Check ---")
    print(f"Output Probability Shape: {prob.shape} (Expected: [1, 1])")
    print(f"Output Attention Shape:   {attention_weights.shape} (Expected: [1000, 1])")
    print(f"Sample Prediction:        {prob.item():.4f} (Probability of MSI)")
