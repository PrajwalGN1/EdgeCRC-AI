import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import random

class MultimodalTCGADataset(Dataset):
    def __init__(self, registry_path, tensor_dir, max_patches=1000):
        """
        Loads the Vision and Text tensors dynamically.
        Applies a random sampling cap if patch count exceeds max_patches.
        """
        self.tensor_dir = tensor_dir
        self.max_patches = max_patches
        
        # Load the Data Contract
        with open(registry_path, 'r') as f:
            full_registry = json.load(f)
            
        # Verify that the tensors actually exist on disk before adding to our active list
        # self.valid_patients = []
        # for patient_id, data in full_registry.items():
        #     vision_path = os.path.join(tensor_dir, f"{patient_id}_vision.pt")
        #     text_path = os.path.join(tensor_dir, f"{patient_id}_text.pt")
        #
        #     if os.path.exists(vision_path) and os.path.exists(text_path):
        #
        #         # --- THE FIX: INTERCEPT AND FLIP THE LABEL ---
        #         raw_label = data["label"]
        #
        #         # If the JSON saved them as strings:
        #         if isinstance(raw_label, str):
        #             correct_label = 1 if "MSI" in raw_label.upper() else 0
        #         # If the JSON saved them as ints (0 for MSI, 1 for MSS), invert them mathematically:
        #         else:
        #             correct_label = 1 - raw_label
        #         # ---------------------------------------------
        #
        #         self.valid_patients.append({
        #             "id": patient_id,
        #             "label": correct_label,  # NOW FORCED: 1 for MSI, 0 for MSS
        #             "vision_path": vision_path,
        #             "text_path": text_path
        #         })
        # Verify that the tensors actually exist on disk before adding to our active list
        self.valid_patients = []
        for patient_id, data in full_registry.items():
            vision_path = os.path.join(tensor_dir, f"{patient_id}_vision.pt")
            text_path = os.path.join(tensor_dir, f"{patient_id}_text.pt")

            if os.path.exists(vision_path) and os.path.exists(text_path):
                self.valid_patients.append({
                    "id": patient_id,
                    "label": data["label"],  # 1 for MSI, 0 for MSS
                    "vision_path": vision_path,
                    "text_path": text_path
                })
                
        print(f"✅ Verified {len(self.valid_patients)} multimodal patients ready for training.")

    def __len__(self):
        return len(self.valid_patients)

    def __getitem__(self, idx):
        patient = self.valid_patients[idx]
        
        # 1. Load the pre-extracted tensors
        # vision_tensor shape: [N, 1024]
        # text_tensor shape: [1, 768]
        vision_tensor = torch.load(patient["vision_path"], weights_only=True)
        text_tensor = torch.load(patient["text_path"], weights_only=True)
        
        # 2. The Dynamic Cap (No Padding!)
        num_patches = vision_tensor.size(0)
        
        if num_patches > self.max_patches:
            # Randomly sample 'max_patches' indices to prevent bias
            # This ensures we get a diverse representation of the tumor
            indices = torch.randperm(num_patches)[:self.max_patches]
            vision_tensor = vision_tensor[indices]
            
        # 3. Format the Label
        label = torch.tensor(patient["label"], dtype=torch.float32)
        
        return {
            "patient_id": patient["id"],
            "vision": vision_tensor,
            "text": text_tensor.squeeze(0), # Remove extra dimension: becomes [768]
            "label": label
        }

# --- Quick Diagnostic Test ---
if __name__ == "__main__":
    # Update these paths to where you saved the Kaggle downloads on your Ubuntu machine
    REGISTRY = "../data/patient_registry.json"
    TENSORS = "../data/clean_tensors" # Directory containing both vision and text .pt files
    
    dataset = MultimodalTCGADataset(REGISTRY, TENSORS, max_patches=1000)
    
    # We use batch_size=1 so PyTorch doesn't crash trying to stack variable N sizes
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Test pull one patient
    for batch in dataloader:
        print("\n--- DataLoader Sanity Check ---")
        print(f"Patient ID:   {batch['patient_id'][0]}")
        print(f"Vision Shape: {batch['vision'].shape}  <- [Batch=1, N_Patches, 1024]")
        print(f"Text Shape:   {batch['text'].shape}    <- [Batch=1, 768]")
        print(f"Label:        {batch['label'].item()} (1=MSI, 0=MSS)")
        break
