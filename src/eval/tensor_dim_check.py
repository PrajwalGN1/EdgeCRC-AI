import torch

# Replace with the actual path to one of the patient files in your clean_tensors folder
# If they are .h5 files instead of .pt, let me know!
file_path = "clean_tensors/TCGA-3L-AA1B_vision.pt"

try:
    data = torch.load(file_path)
    data = torch.load(file_path)
    print("Successfully loaded file!")
    print("Object Type:", type(data))
    print("Tensor Shape:", data.shape)
    
    if "vision" in data:
        print("Vision Tensor Shape:", data["vision"].shape)
    if "text" in data:
        print("Text Tensor Shape:", data["text"].shape)
        
except Exception as e:
    print("Error loading file:", e)
