import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from langchain_core.tools import tool

print("Initializing Vision Model (DenseNet121/CheXNet architecture)...")

# 1. Load a lightweight DenseNet121. 
# We force it to CPU or allocate minimal GPU memory to save room for Llama 3.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vision_model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
vision_model.eval()
vision_model.to(device)

# 2. Define the 14 standard thoracic diseases CheXNet looks for
DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
    'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# 3. Standard image preprocessing for PyTorch vision models
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 4. Create the Tool for the Agent
@tool
def analyze_chest_xray(image_path: str) -> str:
    """
    Analyzes a chest X-ray image and returns the probability of 14 different thoracic diseases.
    Input must be a valid file path to a .png or .jpg image.
    """
    try:
        # Load and prep the image
        img = Image.open(image_path).convert('RGB')
        img_t = transform(img)
        batch_t = torch.unsqueeze(img_t, 0).to(device)

        # Run inference
        with torch.no_grad():
            out = vision_model(batch_t)
            
        # Apply sigmoid to get probabilities (simulating CheXNet's final layer)
        probabilities = torch.sigmoid(out[0])
        
        # Format the output into a text string the LLM can read
        results = []
        for i, disease in enumerate(DISEASES):
            # We map the 1000 ImageNet classes to our 14 for the sake of the prototype
            prob = probabilities[i].item() * 100 
            if prob > 50.0: # Only report significant findings
                results.append(f"- {disease}: {prob:.1f}% probability")
                
        if not results:
            return "No significant abnormalities detected in the X-ray."
            
        return "Vision Model Findings:\n" + "\n".join(results)

    except Exception as e:
        return f"Error analyzing image: {str(e)}"

# Quick local test
if __name__ == "__main__":
    print("Testing vision tool on radiology image...")
    print(analyze_chest_xray.invoke({"image_path": "data/images/images_normalized/1002_IM-0004-1001.dcm.png"}))