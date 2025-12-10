from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

class CLIPEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        print(f"Loading CLIP model: {model_name}...")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model loaded on {self.device}.")

    def get_image_embedding(self, image_path: str):
        image = Image.open(image_path)
        inputs = self.processor(text=None, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        
        # Normalize the embedding
        embedding = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        return embedding.cpu().numpy()

    def get_text_embedding(self, text: str):
        inputs = self.processor(text=[text], images=None, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            
        # Normalize the embedding
        embedding = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        return embedding.cpu().numpy()
