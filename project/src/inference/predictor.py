import os
from typing import Dict, List

import torch
import yaml
from PIL import Image
from torchvision import transforms

from src.models.classifier import LandscapeClassifier


class LandscapePredictor:
    """Предиктор для инференса классификации ландшафтов"""
    
    def __init__(self, model_path: str, config_path: str = "configs/train_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = self.config['classes']
        
        self.model = LandscapeClassifier(
            num_classes=len(self.classes),
            backbone=self.config['model']['backbone'],
            pretrained=False
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((self.config['data']['img_size'], self.config['data']['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image: Image.Image) -> Dict:
        """Предсказать класс изображения"""
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        probs = probabilities.squeeze().cpu().numpy()
        predicted_idx = int(probs.argmax())
        
        return {
            "predicted_class": self.classes[predicted_idx],
            "confidence": float(probs[predicted_idx]),
            "probabilities": {
                cls: float(prob) for cls, prob in zip(self.classes, probs)
            }
        }
    
    def predict_batch(self, images: List[Image.Image]) -> List[Dict]:
        """Батчевое предсказание"""
        tensors = torch.stack([self.transform(img) for img in images]).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensors)
            probabilities = torch.softmax(outputs, dim=1)
        
        probs = probabilities.cpu().numpy()
        results = []
        for prob in probs:
            idx = int(prob.argmax())
            results.append({
                "predicted_class": self.classes[idx],
                "confidence": float(prob[idx]),
                "probabilities": {cls: float(p) for cls, p in zip(self.classes, prob)}
            })
        return results