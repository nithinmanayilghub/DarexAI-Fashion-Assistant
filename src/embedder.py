import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import os
import threading
from src import config

# Global model and processor instances for lazy loading
_model = None
_processor = None
_lock = threading.RLock()

def get_model_and_processor():
    """Lazy loads and caches the CLIP model and processor to save startup time."""
    global _model, _processor
    if _model is None or _processor is None:
        with _lock:
            if _model is None or _processor is None:
                print(f"Loading CLIP model '{config.CLIP_MODEL_NAME}' onto device '{config.device}'...")
                _model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME).to(config.device)
                _processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
                _model.eval()  # Set model to evaluation mode
    return _model, _processor

def get_text_embeddings(texts):
    """Generates L2-normalized embeddings for a list of text descriptions."""
    model, processor = get_model_and_processor()
    
    # Ensure inputs is a list
    if isinstance(texts, str):
        texts = [texts]
        
    with torch.no_grad():
        inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(config.device)
        text_outputs = model.text_model(**inputs)
        pooled_output = text_outputs[1]
        text_features = model.text_projection(pooled_output)
        # Convert to numpy and L2 normalize
        feats = text_features.cpu().numpy()
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        # Avoid division by zero
        normalized_feats = feats / np.where(norms == 0, 1.0, norms)
        return normalized_feats

def get_image_embedding(image_path):
    """Generates L2-normalized embedding for a single product image."""
    model, processor = get_model_and_processor()
    
    if not os.path.exists(image_path):
        # Fallback if image doesn't exist
        print(f"Warning: Image path not found: {image_path}. Returning zero vector.")
        return np.zeros((1, 512), dtype=np.float32)
        
    try:
        image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(config.device)
            vision_outputs = model.vision_model(**inputs)
            pooled_output = vision_outputs[1]
            image_features = model.visual_projection(pooled_output)
            # Convert to numpy and L2 normalize
            feat = image_features.cpu().numpy()
            norm = np.linalg.norm(feat, axis=1, keepdims=True)
            normalized_feat = feat / np.where(norm == 0, 1.0, norm)
            return normalized_feat
    except Exception as e:
        print(f"Error embedding image {image_path}: {e}. Returning zero vector.")
        return np.zeros((1, 512), dtype=np.float32)
