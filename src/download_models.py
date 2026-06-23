import os
import sys

# Append project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import CLIPProcessor, CLIPModel
from src import config

def pre_download():
    print(f"Pre-downloading and caching CLIP model weights for '{config.CLIP_MODEL_NAME}'...")
    # This downloads the model and processor weights to the local cache directory
    CLIPModel.from_pretrained(config.CLIP_MODEL_NAME)
    CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
    print("CLIP model weights successfully cached.")

if __name__ == '__main__':
    pre_download()
