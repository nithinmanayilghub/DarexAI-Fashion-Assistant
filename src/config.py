import os

# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths
DATA_DIR = os.path.join(BASE_DIR, "ML-TASK")
PRODUCTS_CSV = os.path.join(DATA_DIR, "products.csv")
OUTFITS_CSV = os.path.join(DATA_DIR, "outfits.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

# Embeddings Cache Path
EMBEDDINGS_CACHE_PATH = os.path.join(BASE_DIR, "src", "product_embeddings.pkl")

# Machine Learning Configurations
import torch
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"  # Auto-detect GPU acceleration
HYBRID_ALPHA = 0.05  # Weight given to visual image embeddings (relative to text description)

# Gemini API Model Configuration
GEMINI_MODEL_NAME = "gemini-1.5-flash"
