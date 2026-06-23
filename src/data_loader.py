import pandas as pd
import os
from src import config

def clean_tags(tag_str):
    if pd.isna(tag_str) or not isinstance(tag_str, str):
        return []
    return [t.strip().lower() for t in tag_str.split(";") if t.strip()]

def load_products():
    """Loads the products metadata and cleans text/path fields."""
    if not os.path.exists(config.PRODUCTS_CSV):
        raise FileNotFoundError(f"Products CSV not found at {config.PRODUCTS_CSV}")
    
    df = pd.read_csv(config.PRODUCTS_CSV)
    
    # Fill missing values
    df['rating'] = df['rating'].fillna(3.5)
    df['rating_count'] = df['rating_count'].fillna(0)
    df['description'] = df['description'].fillna("")
    df['tags_list'] = df['tags'].apply(clean_tags)
    df['brand'] = df['brand'].fillna("Unknown Brand")
    
    # Clean image paths to make them absolute/relative to workspace root correctly
    def fix_image_path(path):
        if pd.isna(path):
            return None
        # ML-TASK has images/... we need to map to config.DATA_DIR/images/...
        # Let's clean up any double slashes and convert to absolute paths
        cleaned = path.strip().replace("\\", "/")
        if cleaned.startswith("images/"):
            return os.path.abspath(os.path.join(config.DATA_DIR, cleaned))
        return os.path.abspath(os.path.join(config.DATA_DIR, "images", cleaned))
        
    df['absolute_image_path'] = df['image'].apply(fix_image_path)
    return df

def load_outfits():
    """Loads the curated outfit mappings."""
    if not os.path.exists(config.OUTFITS_CSV):
        raise FileNotFoundError(f"Outfits CSV not found at {config.OUTFITS_CSV}")
    
    df = pd.read_csv(config.OUTFITS_CSV)
    df['stylist_rationale'] = df['stylist_rationale'].fillna("No rationale provided.")
    df['palette'] = df['palette'].fillna("neutral")
    return df

def get_product_by_id(products_df, product_id):
    """Retrieves a single product details by its unique ID."""
    match = products_df[products_df['id'] == product_id]
    if len(match) > 0:
        return match.iloc[0].to_dict()
    return None
