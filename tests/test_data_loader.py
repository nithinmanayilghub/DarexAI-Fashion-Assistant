import pytest
import pandas as pd
from src import data_loader

def test_clean_tags():
    # Test with normal strings
    assert data_loader.clean_tags("men;casual;shirt") == ["men", "casual", "shirt"]
    # Test with extra whitespace and caps
    assert data_loader.clean_tags("  Men ; CASUAL ; shirt  ") == ["men", "casual", "shirt"]
    # Test with empty strings and NaNs
    assert data_loader.clean_tags("") == []
    assert data_loader.clean_tags(None) == []

def test_load_products():
    products_df = data_loader.load_products()
    assert isinstance(products_df, pd.DataFrame)
    assert len(products_df) == 68
    
    # Check important columns are present
    required_cols = ["id", "name", "brand", "price_inr", "rating", "gender", "category", "image", "absolute_image_path", "tags_list"]
    for col in required_cols:
        assert col in products_df.columns
        
    # Check that missing ratings are filled with fallback
    assert products_df["rating"].isnull().sum() == 0

def test_load_outfits():
    outfits_df = data_loader.load_outfits()
    assert isinstance(outfits_df, pd.DataFrame)
    assert len(outfits_df) == 25
    
    # Check styling columns are present
    assert "outfit_id" in outfits_df.columns
    assert "hero_id" in outfits_df.columns
    assert "footwear_id" in outfits_df.columns
    assert "stylist_rationale" in outfits_df.columns
