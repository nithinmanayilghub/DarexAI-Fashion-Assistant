import pytest
from src import data_loader
from src.vector_store import VectorStore
from src.compatibility import CompatibilityEngine

@pytest.fixture(scope="module")
def shared_resources():
    products_df = data_loader.load_products()
    outfits_df = data_loader.load_outfits()
    v_store = VectorStore(products_df)
    engine = CompatibilityEngine(outfits_df)
    return products_df, v_store, engine

def test_curated_companions_retrieval(shared_resources):
    products_df, _, engine = shared_resources
    
    # Bodycon dress (ajio_703182002) is a curated item in W1
    curated_outfit, companions = engine.get_curated_companions("ajio_703182002", products_df)
    assert curated_outfit is not None
    assert curated_outfit["outfit_id"] == "outfit W1"
    assert "footwear" in companions  # Heels
    assert "accessory_1" in companions  # Clutch

def test_recommend_compatible_outfit_curated(shared_resources):
    products_df, v_store, engine = shared_resources
    
    hero_product = products_df[products_df["id"] == "ajio_703182002"].iloc[0].to_dict()
    outfit = engine.recommend_compatible_outfit(hero_product, products_df, v_store)
    
    assert outfit["source"] == "Curated Outfit"
    assert outfit["outfit_id"] == "outfit W1"
    assert len(outfit["items"]) >= 2
    assert "hero" in outfit["items"]
    assert outfit["items"]["hero"]["id"] == "ajio_703182002"

def test_recommend_compatible_outfit_dynamic(shared_resources):
    products_df, v_store, engine = shared_resources
    
    # Grab a product that is not part of any curated outfits if any, or create a mock product
    # Actually all 68 products are in outfits. Let's temporarily change id to mock dynamic retrieval
    hero_product = products_df.iloc[0].to_dict().copy()
    hero_product["id"] = "mock_id_99999"  # Will trigger dynamic pairing
    hero_product["category"] = "formal-shirts"
    hero_product["gender"] = "men"
    hero_product["occasion"] = "office"
    
    outfit = engine.recommend_compatible_outfit(hero_product, products_df, v_store)
    assert outfit["source"] == "AI Recommended"
    assert "hero" in outfit["items"]
    assert "bottomwear" in outfit["items"] or "footwear" in outfit["items"]
