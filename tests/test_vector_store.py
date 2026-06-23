import pytest
from src import data_loader
from src.vector_store import VectorStore

@pytest.fixture(scope="module")
def shared_store():
    # Share a single store instance across tests to avoid rebuilding index multiple times
    products_df = data_loader.load_products()
    return VectorStore(products_df)

def test_vector_store_initialization(shared_store):
    assert len(shared_store.product_ids) == 68
    assert shared_store.hybrid_embeddings.shape == (68, 512)

def test_search_without_filters(shared_store):
    hits = shared_store.search("formal shirt", top_k=5)
    assert len(hits) == 5
    for hit in hits:
        assert "product" in hit
        assert "score" in hit
        assert isinstance(hit["score"], float)

def test_search_with_gender_filter(shared_store):
    # Test strict 'men' filter
    hits = shared_store.search("shoes", gender="men", top_k=3)
    for hit in hits:
        assert hit["product"]["gender"] in ["men", "unisex"]

    # Test strict 'women' filter
    hits = shared_store.search("bag", gender="women", top_k=3)
    for hit in hits:
        assert hit["product"]["gender"] in ["women", "unisex"]

def test_search_with_category_filter(shared_store):
    # Retrieve only heels
    hits = shared_store.search("footwear", categories=["heels"], top_k=3)
    for hit in hits:
        assert hit["product"]["category"] == "heels"

    # Retrieve only bottomwear (trousers or jeans)
    hits = shared_store.search("blue denim", categories=["trousers", "jeans"], top_k=3)
    for hit in hits:
        assert hit["product"]["category"] in ["trousers", "jeans"]
