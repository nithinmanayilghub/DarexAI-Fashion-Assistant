import pytest
import numpy as np
from src import embedder

def test_text_embeddings_shape():
    # Test shape of single text query embedding
    emb = embedder.get_text_embeddings("classic white shirt")
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (1, 512)

def test_text_embeddings_multiple():
    # Test shape of multiple text queries
    embs = embedder.get_text_embeddings(["white shirt", "blue jeans"])
    assert isinstance(embs, np.ndarray)
    assert embs.shape == (2, 512)

def test_embeddings_normalization():
    emb = embedder.get_text_embeddings("sports running shoes")
    # L2 Norm should be exactly 1.0 (or very close due to float precision)
    norm = np.linalg.norm(emb, axis=1)
    assert np.allclose(norm, 1.0, atol=1e-5)

def test_image_embeddings_fallback():
    # Test that a non-existent image path fails gracefully and returns a zero vector
    emb = embedder.get_image_embedding("non_existent_image.jpg")
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (1, 512)
    assert np.all(emb == 0.0)
