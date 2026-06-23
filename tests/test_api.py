import pytest
from fastapi.testclient import TestClient
from src.main import app, startup_event

# Initialize startup events to load dataset/store
startup_event()

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "pytorch_device" in data
    assert "embeddings_cached" in data
    assert data["total_products"] == 68

def test_chat_recommendation_endpoint():
    payload = {
        "message": "I need a formal white shirt",
        "gender_override": "Men",
        "occasion_override": "Office"
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "content" in data
    assert "outfit" in data
    
    outfit = data["outfit"]
    assert outfit is not None
    assert "items" in outfit
    assert len(outfit["items"]) >= 2
    assert "hero" in outfit["items"]

def test_chat_no_match_endpoint():
    # Query something completely random that doesn't match any item
    payload = {
        "message": "xyzunknownproductabc",
        "gender_override": "Unisex",
        "occasion_override": "Casual"
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "content" in data
    # Fallback when no direct catalog matches are found
    assert "couldn't find any direct matches" in data["content"] or "outfit" in data

def test_rebuild_index_endpoint():
    response = client.post("/api/rebuild-index")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
