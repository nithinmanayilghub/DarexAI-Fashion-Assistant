import pytest
from src import assistant

def test_intent_parsing_fallback_gender():
    # Men's casual query
    intent = assistant.parse_user_intent_fallback("I am looking for a shirt for a young guy")
    assert intent["gender"] == "men"
    
    # Women's formal query
    intent = assistant.parse_user_intent_fallback("recommend a formal blazer for a business lady")
    assert intent["gender"] == "women"
    
    # Neutral/Unisex query
    intent = assistant.parse_user_intent_fallback("show me standard unisex activewear")
    assert intent["gender"] is None

def test_intent_parsing_fallback_occasion():
    # Office occasion
    intent = assistant.parse_user_intent_fallback("suggest a shirt for a corporate interview")
    assert intent["occasion"] == "office"
    
    # Wedding occasion
    intent = assistant.parse_user_intent_fallback("Sherwani options for attending a wedding next week")
    assert intent["occasion"] == "wedding"
    
    # Vacation occasion
    intent = assistant.parse_user_intent_fallback("beach holiday shorts and caps")
    assert intent["occasion"] == "vacation"

def test_intent_parsing_fallback_query_cleaning():
    intent = assistant.parse_user_intent_fallback("I need an outfit for a business meeting containing formal black trousers")
    # Verified query is stripped of intent phrases
    assert "i need" not in intent["query"].lower()
    assert "outfit for" not in intent["query"].lower()

def test_validate_product_match():
    # Test fallback validation when user asks for Dhoti but gets a Kurta
    product = {"name": "Roman Silk Embroidery Kurta"}
    res = assistant.validate_product_match("Kerala Traditional Dhothi", product)
    assert isinstance(res, tuple)
    is_match, explanation = res
    assert is_match is False
    assert "Dhoti" in explanation or "Kurta" in explanation
