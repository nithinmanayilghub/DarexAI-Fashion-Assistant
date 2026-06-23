import os
import sys
import math
import numpy as np
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Programmatically append project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, data_loader, assistant
from src.vector_store import VectorStore
from src.compatibility import CompatibilityEngine

app = FastAPI(
    title="DareXAI API Backend",
    description="Backend API for the AI Fashion Stylist & Recommendation Assistant",
    version="1.0.0"
)

# Enable CORS for the Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables loaded on startup
products_df = None
outfits_df = None
v_store = None
compat_engine = None

def sanitize_for_json(obj):
    """Recursively clean objects to ensure JSON compatibility (resolves float NaN issues)."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif isinstance(obj, np.generic):
        return sanitize_for_json(obj.item())
    elif isinstance(obj, pd.Series):
        return sanitize_for_json(obj.to_dict())
    return obj

@app.on_event("startup")
def startup_event():
    global products_df, outfits_df, v_store, compat_engine
    try:
        products_df = data_loader.load_products()
        outfits_df = data_loader.load_outfits()
        v_store = VectorStore(products_df)
        compat_engine = CompatibilityEngine(outfits_df)
    except Exception as e:
        print(f"Startup loading failed: {e}")

class ChatRequest(BaseModel):
    message: str
    gender_override: Optional[str] = None
    occasion_override: Optional[str] = None

@app.get("/health")
def health():
    if v_store is None or products_df is None:
        raise HTTPException(status_code=503, detail="System starting up or initialization failed")
    
    import torch
    return {
        "status": "healthy",
        "pytorch_device": "cuda" if torch.cuda.is_available() else "cpu",
        "embeddings_cached": os.path.exists(config.EMBEDDINGS_CACHE_PATH),
        "gemini_api_key_connected": bool(os.environ.get("GEMINI_API_KEY")),
        "total_products": len(products_df),
        "total_outfits": len(outfits_df)
    }

@app.post("/api/chat")
def chat(payload: ChatRequest):
    global products_df, outfits_df, v_store, compat_engine
    if v_store is None:
        raise HTTPException(status_code=503, detail="System starting up or initialization failed")
    
    user_msg = payload.message
    gender_input = payload.gender_override
    occasion_input = payload.occasion_override
    
    # 1. Parse user intent using LLM (or regex fallback)
    intent = assistant.parse_user_intent(user_msg)
    
    # Apply manual overrides if provided
    gender_filter = intent.get("gender")
    if gender_input and gender_input != "AI Auto":
        gender_filter = gender_input.lower()
        
    occasion_filter = intent.get("occasion")
    if occasion_input and occasion_input != "AI Auto":
        occasion_filter = occasion_input.lower()
        
    search_query = intent.get("query", user_msg)
    
    # Determine allowed categories for the hero item based on the user request
    msg_lower = user_msg.lower()
    footwear_kws = ["shoe", "sneaker", "jutti", "boot", "loafer", "sandal", "footwear", "flat", "heel", "slip-on"]
    accessory_kws = ["necklace", "earring", "watch", "clutch", "handbag", "sunglass", "cap"]
    
    is_searching_footwear = any(kw in msg_lower for kw in footwear_kws)
    is_searching_accessory = any(kw in msg_lower for kw in accessory_kws)
    
    if is_searching_footwear:
        hero_categories = compat_engine.category_groups['footwear']
    elif is_searching_accessory:
        hero_categories = compat_engine.category_groups['accessory']
    else:
        hero_categories = compat_engine.category_groups['topwear'] + compat_engine.category_groups['one_piece']
        
    hits = v_store.search(
        query_text=search_query,
        gender=gender_filter,
        categories=hero_categories,
        top_k=3
    )
    
    # If no targeted category hits found, search the whole database as a fallback
    if not hits:
        hits = v_store.search(
            query_text=search_query,
            gender=gender_filter,
            top_k=3
        )
        
    if not hits:
        # Fallback if catalog search yields nothing
        assistant_response = f"I searched our fashion catalog for *'{search_query}'* but couldn't find any direct matches. Could you try describing the style or clothing type in a different way? (e.g. 'linen shirt', 'saree', or 'formal trousers')"
        return sanitize_for_json({
            "content": assistant_response,
            "outfit": None
        })
        
    # Select the top matching product as the Hero item
    hero_product = hits[0]["product"]
    
    # Validate if it is a reasonable semantic match/alternative for what the user requested
    validation_res = assistant.validate_product_match(user_msg, hero_product)
    if isinstance(validation_res, tuple):
        is_match, explanation = validation_res
    else:
        is_match, explanation = validation_res, ""
        
    # 3. Use Compatibility Engine to construct the full outfit
    recommended_outfit = compat_engine.recommend_compatible_outfit(
        hero_product=hero_product,
        products_df=products_df,
        v_store=v_store
    )
    
    if not is_match:
        # Politely explain the unavailability and propose the fallback alternative
        if explanation:
            recommended_outfit['stylist_rationale'] = explanation
        else:
            recommended_outfit['stylist_rationale'] = f"I'm sorry, we don't have the exact item you requested in our catalog. However, I've curated a compatible alternative using {hero_product['name']} by {hero_product['brand']}!"
            
        assistant_response = f"I searched for your requested item, but it is not currently available in our catalog. Here is a stylish alternative combination I curated for you:"
    else:
        # Generate the personalized stylist explanation / rationale normally
        styled_rationale = assistant.generate_styling_rationale(user_msg, recommended_outfit)
        recommended_outfit['stylist_rationale'] = styled_rationale
        assistant_response = f"Based on your request, I've curated a styled outfit starting with a primary hero item: **{hero_product['name']}**."
        
    return sanitize_for_json({
        "content": assistant_response,
        "outfit": recommended_outfit
    })

@app.post("/api/rebuild-index")
def rebuild_index():
    global v_store
    if v_store is None:
        raise HTTPException(status_code=503, detail="System starting up or initialization failed")
    
    try:
        if os.path.exists(config.EMBEDDINGS_CACHE_PATH):
            os.remove(config.EMBEDDINGS_CACHE_PATH)
        v_store.build_index()
        return {"status": "success", "message": "Vector index successfully rebuilt and cached"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rebuild failed: {str(e)}")
