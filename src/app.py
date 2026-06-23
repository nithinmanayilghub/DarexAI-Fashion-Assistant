import streamlit as st
import os
import pandas as pd
from PIL import Image

# Import local modules
from src import config, data_loader, assistant
from src.vector_store import VectorStore
from src.compatibility import CompatibilityEngine

# Page configuration with a stylish title and icon
st.set_page_config(
    page_title="Dare XAI | Fashion Assistant",
    page_icon="✨",
    layout="wide"
)

# Custom Premium Styling (CSS)
st.markdown("""
<style>
    /* Global styles */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header container styling */
    .header-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #311042 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .header-container::after {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 80%);
        pointer-events: none;
    }
    
    .header-title {
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
        background: linear-gradient(to right, #a5f3fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    /* Card design for outfits */
    .outfit-card {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border: 1px solid #f1f5f9;
        transition: transform 0.2s ease-in-out;
    }
    .outfit-card:hover {
        transform: translateY(-2px);
    }
    
    /* Stylist rationale quote box */
    .rationale-box {
        background-color: #f8fafc;
        border-left: 5px solid #8b5cf6;
        padding: 1rem 1.25rem;
        border-radius: 0 12px 12px 0;
        font-style: italic;
        color: #334155;
        margin-bottom: 1.5rem;
        font-size: 1.05rem;
    }
    
    /* Product card style */
    .product-card {
        border-radius: 12px;
        padding: 0.75rem;
        background: #fdfdfd;
        border: 1px solid #e2e8f0;
        text-align: center;
        height: 100%;
    }
    
    .product-title {
        font-weight: 600;
        font-size: 0.95rem;
        color: #1e293b;
        margin: 0.5rem 0 0.25rem 0;
        min-height: 2.4rem;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    
    .product-meta {
        font-size: 0.8rem;
        color: #64748b;
        margin-bottom: 0.5rem;
    }
    
    .product-price {
        font-weight: 800;
        color: #0f172a;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .product-rating {
        color: #f59e0b;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
    }
    
    .buy-btn {
        display: inline-block;
        background-color: #4f46e5;
        color: white !important;
        text-decoration: none !important;
        padding: 0.4rem 1rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        transition: background-color 0.2s;
    }
    .buy-btn:hover {
        background-color: #4338ca;
    }
    
    /* Chat message styling */
    .chat-user {
        background-color: #eff6ff;
        border-radius: 16px 16px 0 16px;
        padding: 1rem;
        margin-bottom: 1rem;
        max-width: 80%;
        margin-left: auto;
        border: 1px solid #dbeafe;
    }
    
    .chat-assistant {
        background-color: #f8fafc;
        border-radius: 16px 16px 16px 0;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
<div class="header-container">
    <div class="header-title">✨ DARE XAI FASHION ASSISTANT</div>
    <div class="header-subtitle">Your AI-Powered Stylist using CLIP Embeddings & Gemini Intelligence</div>
</div>
""", unsafe_allow_html=True)

# Load data models
@st.cache_resource
def get_resources():
    products_df = data_loader.load_products()
    outfits_df = data_loader.load_outfits()
    # Initialize the Vector store
    v_store = VectorStore(products_df)
    # Initialize Compatibility Engine
    compat_engine = CompatibilityEngine(outfits_df)
    return products_df, outfits_df, v_store, compat_engine

try:
    products_df, outfits_df, v_store, compat_engine = get_resources()
except Exception as e:
    st.error(f"Error loading system resources: {e}")
    st.info("Make sure the dataset is placed under the 'ML-TASK/' folder.")
    st.stop()

# Sidebar: Controls and Status
with st.sidebar:
    st.markdown("### 🛠️ Personalize Stylist Filters")
    st.write("Force specific options or leave them as 'AI Auto' to let the LLM decide from your chat.")
    
    gender_input = st.selectbox("Target Gender", ["AI Auto", "Men", "Women", "Unisex"])
    occasion_input = st.selectbox("Target Occasion", ["AI Auto", "Casual", "Office", "Party", "Wedding", "Festive", "Sports", "Vacation", "Winter"])
    
    st.markdown("---")
    st.markdown("### 📊 Dataset Overview")
    st.metric("Total Products", len(products_df))
    st.metric("Curated Outfits", len(outfits_df))
    
    st.markdown("---")
    st.markdown("### ⚙️ System Settings")
    if st.button("🔄 Rebuild Vector Index", help="Re-generate and cache CLIP embeddings for all catalog items"):
        with st.spinner("Generating embeddings (this may take up to a minute)..."):
            if os.path.exists(config.EMBEDDINGS_CACHE_PATH):
                os.remove(config.EMBEDDINGS_CACHE_PATH)
            # Re-initialize to trigger build
            v_store.build_index()
            st.success("Vector index successfully rebuilt and cached!")
            st.rerun()
            
    # Display API Key status
    api_key_status = "Connected ✅" if os.environ.get("GEMINI_API_KEY") else "Offline (Rule Fallback) ⚠️"
    st.write(f"**Gemini Status:** {api_key_status}")

# Initialize chat session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your personal fashion stylist. Tell me what occasion or style you are shopping for (e.g., *'I need a smart casual outfit for a business meeting'* or *'Suggest a summer vacation look'*), and I will build you a complete outfit!", "outfit": None}
    ]

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-user">💬 <b>You:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-assistant">🧑‍🎨 <b>Stylist Assistant:</b>', unsafe_allow_html=True)
        # Show text content
        st.markdown(msg["content"])
        
        # If there's an outfit recommendation in the message, render it beautifully
        if msg.get("outfit"):
            outfit = msg["outfit"]
            
            # Show Stylist Rationale Quote
            st.markdown(f'<div class="rationale-box">💬 <b>Stylist Rationale:</b><br>"{outfit["stylist_rationale"]}"</div>', unsafe_allow_html=True)
            
            st.markdown("##### 👔 Recommended Outfit Combination:")
            
            # Render items side-by-side
            items = outfit["items"]
            cols = st.columns(len(items))
            
            for i, (role, prod) in enumerate(items.items()):
                with cols[i]:
                    # Render product card
                    img_path = prod.get("absolute_image_path")
                    
                    # Display product image
                    if img_path and os.path.exists(img_path):
                        try:
                            pil_img = Image.open(img_path)
                            st.image(pil_img, use_container_width=True)
                        except Exception:
                            st.image("https://via.placeholder.com/200x250.png?text=No+Image", use_container_width=True)
                    else:
                        st.image("https://via.placeholder.com/200x250.png?text=No+Image", use_container_width=True)
                    
                    rating_stars = "⭐" * int(round(prod.get("rating", 3.5)))
                    
                    st.markdown(f"""
                    <div class="product-card">
                        <div style="background-color: #e0e7ff; color: #4338ca; font-size: 0.75rem; font-weight: bold; padding: 0.2rem 0.5rem; border-radius: 4px; display: inline-block; margin-bottom: 0.3rem;">
                            {role.replace('_', ' ').upper()}
                        </div>
                        <div class="product-title">{prod.get('name')}</div>
                        <div class="product-meta">Brand: <b>{prod.get('brand')}</b> | Cat: {prod.get('category_label')}</div>
                        <div class="product-rating">{rating_stars} ({int(prod.get('rating_count', 0))})</div>
                        <div class="product-price">₹{int(prod.get('price_inr')):,}</div>
                        <a class="buy-btn" href="{prod.get('product_url', '#')}" target="_blank">View Product</a>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show outfit pricing summary
            total_price = sum(int(prod.get('price_inr', 0)) for prod in items.values())
            st.markdown(f"**Total Outfit Price:** `₹{total_price:,}` | **Style Source:** *{outfit.get('source', 'AI Recommendation')} ({outfit.get('outfit_id', 'Gen')})*")
            
        st.markdown('</div>', unsafe_allow_html=True)

# User Chat Input
if user_input := st.chat_input("Ask for style advice..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input, "outfit": None})
    st.rerun()

# Process the latest message
if st.session_state.messages[-1]["role"] == "user":
    user_msg = st.session_state.messages[-1]["content"]
    
    with st.spinner("Analyzing request and styling outfit..."):
        # 1. Parse user intent using LLM (or regex fallback)
        intent = assistant.parse_user_intent(user_msg)
        
        # Override filters if set in Sidebar
        gender_filter = intent.get("gender")
        if gender_input != "AI Auto":
            gender_filter = gender_input.lower()
            
        occasion_filter = intent.get("occasion")
        if occasion_input != "AI Auto":
            occasion_filter = occasion_input.lower()
            
        search_query = intent.get("query", user_msg)
        
        # 2. Search for the best "Hero" item using the Vector Store matching query & filters
        st.write(f"*Searching catalog for primary item matching: '{search_query}' (Gender: {gender_filter or 'Auto'}, Occasion: {occasion_filter or 'Auto'})...*")
        
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
            # Default: Prioritize apparel first
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
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_response,
                "outfit": None
            })
        else:
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
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_response,
                "outfit": recommended_outfit
            })
            
    st.rerun()
