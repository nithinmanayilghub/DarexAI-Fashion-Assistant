import os
import json
import re
from src import config

# Try to initialize the Gemini API
has_gemini = False
try:
    import google.generativeai as genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        has_gemini = True
        print("Gemini API configured successfully.")
    else:
        print("Warning: GEMINI_API_KEY environment variable not found. Running in offline/fallback mode.")
except ImportError:
    print("Warning: google-generativeai package not found. Running in offline/fallback mode.")

def parse_user_intent_fallback(text):
    """
    Regex and keyword-based fallback parser to extract gender, occasion, and query
    if the Gemini API key is missing.
    """
    text_lower = text.lower()
    # Extract clean list of words
    words = re.findall(r"\b\w+\b", text_lower)
    
    # 1. Detect Gender
    gender = None
    if any(w in words for w in ["men", "man", "male", "guy", "boy", "gentleman", "gentlemen"]):
        gender = "men"
    elif any(w in words for w in ["women", "woman", "female", "lady", "girl", "ladies"]):
        gender = "women"
        
    # 2. Detect Occasion
    # Occasions in products.csv: office, wedding, casual, sports, vacation, festive, party, winter
    occasion = None
    if any(w in words for w in ["office", "work", "meeting", "formal", "interview", "corporate"]):
        occasion = "office"
    elif any(w in words for w in ["wedding", "ceremony", "sherwani", "marriage"]):
        occasion = "wedding"
    elif any(w in words for w in ["casual", "daily", "everyday", "streetwear", "home"]):
        occasion = "casual"
    elif any(w in words for w in ["sports", "workout", "gym", "running", "activewear", "hike"]):
        occasion = "sports"
    elif any(w in words for w in ["vacation", "beach", "summer", "holiday", "travel"]):
        occasion = "vacation"
    elif any(w in words for w in ["festive", "diwali", "kurta", "celebration"]):
        occasion = "festive"
    elif any(w in words for w in ["party", "night out", "clubbing", "cocktail", "dinner", "date"]):
        occasion = "party"
    elif any(w in words for w in ["winter", "cold", "jacket", "coat", "sweater"]):
        occasion = "winter"
        
    # 3. Clean search query
    query = text
    # Strip common phrases
    query = re.sub(r"\b(i want|i need|suggest|recommend|looking for|outfit for|show me)\b", "", query, flags=re.IGNORECASE)
    query = query.strip()
    
    return {
        "gender": gender,
        "occasion": occasion,
        "query": query if query else "stylish outfit"
    }

def parse_user_intent(user_message):
    """
    Parses a natural language message using Gemini LLM to extract structured fields.
    Falls back to a keyword-based rule parser if Gemini is unavailable.
    """
    if not has_gemini:
        return parse_user_intent_fallback(user_message)
        
    prompt = f"""
    You are an expert fashion stylist. Read the user request and extract key search filters.
    Return ONLY a valid JSON object with the following keys, no markdown wrapper, no extra text:
    - "gender": string (either "men", "women", or null)
    - "occasion": string (choose one of: "office", "wedding", "casual", "sports", "vacation", "festive", "party", "winter", or null)
    - "query": string (a short 2-3 word query describing the main fashion product requested)

    User Request: "{user_message}"
    JSON Output:
    """
    
    try:
        model = genai.GenerativeModel(config.GEMINI_MODEL_NAME)
        response = model.generate_content(prompt)
        text = response.text.strip()
        # Clean markdown if generated
        if text.startswith("```json"):
            text = text.split("```json")[1].split("```")[0].strip()
        elif text.startswith("```"):
            text = text.split("```")[1].split("```")[0].strip()
            
        data = json.loads(text)
        return data
    except Exception as e:
        print(f"Error calling Gemini for intent parsing: {e}. Falling back to rule-based parser.")
        return parse_user_intent_fallback(user_message)

def generate_styling_rationale(user_message, outfit):
    """
    Generates a personalized styling explanation using Gemini based on the selected outfit items.
    """
    if not has_gemini:
        # Fallback uses the pre-styled/rule-styled rationale
        return outfit['stylist_rationale']
        
    # Build list of items
    items_desc = []
    for role, prod in outfit['items'].items():
        items_desc.append(f"- {role.capitalize()}: {prod['name']} by {prod['brand']} ({prod['category_label']}) - {prod['description']}")
        
    prompt = f"""
    You are an expert personal fashion stylist. 
    A user asked for: "{user_message}"
    
    We selected the following outfit:
    Theme: {outfit['theme']}
    Palette: {outfit['palette']}
    Items:
    {chr(10).join(items_desc)}
    
    Write a brief, friendly, and convincing stylist rationale (2-3 sentences) explaining why this outfit is compatible, matches their occasion, and fits well together. Keep it professional and sleek. Do not mention "as requested" or system details.
    """
    
    try:
        model = genai.GenerativeModel(config.GEMINI_MODEL_NAME)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error calling Gemini for rationale: {e}. Returning default.")
        return outfit['stylist_rationale']

def validate_product_match(user_message, product):
    """
    Checks if the retrieved product is actually a valid semantic match for what the user requested,
    or if it's a completely different item that was pulled because it was the closest vector.
    """
    if not has_gemini:
        # Fallback in offline mode: if the user explicitly requested "dhoti" or "dhothi",
        # and the matching product name does not contain "dhoti" or "dhothi", we say it's not a match.
        msg_lower = user_message.lower()
        if "dhoti" in msg_lower or "dhothi" in msg_lower:
            prod_name = product["name"].lower()
            if "dhoti" not in prod_name and "dhothi" not in prod_name:
                return False, "I'm sorry, we don't have a traditional Kerala Dhoti in our current catalog. However, I can suggest a traditional off-white Roman Silk Embroidered Kurta set as a perfect festive alternative!"
        return True, ""

    prompt = f"""
    A user requested a fashion item: "{user_message}"
    The closest matching item in our database is: "{product['name']}" (Category: {product['category_label']})
    
    Is this product a reasonable semantic match, synonym, or close styling alternative for what the user requested?
    Respond with ONLY a JSON object:
    {{
       "is_match": true or false,
       "stylist_explanation": "A polite explanation if is_match is false, explaining that the exact item is not available, but suggesting this item as a great alternative."
    }}
    """
    try:
        model = genai.GenerativeModel(config.GEMINI_MODEL_NAME)
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```json"):
            text = text.split("```json")[1].split("```")[0].strip()
        elif text.startswith("```"):
            text = text.split("```")[1].split("```")[0].strip()
            
        data = json.loads(text)
        return data.get("is_match", True), data.get("stylist_explanation", "")
    except Exception as e:
        print(f"Error in validate_product_match: {e}")
        # Rule-based fallback
        msg_lower = user_message.lower()
        if "dhoti" in msg_lower or "dhothi" in msg_lower:
            prod_name = product["name"].lower()
            if "dhoti" not in prod_name and "dhothi" not in prod_name:
                return False, "I'm sorry, we don't have a traditional Kerala Dhoti in our current catalog. However, I can suggest a traditional off-white Roman Silk Embroidered Kurta set as a perfect festive alternative!"
        return True, ""
