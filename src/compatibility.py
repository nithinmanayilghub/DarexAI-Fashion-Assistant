import pandas as pd
from src import data_loader, vector_store

class CompatibilityEngine:
    def __init__(self, outfits_df=None):
        self.outfits_df = outfits_df if outfits_df is not None else data_loader.load_outfits()
        self.category_groups = {
            'topwear': ['formal-shirts', 'casual-shirts', 'party-shirts', 'tshirts', 'polo-tshirts', 'sweatshirts', 'tops', 'activewear', 'sweaters', 'linen-shirts'],
            'bottomwear': ['trousers', 'jeans', 'chinos', 'shorts', 'track-pants', 'skirts', 'leggings'],
            'one_piece': ['suits', 'sherwanis', 'party-dresses', 'wedding-sarees', 'sharara-sets', 'casual-dresses', 'maxi-dresses', 'co-ord-sets', 'salwar-suits', 'kurta-sets'],
            'footwear': ['running-shoes', 'sneakers', 'ethnic-footwear', 'heels', 'boots', 'flats', 'formal-shoes', 'loafers', 'sandals'],
            'layer': ['nehru-jackets', 'denim-jackets', 'long-coats', 'blazers'],
            'accessory': ['necklaces', 'clutches', 'handbags', 'earrings', 'sunglasses', 'watches', 'caps']
        }

    def get_category_group(self, category):
        """Identifies which styling group a category belongs to."""
        cat_lower = str(category).lower().strip()
        for group, cats in self.category_groups.items():
            if cat_lower in cats:
                return group
        return 'accessory'  # Default fallback

    def get_curated_companions(self, product_id, products_df):
        """
        Checks if the product is in any expert curated outfits.
        Returns the outfit row and a list of companion product details if found.
        """
        for _, row in self.outfits_df.iterrows():
            item_ids = [
                row.get('hero_id'), row.get('second_id'), row.get('layer_id'),
                row.get('footwear_id'), row.get('accessory_1_id'), row.get('accessory_2_id')
            ]
            # Remove NaNs and filter
            item_ids = [idx for idx in item_ids if isinstance(idx, str) and idx == product_id]
            if item_ids:
                # Found the outfit! Let's retrieve all companion items
                companions = {}
                roles = ['hero', 'second', 'layer', 'footwear', 'accessory_1', 'accessory_2']
                for role in roles:
                    role_id_col = f"{role}_id"
                    role_id = row.get(role_id_col)
                    if isinstance(role_id, str) and role_id != product_id:
                        prod = data_loader.get_product_by_id(products_df, role_id)
                        if prod:
                            companions[role] = prod
                return row.to_dict(), companions
        return None, None

    def recommend_compatible_outfit(self, hero_product, products_df, v_store):
        """
        Main function to assemble a complete outfit around a given hero product.
        """
        hero_id = hero_product['id']
        hero_gender = hero_product['gender']
        hero_occasion = hero_product['occasion']
        hero_wear_type = hero_product['wear_type']
        hero_category = hero_product['category']
        
        # 1. First check if it is part of a pre-styled outfit
        curated_outfit, companions = self.get_curated_companions(hero_id, products_df)
        if curated_outfit:
            # We found an exact stylist curated outfit
            # Identify the role the user's selected product plays in this curated outfit
            user_role = 'hero'
            for r in ['hero', 'second', 'layer', 'footwear', 'accessory_1', 'accessory_2']:
                if curated_outfit.get(f"{r}_id") == hero_id:
                    user_role = r
                    break
                    
            outfit = {
                'source': 'Curated Outfit',
                'outfit_id': curated_outfit['outfit_id'],
                'theme': curated_outfit['theme'],
                'palette': curated_outfit['palette'],
                'stylist_rationale': curated_outfit['stylist_rationale'],
                'items': {
                    user_role: hero_product
                }
            }
            # Add companions without overwriting the user's product
            for role, prod in companions.items():
                outfit['items'][role] = prod
            return outfit

        # 2. Hybrid Matching: Build an outfit dynamically using category pairs & CLIP similarity
        hero_group = self.get_category_group(hero_category)
        
        # Determine what other categories we need to search for
        needed_groups = []
        if hero_group == 'one_piece':
            needed_groups = ['footwear', 'accessory']
        elif hero_group == 'topwear':
            needed_groups = ['bottomwear', 'footwear', 'accessory']
        elif hero_group == 'bottomwear':
            needed_groups = ['topwear', 'footwear', 'accessory']
        elif hero_group == 'footwear':
            needed_groups = ['topwear', 'bottomwear', 'accessory']
        else:  # accessory or layer
            needed_groups = ['topwear', 'bottomwear', 'footwear']
            
        outfit_items = {'hero': hero_product}
        
        # For each needed group, find the best matching item in the catalog
        # Search query is formed by combining the hero's features (name, occasion, style)
        search_query = f"{hero_product['name']} style {hero_occasion} {hero_wear_type}"
        
        for group in needed_groups:
            allowed_categories = self.category_groups[group]
            # Query Vector Store with filters
            hits = v_store.search(
                query_text=search_query,
                gender=hero_gender,
                categories=allowed_categories,
                top_k=3
            )
            
            if hits:
                # Take the highest ranking hit that isn't the hero itself
                best_hit = None
                for hit in hits:
                    if hit['product']['id'] != hero_id:
                        best_hit = hit['product']
                        break
                if best_hit:
                    outfit_items[group] = best_hit

        # Construct dynamic outfit result
        # Try to find a style rationale template or default styling rule
        palette_match = "harmonious colors"
        rationale = f"This dynamic outfit was generated matching {hero_product['name']} with compatible {', '.join(needed_groups)} suitable for a {hero_occasion} style."
        
        # Check if there is an outfit matching wear_type and occasion to steal rationale context
        similar_outfits = self.outfits_df[
            (self.outfits_df['wear_type'] == hero_wear_type) & 
            (self.outfits_df['occasion'] == hero_occasion)
        ]
        if len(similar_outfits) > 0:
            sample = similar_outfits.iloc[0]
            palette_match = sample['palette']
            rationale = f"Inspired by the curated '{sample['theme']}' style: " + sample['stylist_rationale']

        outfit = {
            'source': 'AI Recommended',
            'outfit_id': f"AI-Gen-{hero_id[:6]}",
            'theme': f"Dynamic {hero_occasion.title()} Outfit",
            'palette': palette_match,
            'stylist_rationale': rationale,
            'items': outfit_items
        }
        return outfit
