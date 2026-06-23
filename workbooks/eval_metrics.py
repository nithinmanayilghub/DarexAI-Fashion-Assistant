import os
import sys
import numpy as np
import pandas as pd

# Append project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, data_loader
from src.vector_store import VectorStore
from src.compatibility import CompatibilityEngine

def run_evaluation():
    print("==================================================")
    # 1. Load data
    products_df = data_loader.load_products()
    outfits_df = data_loader.load_outfits()
    
    # 2. Initialize VectorStore
    v_store = VectorStore(products_df)
    compat = CompatibilityEngine(outfits_df)
    
    print(f"Loaded {len(products_df)} products and {len(outfits_df)} curated outfits.")
    
    # Categories mapped to groups for evaluation
    roles_to_evaluate = {
        'second_id': 'bottomwear',
        'footwear_id': 'footwear',
        'layer_id': 'layer',
        'accessory_1_id': 'accessory'
    }
    
    # Aggregated metrics dictionaries
    # keys: 'random', 'text_only', 'hybrid'
    # values: list of reciprocals and hits
    results = {
        'random': {'rr': [], 'hit_1': [], 'hit_3': []},
        'text_only': {'rr': [], 'hit_1': [], 'hit_3': []},
        'hybrid': {'rr': [], 'hit_1': [], 'hit_3': []}
    }
    
    total_evals = 0
    
    for idx, row in outfits_df.iterrows():
        hero_id = row['hero_id']
        if pd.isna(hero_id):
            continue
            
        hero_product = data_loader.get_product_by_id(products_df, hero_id)
        if not hero_product:
            continue
            
        # Build search query from the hero product details
        search_query = f"{hero_product['name']} style {hero_product['occasion']} {hero_product['wear_type']}"
        gender_filter = hero_product['gender']
        
        # We will evaluate matching other items in this curated outfit
        for col_name, group_name in roles_to_evaluate.items():
            target_id = row[col_name]
            if pd.isna(target_id) or not isinstance(target_id, str):
                continue
                
            target_product = data_loader.get_product_by_id(products_df, target_id)
            if not target_product:
                continue
                
            allowed_categories = compat.category_groups[group_name]
            
            # --- EVALUATE HYBRID SEARCH ---
            # Retrieve all hits matching target group and gender
            hybrid_hits = v_store.search(
                query_text=search_query,
                gender=gender_filter,
                categories=allowed_categories,
                top_k=len(products_df)  # retrieve all possible matches to compute full rank
            )
            hybrid_ids = [hit['product']['id'] for hit in hybrid_hits if hit['product']['id'] != hero_id]
            
            # --- EVALUATE TEXT-ONLY SEARCH ---
            # Temporarily override vector store scores to text-only dot product
            query_emb = v_store.search(search_query, top_k=1) # get query embedder trigger
            # compute score using only text embeddings
            query_text_emb = v_store.search(search_query, top_k=1)
            query_flat = v_store.search(search_query, top_k=1) # helper
            
            # We calculate manually using text_embeddings
            q_emb = v_store.search(search_query, top_k=1) # trigger get text embeddings
            # Embed the query
            from src import embedder
            q_flat = embedder.get_text_embeddings(search_query).flatten()
            text_scores = np.dot(v_store.text_embeddings, q_flat)
            
            # Rank text-only products
            text_ranked_indices = np.argsort(text_scores)[::-1]
            text_hits = []
            for t_idx in text_ranked_indices:
                p_id = v_store.product_ids[t_idx]
                p_row = products_df[products_df['id'] == p_id].iloc[0]
                
                # Apply same filters
                # Gender filter
                if gender_filter and isinstance(gender_filter, str):
                    if p_row['gender'] != 'unisex' and gender_filter.lower() != 'unisex' and p_row['gender'] != gender_filter.lower():
                        continue
                # Category filter
                if p_row['category'] not in allowed_categories:
                    continue
                if p_id != hero_id:
                    text_hits.append(p_id)
            
            # --- RANDOM BASELINE ---
            # All available products in the allowed category group and gender
            all_options = []
            for _, p_row in products_df.iterrows():
                if p_row['id'] == hero_id:
                    continue
                if gender_filter and isinstance(gender_filter, str):
                    if p_row['gender'] != 'unisex' and gender_filter.lower() != 'unisex' and p_row['gender'] != gender_filter.lower():
                        continue
                if p_row['category'] in allowed_categories:
                    all_options.append(p_row['id'])
            
            if not all_options or target_id not in all_options:
                continue
                
            # Random expected rank: since it's uniform random, expected rank is (N+1)/2
            # expected reciprocal rank is sum(1/i for i in 1..N) / N
            N = len(all_options)
            random_rr = sum(1.0 / i for i in range(1, N + 1)) / N
            random_hit_1 = 1.0 / N
            random_hit_3 = min(3.0 / N, 1.0)
            
            # Compute Hybrid Rank
            try:
                hybrid_rank = hybrid_ids.index(target_id) + 1  # 1-indexed
            except ValueError:
                hybrid_rank = len(all_options) # fallback to last
                
            # Compute Text-Only Rank
            try:
                text_rank = text_hits.index(target_id) + 1  # 1-indexed
            except ValueError:
                text_rank = len(all_options) # fallback to last
                
            # Accumulate Results
            total_evals += 1
            
            results['random']['rr'].append(random_rr)
            results['random']['hit_1'].append(random_hit_1)
            results['random']['hit_3'].append(random_hit_3)
            
            results['text_only']['rr'].append(1.0 / text_rank)
            results['text_only']['hit_1'].append(1.0 if text_rank == 1 else 0.0)
            results['text_only']['hit_3'].append(1.0 if text_rank <= 3 else 0.0)
            
            results['hybrid']['rr'].append(1.0 / hybrid_rank)
            results['hybrid']['hit_1'].append(1.0 if hybrid_rank == 1 else 0.0)
            results['hybrid']['hit_3'].append(1.0 if hybrid_rank <= 3 else 0.0)
            
    print(f"Evaluated {total_evals} matching companion items across the catalog.")
    
    # 3. Print out results
    metrics_summary = []
    for model_name in ['random', 'text_only', 'hybrid']:
        mrr = np.mean(results[model_name]['rr'])
        hit1 = np.mean(results[model_name]['hit_1'])
        hit3 = np.mean(results[model_name]['hit_3'])
        metrics_summary.append({
            'Model': model_name.upper().replace('_', '-'),
            'MRR': f"{mrr:.4f}",
            'Hit Rate @ 1': f"{hit1 * 100:.2f}%",
            'Hit Rate @ 3': f"{hit3 * 100:.2f}%"
        })
        
    summary_df = pd.DataFrame(metrics_summary)
    print("\nRETRIEVAL PERFORMANCE EVALUATION SUMMARY:")
    print(summary_df.to_string(index=False))
    print("==================================================")
    
if __name__ == "__main__":
    run_evaluation()
