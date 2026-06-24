import os
import pickle
import numpy as np
from src import config, data_loader, embedder

class VectorStore:
    def __init__(self, products_df=None):
        self.cache_path = config.EMBEDDINGS_CACHE_PATH
        self.products_df = products_df if products_df is not None else data_loader.load_products()
        self.product_ids = []
        self.image_embeddings = None
        self.text_embeddings = None
        self.hybrid_embeddings = None
        
        # Load or build index
        self.load_or_build_index()

    def _compute_hybrid_embeddings(self):
        """Computes hybrid embeddings dynamically using current config.HYBRID_ALPHA."""
        alpha = getattr(config, 'HYBRID_ALPHA', 0.05)
        print(f"Computing hybrid embeddings dynamically (HYBRID_ALPHA={alpha})...")
        hybrid_raw = (alpha * self.image_embeddings) + ((1.0 - alpha) * self.text_embeddings)
        norms = np.linalg.norm(hybrid_raw, axis=1, keepdims=True)
        self.hybrid_embeddings = hybrid_raw / np.where(norms == 0, 1.0, norms)

    def load_or_build_index(self):
        """Loads embeddings cache if it exists, otherwise generates it."""
        if os.path.exists(self.cache_path):
            print(f"Loading cached embeddings from {self.cache_path}...")
            with open(self.cache_path, 'rb') as f:
                data = pickle.load(f)
                
                # Check for raw embeddings to compute dynamic blending
                if 'image_embeddings' in data and 'text_embeddings' in data:
                    self.product_ids = data['product_ids']
                    self.image_embeddings = data['image_embeddings']
                    self.text_embeddings = data['text_embeddings']
                    self._compute_hybrid_embeddings()
                else:
                    # Fallback for old cache formats
                    print("Warning: Raw embeddings not found in cache. Using cached hybrid embeddings.")
                    self.product_ids = data['product_ids']
                    self.hybrid_embeddings = data['hybrid_embeddings']
                    
            print(f"Loaded {len(self.product_ids)} product embeddings successfully.")
        else:
            print("No cached embeddings found. Building index...")
            self.build_index()

    def build_index(self):
        """Generates CLIP embeddings for both product images and text descriptions."""
        product_ids = []
        img_embs = []
        txt_embs = []
        
        total = len(self.products_df)
        for i, row in self.products_df.iterrows():
            prod_id = row['id']
            print(f"Encoding product {i+1}/{total}: {prod_id}")
            product_ids.append(prod_id)
            
            # Generate image embedding
            img_path = row['absolute_image_path']
            img_emb = embedder.get_image_embedding(img_path)
            img_embs.append(img_emb.flatten())
            
            # Generate text embedding using CLIP-Specific Prompt Engineering
            desc_clean = str(row['description']).strip()
            full_text = f"A studio catalog photograph of a {row['brand']} {row['category_label']} in a {row['occasion']} style. Details: {desc_clean}"
            txt_emb = embedder.get_text_embeddings(full_text)
            txt_embs.append(txt_emb.flatten())
            
        self.product_ids = product_ids
        self.image_embeddings = np.array(img_embs)
        self.text_embeddings = np.array(txt_embs)
        
        # Compute hybrid embeddings dynamically using our helper
        self._compute_hybrid_embeddings()
        
        # Save to disk
        cache_dir = os.path.dirname(self.cache_path)
        os.makedirs(cache_dir, exist_ok=True)
        with open(self.cache_path, 'wb') as f:
            pickle.dump({
                'product_ids': self.product_ids,
                'image_embeddings': self.image_embeddings,
                'text_embeddings': self.text_embeddings,
                'hybrid_embeddings': self.hybrid_embeddings  # Saved for backwards-compatibility
            }, f)
        print(f"Embeddings cache successfully saved to {self.cache_path}")

    def search(self, query_text, gender=None, categories=None, top_k=5):
        """
        Searches catalog using text-to-text / text-to-image similarity.
        Optionally filters results by gender and category list.
        """
        # Embed the search query
        query_emb = embedder.get_text_embeddings(query_text).flatten()
        
        # Compute cosine similarity using dot product against hybrid embeddings
        # self.hybrid_embeddings is of shape (N, 512), query_emb is (512,)
        scores = np.dot(self.hybrid_embeddings, query_emb)
        
        # Rank all products
        ranked_indices = np.argsort(scores)[::-1]
        
        results = []
        for idx in ranked_indices:
            prod_id = self.product_ids[idx]
            prod_score = float(scores[idx])
            
            # Retrieve product row
            prod_row = self.products_df[self.products_df['id'] == prod_id].iloc[0]
            
            # Apply Gender Filter
            if gender and isinstance(gender, str):
                g_lower = gender.lower()
                p_gender = str(prod_row['gender']).lower()
                # 'unisex' can match either gender, or exact match
                if p_gender != 'unisex' and g_lower != 'unisex' and p_gender != g_lower:
                    continue
                    
            # Apply Category Filter
            if categories:
                # categories can be a list of strings
                p_cat = str(prod_row['category']).lower()
                if p_cat not in [c.lower() for c in categories]:
                    continue
                    
            results.append({
                'product': prod_row.to_dict(),
                'score': prod_score
            })
            
            if len(results) >= top_k:
                break
                
        return results
