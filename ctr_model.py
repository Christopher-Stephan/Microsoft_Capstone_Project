# ctr_model.py

import os
import re
import ast
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

def ctr_recommender(user_id, n_recommendations=5,
                    behaviors_path="../2 - Preprocessing/processed_behaviours_train.parquet",
                    news_path="../2 - Preprocessing/processed_news_train.parquet"):
    """
    For a given user, loads and processes behaviors and news data, and recommends news articles
    using a simulated CTR approach based on entity similarity.
    
    This function:
      1. Loads data from the specified parquet files.
      2. Filters behaviors for the given user and extracts the reading history.
      3. Builds a news dictionary (mapping news IDs to CTR-related features).
      4. Computes a user entity vector based on the articles the user has read.
      5. Ranks unseen articles by cosine similarity between their entity vectors and the user vector,
         using a vectorized dot-product for speed.
      6. Returns the top n recommendations along with an explanation and evaluation metrics.
    
    Returns:
       recommendations_df (pd.DataFrame): DataFrame with columns [news_id, category, title, url]
       explanation_str (str): Explanation text.
       metrics_dict (dict): Evaluation metrics (if available), otherwise empty.
    """
    
    # --- Helper Functions ---
    def parse_entity_vector(s):
        if isinstance(s, (list, np.ndarray)):
            return np.array(s, dtype=float)
        if pd.isna(s) or s == '':
            return np.zeros(100, dtype=float)
        try:
            return np.array(ast.literal_eval(s), dtype=float)
        except Exception:
            return np.zeros(100, dtype=float)
    
    def get_user_entity_vector(history, news_dict):
        # Split the history into news IDs and obtain their entity vectors from the dictionary.
        ids = history.split()
        vectors = [news_dict[nid]['entity_vector'] for nid in ids if nid in news_dict]
        return np.mean(vectors, axis=0) if vectors else np.zeros(100, dtype=float)
    
    def compute_metrics(ground_truth_ids, recommended_ids):
        # Simple precision, recall, and F1-score
        if not ground_truth_ids:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        recommended_set = set(recommended_ids)
        ground_truth_set = set(ground_truth_ids)
        intersect_count = len(recommended_set & ground_truth_set)
        precision = intersect_count / len(recommended_set) if recommended_set else 0.0
        recall = intersect_count / len(ground_truth_set) if ground_truth_set else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}
    
    # --- Main Pipeline ---
    try:
        df_behav = pd.read_parquet(behaviors_path)
        df_news = pd.read_parquet(news_path)
    except Exception as e:
        return None, f"Error loading data files: {e}", {}
    
    # Filter behaviors for the given user
    user_behav = df_behav[df_behav["user_id"] == user_id]
    if user_behav.empty:
        return None, f"No behavior records found for user {user_id}.", {}
    
    # Extract reading history (assumed to be space-separated news IDs)
    history_series = user_behav["history"].dropna().astype(str)
    read_ids = " ".join(history_series).split()
    if not read_ids:
        return None, f"User {user_id} has no reading history.", {}
    
    # Build a news dictionary for CTR features.
    df_news['entity_vector'] = df_news['entity_vector'].apply(parse_entity_vector)
    # (We also build a combined category if needed)
    df_news['category_combined'] = df_news['category'] + ' ' + df_news['subcategory']
    news_dict = df_news.set_index('news_id')[['category', 'subcategory', 'entity_vector']].to_dict('index')
    
    # Compute the user entity vector from the user's reading history.
    user_entity_vector = get_user_entity_vector(" ".join(read_ids), news_dict)
    
    # Get candidate news articles (unseen by the user)
    candidate_news = df_news[~df_news['news_id'].isin(read_ids)].copy()
    if candidate_news.empty:
        return None, "No unseen articles available for recommendation.", {}
    
    # --- Vectorized cosine similarity ---
    # Stack candidate entity vectors into a 2D NumPy array
    candidate_vectors = np.stack(candidate_news['entity_vector'].values)
    # Normalize the user vector and candidate vectors
    user_norm = np.linalg.norm(user_entity_vector)
    if user_norm == 0:
        user_norm = 1
    user_unit = user_entity_vector / user_norm
    candidate_norms = np.linalg.norm(candidate_vectors, axis=1)
    candidate_norms[candidate_norms == 0] = 1
    candidate_unit = candidate_vectors / candidate_norms[:, None]
    # Compute dot product between each candidate and the user vector
    similarity_scores = np.dot(candidate_unit, user_unit)
    
    candidate_news['similarity'] = similarity_scores
    candidate_news.sort_values(by='similarity', ascending=False, inplace=True)
    
    # Select top n recommendations
    recommendations = candidate_news.head(n_recommendations)
    if recommendations.empty:
        return None, "No recommendations could be generated.", {}
    
    explanation = f"Based on your CTR behavior, here are {len(recommendations)} recommendations."
    
    # For evaluation, here we simulate ground truth using the read_ids (this is only a placeholder)
    metrics_dict = compute_metrics(read_ids, list(recommendations['news_id'].values))
    
    return recommendations[["news_id", "category", "title", "url"]], explanation, metrics_dict
