import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def content_based_recommender(
    user_id,
    n_recommendations=5,
    behaviors_path="../2 - Preprocessing/processed_behaviours_train.parquet",
    news_path="../2 - Preprocessing/processed_news_train.parquet"
):
    """
    For a given user, loads local behaviors & news data from the specified parquet files,
    builds a user profile based on the textual content (TF-IDF) of articles read,
    and returns top n articles (that the user hasn't read) based on content similarity.

    Parameters:
      user_id (str): The user identifier from the behaviors DataFrame (column 'user_id').
      n_recommendations (int): Number of articles to recommend.
      behaviors_path (str): Path to the behaviors .parquet file.
      news_path (str): Path to the news .parquet file.

    Returns:
      (recommendations_df, explanation_str)
        - recommendations_df: DataFrame with columns [news_id, category, title, url] 
          or None if no recommendations found.
        - explanation_str: Explanation text or reason if no recommendations found.
    """
    # 1) Load the parquet files
    df_behav = pd.read_parquet(behaviors_path)
    df_news = pd.read_parquet(news_path)
    
    # 2) Filter behaviors for the specified user
    user_behav = df_behav[df_behav["user_id"] == user_id]
    if user_behav.empty:
        return None, f"No behavior records found for user {user_id}."
    
    # 3) Extract the reading history from the 'history' column (assumed to be space-separated news IDs)
    user_history_series = user_behav["history"].dropna().astype(str)
    all_read_ids = " ".join(user_history_series).split()
    if not all_read_ids:
        return None, f"User {user_id} has no reading history."
    
    # 4) Preprocess the 'content' column: lowercase and remove punctuation
    df_news["content_clean"] = df_news["content"].astype(str).str.lower().str.replace(r'[^\w\s]', '', regex=True)
    
    # 5) Build TF-IDF matrix on the cleaned content using scikit-learn’s built-in stop words.
    #    We use unigrams and bigrams, ignore very rare terms, and limit vocabulary size.
    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=10000,
        lowercase=True
    )
    tfidf_matrix = tfidf.fit_transform(df_news["content_clean"])
    
    # 6) Get indices for articles read by the user
    read_indices = df_news[df_news["news_id"].isin(all_read_ids)].index.tolist()
    all_indices = list(range(tfidf_matrix.shape[0]))
    unread_indices = [i for i in all_indices if i not in read_indices]
    if not read_indices or not unread_indices:
        return None, "Insufficient data: either no read articles or no unseen articles available."
    
    # 7) Construct the user profile vector by averaging the TF-IDF vectors of read articles
    user_profile_vector = np.asarray(tfidf_matrix[read_indices].mean(axis=0)).flatten()
    
    # 8) Compute cosine similarity between the user profile and TF-IDF vectors of unseen articles
    unread_matrix = tfidf_matrix[unread_indices]
    similarity_scores = cosine_similarity(user_profile_vector.reshape(1, -1), unread_matrix).flatten()
    top_indices = similarity_scores.argsort()[::-1][:n_recommendations]
    top_unread_indices = [unread_indices[i] for i in top_indices]
    
    # 9) Get recommendations from the news DataFrame
    recommendations = df_news.iloc[top_unread_indices][["news_id", "category", "title", "url"]]
    if recommendations.empty:
        return None, "No unseen articles found for recommendation."
    
    explanation = (
        f"Based on your reading history, here are {len(recommendations)} content-based recommendations:"
    )
    return recommendations, explanation
