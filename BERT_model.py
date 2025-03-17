# content_based_model.py

import os
import re
import ast
import numpy as np
import pandas as pd
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

def BERT_recommender(user_id,
                              n_recommendations=5,
                              behaviors_path="../2 - Preprocessing/processed_behaviours_train.parquet",
                              news_path="../2 - Preprocessing/processed_news_train.parquet"):
    """
    Loads behaviors and news data from parquet files, builds a TF-IDF model over the cleaned
    combined text of articles (title + abstract + category + subcategory), creates a user profile
    by averaging the TF-IDF vectors for articles the user has read, and returns the top n unseen articles
    by cosine similarity.

    If the behaviors data contains a 'clicked_news' column, evaluation metrics (precision, recall, F1)
    are computed.

    Returns:
       recommendations_df (pd.DataFrame): DataFrame with columns [news_id, category, title, url]
       explanation_str (str): Explanation text.
       metrics_dict (dict): Evaluation metrics (if available), otherwise an empty dictionary.
    """

    # --- Helper functions defined inside the main function ---
    def simple_tokenize(text):
        """Lowercase text, remove punctuation using regex, and split by whitespace."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        return tokens

    def compute_metrics(ground_truth_ids, recommended_ids):
        """Compute precision, recall, and F1-score given ground-truth and recommended news IDs."""
        if not ground_truth_ids:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        # Normalize IDs by stripping whitespace and ensuring strings
        recommended_set = set(str(r).strip() for r in recommended_ids)
        ground_truth_set = set(str(g).strip() for g in ground_truth_ids)
        intersect_count = len(recommended_set & ground_truth_set)
        precision = intersect_count / len(recommended_set) if recommended_set else 0.0
        recall = intersect_count / len(ground_truth_set) if ground_truth_set else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    def extract_clicked_news(entry):
        """
        Extracts news IDs from a clicked_news entry.
        If the entry is a string representation of a list, attempt to parse it.
        Then use regex to extract IDs like "N12345".
        """
        try:
            parsed = ast.literal_eval(entry)
            if isinstance(parsed, list):
                entry = " ".join(parsed)
        except Exception:
            pass
        return re.findall(r'N\d+', entry)

    # --- Main pipeline ---
    # 1. Load data
    try:
        df_behav = pd.read_parquet(behaviors_path)
        df_news = pd.read_parquet(news_path)
    except Exception as e:
        return None, f"Error loading data files: {e}", {}

    # 2. Filter behaviors for the given user
    user_behav = df_behav[df_behav["user_id"] == user_id]
    if user_behav.empty:
        return None, f"No behavior records found for user {user_id}.", {}

    # 3. Extract reading history (assumed to be space-separated news IDs)
    history_series = user_behav["history"].dropna().astype(str)
    read_ids = " ".join(history_series).split()
    if not read_ids:
        return None, f"User {user_id} has no reading history.", {}

    # 4. Build a combined text field for each news article using title, abstract, category, and subcategory
    df_news["title"] = df_news["title"].fillna("")
    df_news["abstract"] = df_news["abstract"].fillna("")
    df_news["category"] = df_news["category"].fillna("")
    df_news["subcategory"] = df_news["subcategory"].fillna("")
    df_news["combined"] = df_news["title"] + " " + df_news["abstract"] + " " + df_news["category"] + " " + df_news["subcategory"]

    # 5. Preprocess the combined text using our simple tokenizer
    df_news["cleaned"] = df_news["combined"].apply(lambda text: " ".join(simple_tokenize(text)))

    # 6. Build TF-IDF matrix over the cleaned text
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_features=10000)
    tfidf_matrix = tfidf.fit_transform(df_news["cleaned"])

    # 7. Determine indices of read and unread articles
    read_indices = df_news.index[df_news["news_id"].isin(read_ids)].tolist()
    all_indices = list(range(tfidf_matrix.shape[0]))
    unread_indices = [i for i in all_indices if i not in read_indices]
    if not read_indices or not unread_indices:
        return None, "Either no articles have been read or no unseen articles available.", {}

    # 8. Build the user profile vector (average of TF-IDF vectors for read articles)
    user_profile_vector = tfidf_matrix[read_indices].mean(axis=0)
    user_profile_vector = np.asarray(user_profile_vector).flatten()

    # 9. Compute cosine similarity between the user profile and unseen articles
    unread_matrix = tfidf_matrix[unread_indices]
    similarity_scores = cosine_similarity(user_profile_vector.reshape(1, -1), unread_matrix).flatten()

    # 10. Rank unseen articles by similarity and select the top n recommendations
    top_indices = similarity_scores.argsort()[::-1][:n_recommendations]
    chosen_indices = [unread_indices[i] for i in top_indices]
    recommendations = df_news.iloc[chosen_indices][["news_id", "category", "title", "url"]]
    if recommendations.empty:
        return None, "No unseen articles found for recommendation.", {}

    explanation = (f"Based on your reading history ({len(read_ids)} articles), "
                   f"here are {len(recommendations)} content-based recommendations.")

    # 11. Compute evaluation metrics if ground truth exists (clicked_news)
    metrics_dict = {}
    if "clicked_news" in user_behav.columns:
        ground_truth_ids = set()
        clicked_series = user_behav["clicked_news"].dropna().astype(str)
        for entry in clicked_series:
            ground_truth_ids.update(extract_clicked_news(entry))
        recommended_ids = list(recommendations["news_id"].values)
        metrics_dict = compute_metrics(ground_truth_ids, recommended_ids)

    return recommendations, explanation, metrics_dict