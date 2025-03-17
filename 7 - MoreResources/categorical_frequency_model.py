import pandas as pd
import os

def frequency_categorical_recommender(
    user_id,
    n_recommendations=5,
    behaviors_path="../2 - Preprocessing/processed_behaviours_train.parquet",
    news_path="../2 - Preprocessing/processed_news_train.parquet"
):
    """
    For a given user, loads local behaviors & news data from the specified parquet files,
    determines the user's most frequently read category, and returns top n articles in that category
    that the user hasn't read yet.

    Parameters:
      user_id (str): The user identifier from the behaviors DataFrame (column 'user_id').
      n_recommendations (int): Number of articles to recommend.
      behaviors_path (str): Path to the behaviors .parquet file.
      news_path (str): Path to the news .parquet file.

    Returns:
      (recommendations_df, explanation_str)
        - recommendations_df: DataFrame with columns [news_id, category, title, url] or None if no recs found
        - explanation_str: Explanation text or reason if no recs found
    """

    # 1) Load the parquet files
    df_behav = pd.read_parquet(behaviors_path)
    df_news = pd.read_parquet(news_path)

    # 2) Filter behaviors for the specified user
    user_behav = df_behav[df_behav["user_id"] == user_id]
    if user_behav.empty:
        return None, f"No behavior records found for user {user_id}"

    # 3) Extract the reading history from the 'history' column (space-separated news IDs)
    #    Example columns: impression_id, user_id, time, history, ...
    #    'history' might look like: "N12345 N67890 N22222"
    user_history_series = user_behav["history"].dropna().astype(str)
    # Join all space-separated news IDs from multiple rows
    all_read_ids = " ".join(user_history_series).split()

    if not all_read_ids:
        return None, f"User {user_id} has no reading history."

    # 4) Count the categories for those read IDs
    #    We assume df_news has columns: news_id, category, title, url, etc.
    read_df = pd.DataFrame({"news_id": all_read_ids})
    merged_read = pd.merge(read_df, df_news[["news_id", "category"]], on="news_id", how="left")
    if merged_read["category"].dropna().empty:
        return None, "No category information found for the user's read news."

    # Determine the top category by frequency
    top_category = merged_read["category"].value_counts().idxmax()

    # 5) Get candidate articles in that category, excluding those read
    read_set = set(all_read_ids)
    candidates = df_news[df_news["category"] == top_category].copy()
    candidates = candidates[~candidates["news_id"].isin(read_set)]

    if candidates.empty:
        explanation = (
            f"Based on your history, you appear to favor **{top_category}** news, "
            "but no unseen articles remain in that category."
        )
        return None, explanation

    # 6) Return first n_recommendations
    #    Optionally, sort by some column (e.g., recency). We'll just do top rows as an example.
    recommendations = candidates.head(n_recommendations)

    explanation = (
        f"Based on your reading history, you appear to favor **{top_category}** news. "
        f"Here are {len(recommendations)} articles you haven't read yet:"
    )
    return recommendations[["news_id", "category", "title", "url"]], explanation
