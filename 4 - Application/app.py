import streamlit as st
import openai
import pandas as pd
import sys
import os

# Adjust Python's path so we can import from ../3 - Models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "3 - Models")))
from BERT_model import BERT_recommender  # This is used by the BERT recommender

# =====================================
# SET UP OPENAI API (via st.secrets)
# =====================================
if "OPENAI_API_KEY" not in st.secrets:
    st.error("Please add your OpenAI API key to the Streamlit advanced settings (st.secrets).")
    st.stop()
else:
    openai.api_key = st.secrets["OPENAI_API_KEY"]

# =====================================
# Initialize Chatbot Session Variables
# =====================================
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": (
            "Hello, I'm ChatNews 🤖! I'm here to help you get the best news recommendations. "
            "To begin, please tell me what type of news interests you the most (e.g., Tech, Sports, Politics, or Movies)."
        )
    })

if "chat_stage" not in st.session_state:
    st.session_state["chat_stage"] = 0
if "chat_profile" not in st.session_state:
    st.session_state["chat_profile"] = None

# -----------------------------------
# PAGE CONFIG & GLOBAL STYLING
# -----------------------------------
st.set_page_config(
    page_title="SokoMind - News Recommender",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
        /* Overall Background & Font */
        body {
            background-color: #F8F3E9 !important;
            font-family: 'Segoe UI', Arial, sans-serif;
            color: #333333;
        }
        .stApp {
            max-width: 1100px;
            margin: 0 auto;
            padding: 20px;
        }
        /* Top-right Logo */
        .top-right-logo {
            position: absolute;
            top: 15px;
            right: 15px;
            z-index: 1000;
        }
        /* Title and Subtitle */
        .main-title {
            color: #0078D4;
            font-size: 2.4em;
            font-weight: 700;
            margin: 0;
        }
        .subtitle {
            color: #555555;
            font-size: 1.05em;
            font-weight: 400;
            margin-top: 6px;
        }
        /* Control Section */
        .control-section {
            background-color: #FAFAFA;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #E0E0E0;
            margin-bottom: 20px;
            display: flex;
            gap: 15px;
            align-items: center;
        }
        .control-label {
            color: #0078D4;
            font-size: 1em;
            font-weight: 500;
            margin-bottom: 5px;
        }
        /* Tabs */
        .stTabs {
            margin-bottom: 20px;
        }
        .stTabs [role="tab"] {
            background-color: #F0F0F0;
            color: #0078D4;
            border-radius: 10px 10px 0 0;
            padding: 10px 20px;
            font-weight: 500;
            font-size: 1em;
            transition: background-color 0.3s, color 0.3s;
        }
        .stTabs [role="tab"][aria-selected="true"] {
            background-color: #FFFFFF;
            color: #F25022;
            border-bottom: 3px solid #F25022;
        }
        .stTabs [role="tab"]:hover {
            background-color: #E8E8E8;
        }
        .tab-content {
            background-color: #FFFFFF;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #E0E0E0;
            margin-bottom: 20px;
        }
        /* Article Cards */
        .article-card {
            background-color: #E6F7FF;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
            border-left: 4px solid #0078D4;
        }
        .article-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        }
        .article-title {
            color: #0078D4;
            font-size: 1.15em;
            font-weight: 600;
            margin-bottom: 6px;
        }
        .article-summary {
            color: #333333;
            font-size: 0.95em;
            line-height: 1.5;
        }
        .description-text {
            color: #666666;
            font-size: 0.95em;
            font-style: italic;
            margin-bottom: 20px;
            line-height: 1.6;
        }
        /* Footer */
        .footer {
            text-align: center;
            color: #666666;
            font-size: 0.85em;
            margin-top: 30px;
            padding: 15px 0;
            border-top: 1px solid #E0E0E0;
        }
        .footer a {
            color: #0078D4;
            text-decoration: none;
        }
        .footer a:hover {
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)

# Title & Subtitle
st.markdown('<h1 class="main-title">SokoMind 🚀</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Discover personalized news recommendations using Microsoft technology ✨</p>', unsafe_allow_html=True)

# ================
# Control Section
# ================
st.markdown('<div class="control-section">', unsafe_allow_html=True)
col1, col2 = st.columns([2, 2])
with col1:
    st.markdown('<p class="control-label">Number of Recommendations</p>', unsafe_allow_html=True)
    num_recommendations = st.slider(
        "Number of Recommendations",
        1, 20, 5,
        label_visibility="collapsed",
        help="Adjust how many news articles you wish to see."
    )
with col2:
    st.write("")  # Empty column for spacing
st.markdown('</div>', unsafe_allow_html=True)

# =====================================
# TABS: Only BERT and LLM Recommenders
# =====================================
tabs = st.tabs(["BERT Recommender", "LLM Recommender"])

# TAB 1: BERT Recommender (Content-Based)
with tabs[0]:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">BERT Recommender 📄</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="description-text">This method recommends news using BERT-based embeddings extracted from the textual content of your read articles. Please enter your user ID for BERT recommendations.</p>',
        unsafe_allow_html=True
    )
    user_id_input_cb = st.text_input("Enter a user ID for content-based recommendations:", value="U12345", key="cb_user")
    
    if st.button("Get BERT Recommendations", key="cb_button"):
        try:
            # Note: We assume BERT_recommender returns (recommendations, explanation, metrics) 
            # but we ignore the metrics here.
            recommendations, explanation, _ = BERT_recommender(
                user_id=user_id_input_cb,
                n_recommendations=num_recommendations,
                behaviors_path="../2 - Preprocessing/processed_behaviours_train.parquet",
                news_path="../2 - Preprocessing/processed_news_train.parquet"
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")
            recommendations = None
            explanation = None

        if recommendations is None:
            st.markdown(f"<p class='description-text'>{explanation}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p class='description-text'>{explanation}</p>", unsafe_allow_html=True)
            for _, row in recommendations.iterrows():
                st.markdown(f"""
                    <div class="article-card">
                        <h3 class="article-title">{row['title']}</h3>
                        <p class="article-summary">Category: {row['category']}</p>
                        <p class="article-summary">
                            <a href="{row['url']}" target="_blank">Read more</a>
                        </p>
                    </div>
                """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
# TAB 2: LLM Recommender
with tabs[1]:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">LLM Recommender 🤖</h2>', unsafe_allow_html=True)
    st.markdown("""
        <p class="description-text">
            Welcome to your personalized news assistant! Please answer the following questions so I can understand your preferences based on your news interests.
            Your answers will be used to build your profile and recommend a dynamic number of articles (as set by the slider above) with clickable titles.
        </p>
    """, unsafe_allow_html=True)
    # Load the processed news dataset for dynamic category and subcategory options
    news_path_abs = r"C:\Users\Fernando Moreno\Desktop\MIND - Recommender System\2 - Preprocessing\processed_news_train.parquet"
    try:
        news_df = pd.read_parquet(news_path_abs)
    except Exception as e:
        st.error(f"Could not load the news dataset: {e}")
        news_df = None

    if news_df is not None:
        available_categories = sorted(news_df['category'].dropna().unique())
    else:
        available_categories = ["General"]

    chosen_category = st.selectbox("Question 1: Select your main news category:", available_categories)

    with st.form("chatbot_profile_form"):
        if news_df is not None and chosen_category:
            subcats = news_df[news_df['category'].str.contains(chosen_category, case=False, na=False)]['subcategory'].dropna().unique()
            subcat_options = ["All"] + sorted(subcats) if len(subcats) > 0 else ["All"]
        else:
            subcat_options = ["All"]
        chosen_subcategory = st.selectbox("Question 2: Select a subcategory (or choose 'All'):", subcat_options)
        title_keywords = st.text_input("Question 3: Enter keywords/phrases for article titles:", "")
        abstract_keywords = st.text_input("Question 4: Enter keywords/phrases for article abstracts:", "")
        additional_pref = st.text_input("Question 5: Any additional preferences? (Optional)", "")
        submitted = st.form_submit_button("Submit Preferences")

    if submitted:
        profile_summary = f"You selected the '{chosen_category}' category"
        if chosen_subcategory != "All":
            profile_summary += f" with a focus on '{chosen_subcategory}'"
        if title_keywords.strip():
            profile_summary += f"; you prefer titles containing '{title_keywords.strip()}'"
        if abstract_keywords.strip():
            profile_summary += f" and abstracts with '{abstract_keywords.strip()}'"
        if additional_pref.strip():
            profile_summary += f". Additional preference: '{additional_pref.strip()}'"
        else:
            profile_summary += "."
        
        explanation_text = f"Based on your preferences: {profile_summary} <br><br> Here are {num_recommendations} articles I recommend for you:"
        if news_df is not None:
            filtered_news = news_df[news_df['category'].str.contains(chosen_category, case=False, na=False)]
            if chosen_subcategory != "All":
                filtered_news = filtered_news[filtered_news['subcategory'].str.contains(chosen_subcategory, case=False, na=False)]
            if title_keywords.strip():
                kw_title = title_keywords.strip().lower()
                filtered_news = filtered_news[filtered_news['title'].str.lower().str.contains(kw_title, na=False)]
            if abstract_keywords.strip():
                kw_abstract = abstract_keywords.strip().lower()
                filtered_news = filtered_news[filtered_news['abstract'].str.lower().str.contains(kw_abstract, na=False)]
            
            if len(filtered_news) < num_recommendations:
                filtered_news = news_df[news_df['category'].str.contains(chosen_category, case=False, na=False)]
                if chosen_subcategory != "All":
                    filtered_news = filtered_news[filtered_news['subcategory'].str.contains(chosen_subcategory, case=False, na=False)]
            
            if 'date' in filtered_news.columns:
                filtered_news = filtered_news.sort_values(by='date', ascending=False)
            recommendations = filtered_news.head(num_recommendations)
            if recommendations.empty:
                explanation_text += "<br>Sorry, no articles match your criteria."
            else:
                bullet_list = "<br>".join(
                    [f"- <a href='{row['url']}' target='_blank'>{row['title']}</a> (Abstract: {row['abstract'][:100]}...)" 
                     for _, row in recommendations.iterrows() if pd.notnull(row['url'])]
                )
                explanation_text += "<br>" + bullet_list
        else:
            explanation_text += "<br>News details not available."

        st.markdown(f"<p class='description-text'>{explanation_text}</p>", unsafe_allow_html=True)

    if st.button("Restart Chatbot"):
        for key in list(st.session_state.keys()):
            if key.startswith("chatbot_") or key.startswith("q") or key == "chatbot_profile_form":
                del st.session_state[key]
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        else:
            st.write("Please refresh the page to restart the chatbot.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown("""
    <div class="footer">
        <p>© 2025 SokoMind - Developed for Microsoft Capstone Project</p>
        <p>Explore the <a href="https://msnews.github.io/" target="_blank">MIND dataset</a> 📊</p>
    </div>
""", unsafe_allow_html=True)
