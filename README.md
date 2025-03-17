# SokoMind News Recommender System - MIND

Welcome to the SokoMind News Recommender System project! This project aims to deliver personalized news article recommendations to users, enhancing their engagement and satisfaction. By implementing recommender models, we can provide accurate and diverse recommendations based on user preferences and article characteristics.

## Table of Contents
1. [Introduction](#introduction)
2. [Exploratory Data Analysis](#data-analysis)
3. [Data Preparation and Feature Engineering](#data-preparation-and-feature-engineering)
4. [Recommender Models](#recommender-models)
5. [Model Evaluation](#model-evaluation)
6. [Business Application and ROI](#business-application-and-roi)
7. [Production and Deployment](#production-and-deployment)
8. [Future Improvements](#future-improvements)

## Introduction
This project explores the development of a news recommender system for SokoMind News, aiming to deliver tailored news articles to users. The importance of personalized news recommendations in today's information-rich world is emphasized, and three main approaches in recommender systems are discussed: category frequency, a BERT based model and finally an NLP based news articles search. 
### Data for Reproducibility

This repository does not contain a folder *Dataset* since the necessary data to perform the analysis is located, due to Github's limitations on handling large data files. While in the future, we want to include a DVC version of this folder.

## Data Analysis
The project utilizes the Microsoft News Dataset (MIND), comprising "news" and "behaviors" datasets. Initial data analysis yielded valuable insights into the dataset's structure, data quality, user interactions, and article characteristics. These findings informed feature engineering, model selection, and system optimization.​


## Data Preparation and Feature Engineering
During the data preparation stage, we addressed missing values, eliminated duplicate entries, and converted variables into the proper formats. The textual data was processed by tokenizing, filtering out stop words, and applying methods such as stemming or lemmatization. We engineered new features by creating additional variables, computing popularity metrics, and incorporating temporal aspects. Additionally, collaborative filtering methods were employed to measure user similarity, which enhanced the accuracy and personalization of the recommendations.


## Recommender Models
To deliver personalized news article recommendations, we have developed a suite of recommender models, each tailored to different stages of the user journey and leveraging advanced Natural Language Processing (NLP) techniques:​

1. Category Frequency-Based Recommender (Model 1):

For new users without prior interaction history, this model suggests popular articles within specific categories. By analyzing the frequency of article views or interactions within each category, the system identifies trending topics and recommends them to new users, ensuring immediate engagement. ​
arxiv.org
arxiv.org

2. BERT-Based Content Filtering Recommender (Model 2):

As users begin interacting with the platform, the system employs a BERT-based model to understand the semantic content of articles. By generating embeddings for each article using BERT, the model captures contextual nuances and recommends articles similar in content to those previously read by the user. This approach ensures that recommendations align closely with user interests based on content similarity. ​In order to be able to run it properly, first run BERT_recommender.py, obtain the nrms_bert_state_dict.pkl and then you will be able to have that recommender available when deploying in Streamlit.

3. LLM-Based Recommender with Filtered Content Access (Model 3):

For users with extensive interaction histories, we utilize a Large Language Model (LLM)-based recommender. This model not only considers user preferences but also incorporates specific filters, restrictions, and keywords to tailor recommendations. By leveraging the advanced capabilities of LLMs, the system can parse complex user queries and deliver highly customized news articles that meet specific criteria, enhancing the personalization aspect of the recommendations.​

By integrating these models, the recommender system adapts to the user's journey, from initial engagement to established interaction patterns, ensuring relevant and personalized news content delivery at every stage.

## Model Evaluation
The ctr model was successfully evaluated using various metrics demonstrating its effectiveness in recommending articles aligned with user preferences. Although the models' performance falls short of state-of-the-art models, it shows significant improvement over a random baseline and hold promise for business applications.

### Proof of Concept Streamlit App
To demonstrate the capabilities of the SokoMind News Recommender System, we've developed a user-friendly application using Streamlit, an open-source Python framework designed for creating interactive data applications. This application showcases the functionality of our three recommendation models at various stages of the user journey.​
streamlit.io

Key Features of the Streamlit Application:

    User Preference Input: Users can specify their interests and preferences, allowing the system to tailor news recommendations accordingly.​

    Interactive Experience: The intuitive interface enables users to engage with the system seamlessly, enhancing the overall user experience.​

    Personalized Recommendations: Based on the provided preferences, the system delivers customized news article suggestions, ensuring content relevance.
To run the app, execute in the terminal the following command after indicating the path to the App.py file:
`streamlit run App.py`

## Production and Deployment
To efficiently deploy models in production, Microsoft Azure Cloud has been selected as the preferred platform due to its dynamic resource allocation, scalability, and cost efficiency through a pay-as-you-go model. Azure's autoscaling capabilities allow for automatic adjustment of resources based on actual demand, optimizing both performance and costs. ​

To ensure compliance with legal and regulatory requirements, Azure offers robust data protection features. All data written to the Azure storage platform is encrypted using 256-bit AES encryption and is FIPS 140-2 compliant. Additionally, Azure provides key management options, including Azure Key Vault, to securely manage encryption keys. ​
azure.microsoft.com

Furthermore, Azure's compliance offerings include adherence to the EU Cloud Code of Conduct, demonstrating its commitment to data privacy and protection. These measures collectively ensure that legal obligations are met, and user trust is maintained.​


## Future Improvements
The SokoMind News Recommender System is committed to enhancing your news reading experience by delivering personalized and relevant content. Our ongoing efforts include developing an in-house dataset that integrates user demographics, article ratings, and real-time external events. This comprehensive approach aims to refine the accuracy and relevance of our recommendations, ensuring a more tailored news experience for each user.​

Continuous improvement and optimization are at the core of our strategy to remain competitive in the dynamic news industry. By incorporating advanced methodologies and staying attuned to user preferences, we strive to provide a platform that not only informs but also engages.​

We appreciate your interest in the SokoMind News Recommender System and look forward to enriching your news consumption journey.

## Acknowledgments

This project is part of the Master in Business Analytics and Big Data program as the capstone Project at IE University in Madrid. We would like to express our gratitude to the program faculty and staff for their guidance and support throughout this project.

The project was developed by:

- Fernando Moreno
- Filippo Lisanti
- Christopher Stephan
- Sofia Depoortere
- Hugo Bojórquez

We would also like to thank Microsoft for providing the necessary resources and data for this project.
