# Wondr App Review Sentiment Analysis

## Overview
This project builds an end-to-end sentiment analysis workflow for Google Play Store reviews of the **Wondr by BNI** mobile application. It collects user reviews, cleans review text, assigns sentiment labels from star ratings, trains several machine learning baselines, evaluates model performance, and saves the best-performing model for reuse.

The project is designed for product, data, or customer-experience stakeholders who need a practical way to summarize large volumes of app feedback into **positive**, **neutral**, and **negative** sentiment categories. It also demonstrates a complete machine learning workflow suitable for a professional data science portfolio.

## Technology Stack
- **Language:** Python
- **Core Libraries:** pandas, NumPy, scikit-learn, NLTK, joblib
- **Visualization:** Matplotlib, Seaborn
- **Data Collection:** google-play-scraper
- **Modeling:** TF-IDF feature extraction, Logistic Regression, Linear SVM
- **Storage:** CSV dataset and serialized `.joblib` model artifact
- **Notebook Environment:** Jupyter Notebook / Google Colab-compatible workflow

## Features
- Scrapes Google Play Store reviews for the Wondr by BNI app.
- Cleans Indonesian and English review text using regex normalization and stopword removal.
- Converts 1–5 star ratings into sentiment labels: negative, neutral, and positive.
- Trains and compares multiple machine learning pipelines.
- Evaluates models with accuracy, macro F1, weighted F1, classification report, and confusion matrix.
- Saves the best model pipeline for future inference.
- Includes a reproducible notebook documenting the full experimentation process.

