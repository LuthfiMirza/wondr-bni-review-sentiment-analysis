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

## Architecture
```text
Google Play Store Reviews
        |
        v
scrape_playstore_reviews.py
        |
        v
wondr_reviews_gps.csv
        |
        v
training_wondr_sentiment.ipynb
        |
        +--> Text cleaning and sentiment labeling
        +--> Train/test split with stratification
        +--> TF-IDF feature extraction
        +--> Logistic Regression / Linear SVM experiments
        +--> Evaluation metrics and visualizations
        |
        v
models/wondr_sentiment_best.joblib
```

## Project Structure
```text
.
├── README.md
├── requirements.txt
├── scrape_playstore_reviews.py
├── training_wondr_sentiment.ipynb
├── wondr_reviews_gps.csv
└── models
    └── wondr_sentiment_best.joblib
```

## Prerequisites
- Python 3.10+ recommended
- pip
- Jupyter Notebook, JupyterLab, or Google Colab for running the notebook
- Internet access if scraping fresh Google Play reviews

## Installation & Setup

### 1. Clone or Prepare the Project
```bash
git clone <your-repository-url>
cd <repository-folder>
```

If you already have the project locally, open the project directory directly.

### 2. Create a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

If your environment does not already include Jupyter, install it separately:
```bash
pip install notebook
```

### 4. Scrape Reviews Optional
The repository already includes `wondr_reviews_gps.csv`. To collect a fresh dataset:

```bash
python scrape_playstore_reviews.py --min-rows 3000 --output wondr_reviews_gps.csv
```

Useful scraper options:
```bash
python scrape_playstore_reviews.py \
  --app-id com.bni.wonder \
  --lang id \
  --country id \
  --min-rows 3000 \
  --batch-size 200 \
  --sleep-sec 0.8 \
  --output wondr_reviews_gps.csv
```

### 5. Train and Evaluate
Open and run the notebook:

```bash
jupyter notebook training_wondr_sentiment.ipynb
```

The notebook loads the CSV dataset, performs preprocessing, trains models, evaluates results, and saves the best pipeline to:

```text
models/wondr_sentiment_best.joblib
```

