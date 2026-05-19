# Lazada App Reviews — Sentiment Analysis

A multiclass NLP sentiment classifier trained on **775,000+ real customer reviews** from Lazada's Google Play Store listing. The project covers the full ML pipeline: raw text ingestion → EDA → preprocessing → feature engineering → model comparison → final model selection.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Pipeline](#project-pipeline)
- [Models & Results](#models--results)
- [Key Challenges & Solutions](#key-challenges--solutions)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)

---

## Overview

The goal is to classify customer sentiment (positive / neutral / negative) from free-form text reviews of the Lazada mobile app. Rather than relying on star ratings alone, the model learns linguistic patterns from cleaned, tokenized review text — making it applicable to unrated text data in the wild.

**Target variable:** `sentiment` (derived from `review_rating`)
- ⭐⭐⭐⭐–⭐⭐⭐⭐⭐ → Positive  
- ⭐⭐⭐ → Neutral  
- ⭐–⭐⭐ → Negative

---

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Lazada App Reviews (Google Store)](https://www.kaggle.com/datasets/bwandowando/lazada-app-reviews-from-google-store/data) |
| Rows | 775,323 |
| Features used | `review_text`, `review_rating` |
| Date range | 2013 – 2023 |
| Class distribution | Severely imbalanced (positive-heavy) |

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn`, `wordcloud` |
| NLP & text | `nltk`, `emoji`, `re` |
| Feature engineering | `TF-IDF` (scikit-learn) |
| ML models | `scikit-learn`, `imbalanced-learn` |
| Model persistence | `joblib` |

---

## Project Pipeline

### 1. Exploratory Data Analysis (EDA)
- Inspected class distribution — found a **severe class imbalance**: positive reviews overwhelmingly dominated the dataset, with neutral reviews being extremely scarce.
- Analysed review text length distribution (mean: 7 words, max: 379 words); identified that long multi-sentiment reviews could confuse classifiers.
- Verified reviewer bias: confirmed no single author posted more than 2 reviews, ruling out individual-level data skew.
- Handled missing values in `review_text` (305 null rows dropped).

### 2. Text Preprocessing
A reusable `clean_text()` function applies the following steps in order:

| Step | Technique | Rationale |
|---|---|---|
| Lowercasing | `str.lower()` | Normalise vocabulary |
| URL / HTML / mention removal | `re.sub()` | Strip non-semantic noise |
| Number removal | `re.sub(r'\d+', '', ...)` | Remove order IDs, dates |
| Punctuation removal | `str.translate()` | Reduce noise |
| Emoji conversion | `emoji.demojize()` | Preserve emoji sentiment as text |
| Repeated character normalisation | regex `(.)\1{2,}` | "goooood" → "good" |
| Tokenisation | `nltk.word_tokenize()` | Split text into tokens |
| Stopword removal | `nltk.corpus.stopwords` | Remove low-signal function words |
| Lemmatisation | `WordNetLemmatizer` | Reduce inflected forms to root ("running" → "run") |

> **Lemmatisation vs Stemming:** Lemmatisation was chosen over stemming for more linguistically accurate root forms, which improves TF-IDF feature quality at the cost of slightly higher computation.

### 3. Feature Engineering — TF-IDF Vectorisation
- **TF-IDF** (Term Frequency–Inverse Document Frequency) used to convert cleaned text into numerical feature vectors.
- `max_features=5000` to cap vocabulary size and prevent overfitting on rare terms.
- `ngram_range=(1, 2)` to capture both unigrams and bigrams (e.g., "not good" treated as a single feature).

### 4. Class Imbalance Handling
- Used `class_weight='balanced'` in Logistic Regression to penalise misclassification of minority classes proportionally.
- Evaluated macro-averaged F1 score (treats all classes equally) alongside accuracy to avoid misleading results from class imbalance.

### 5. Model Training & Hyperparameter Tuning
- Stratified 70/30 train-test split with `stratify=y` to preserve class proportions.
- Scikit-learn `Pipeline` used throughout to prevent data leakage (vectoriser fitted only on training data).
- **Stratified K-Fold Cross-Validation** (5 folds) for robust, unbiased performance estimates.
- **GridSearchCV** used for Naive Bayes and SVM hyperparameter tuning.

---

## Models & Results

Three classifiers were benchmarked against the same preprocessed dataset:

| Model | CV Accuracy | CV F1 Macro | Notes |
|---|---|---|---|
| **Logistic Regression** ✅ | **0.78** | **0.70** | Baseline; best overall performer |
| Linear SVC (SGDClassifier) | 0.76 | 0.68 | Scalable SVM approximation |
| Multinomial Naive Bayes | 0.76 | 0.65 | Tuned with GridSearchCV (alpha, fit_prior) |

**Winner: Logistic Regression with TF-IDF**

Logistic Regression consistently outperformed both alternatives across accuracy and macro F1. Its probabilistic outputs and well-calibrated `class_weight='balanced'` parameter made it particularly effective on the neutral minority class.

All fitted models are serialised to `all_models.pkl` using `joblib` for downstream inference without retraining.

---

## Key Challenges & Solutions

| Challenge | Solution |
|---|---|
| **Severe class imbalance** (positive >> neutral) | `class_weight='balanced'` in Logistic Regression + macro F1 evaluation metric |
| **Data leakage risk** from TF-IDF | Wrapped vectoriser + classifier inside a `sklearn.Pipeline` |
| **Large dataset** (775K rows) — SVC too slow | Replaced `SVC` with `SGDClassifier(loss='hinge')` — equivalent to SVM, scales linearly |
| **Multi-sentiment long reviews** | Lemmatisation + TF-IDF bigrams to capture phrase-level context |
| **Noisy informal text** (emojis, slang) | `emoji.demojize()` to preserve emoji sentiment; regex-based normalisation |

---

## Project Structure

```
├── Sentimental Analysis (Lazada).ipynb   # Main notebook
├── LAZADA_REVIEWS.csv                    # Raw dataset (from Kaggle)
├── all_models.pkl                        # Serialised trained models (joblib)
└── README.md
```

---

## Getting Started

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn nltk emoji wordcloud langdetect
pip install transformers torch
pip install sentencepiece
pip install imbalanced-learn

# Download NLTK assets (first run)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

Then open `Sentimental Analysis (Lazada).ipynb` in Jupyter and run all cells.

To load and use a saved model directly:

```python
import joblib

models = joblib.load("all_models.pkl")
lr_model = models["logistic"]

predictions = lr_model.predict(["This app keeps crashing, very frustrating"])
print(predictions)  # ['negative']
```

---

## Acknowledgements

Dataset sourced from [Kaggle — Lazada App Reviews (Google Store)](https://www.kaggle.com/datasets/bwandowando/lazada-app-reviews-from-google-store/data) by bwandowando.
