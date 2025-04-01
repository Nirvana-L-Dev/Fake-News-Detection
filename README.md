# Fake News Detection

## Overview
This project aims to detect fake news articles using machine learning and natural language processing (NLP) techniques. The system analyzes the content of news articles and classifies them as either real or fake.

## Features
- Preprocessing of text data (removal of stopwords, stemming, tokenization, etc.)
- Machine learning models for classification
- Data visualization and exploratory data analysis (EDA)
- Evaluation of model performance using accuracy, precision, recall, and F1-score

## Technologies Used
- Python
- Pandas & NumPy (for data manipulation)
- Scikit-learn (for machine learning models)
- Matplotlib & Seaborn (for data visualization)
- NLTK & SpaCy (for NLP processing)

## Dataset
The dataset consists of two CSV files:
- `True.csv` - Contains real news articles
- `Fake.csv` - Contains fake news articles

Both files are merged into a single dataset with a **label column**:
- `1` = Real News
- `0` = Fake News

## Model Performance
- Logistic Regression, Decision Trees, and Random Forest models were tested.
- The best-performing model achieved an **accuracy of ~95%**.

## Required Libraries
```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
import spacy
