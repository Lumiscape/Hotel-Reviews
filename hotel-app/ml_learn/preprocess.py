import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
import numpy as np

df = pd.read_csv('./data/Hotel_Reviews.csv')

def cleaner(df):
    df.dropna(subset=['lat', 'lng']).reset_index(drop=True,inplace=True)
    df.columns = df.columns.str.lower()
    columns_to_remove = ['additional_number_of_scoring', 'review_date', 'average_score',
                         'reviewer_nationality','negative_review', 'review_total_negative_word_counts', 'total_number_of_reviews',
                         'review_total_positive_word_counts', 'total_number_of_reviews_reviewer_has_given', 'tags',
                         'days_since_review']
    df = df.drop(columns=columns_to_remove)

    # Replace 'no positive' with an empty string
   # df['positive_review'] = df['positive_review'].replace('No Positive', '')
    df = df[df['positive_review'] != 'No Positive']
    df.rename(columns={'positive_review': 'review'}, inplace=True)

    return df

def preprocess_text(text):

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = word_tokenize(text)
    preprocessed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and not token.isnumeric()]
    preprocessed_text = ' '.join(preprocessed_tokens)

    return preprocessed_text
