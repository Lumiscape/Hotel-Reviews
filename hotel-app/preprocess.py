import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
import numpy as np

df = pd.read_csv('../data/Hotel_Reviews.csv')

def cleaner(df):
    df.dropna(subset=['lat', 'lng'],inplace=True)
    df.columns = df.columns.str.lower()
    columns_to_remove = ['hotel_address', 'additional_number_of_scoring', 'review_date', 'average_score',
                         'reviewer_nationality', 'review_total_negative_word_counts', 'total_number_of_reviews',
                         'review_total_positive_word_counts', 'total_number_of_reviews_reviewer_has_given', 'tags',
                         'days_since_review', 'lat', 'lng']
    df = df.drop(columns=columns_to_remove)
    # Replace 'no negative' with an empty string
    df['negative_review'] = df['negative_review'].replace('No Negative', '')

    # Replace 'no positive' with an empty string
    df['positive_review'] = df['positive_review'].replace('No Positive', '')

    # Combine positive and negative reviews into a new column
    df['combined_review'] = df['positive_review'] + ' ' + df['negative_review']

    # Drop the individual positive and negative review columns
    df = df.drop(['positive_review', 'negative_review'], axis=1)


    return df

def preprocess_text(text):

    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = word_tokenize(text)
    preprocessed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and not token.isnumeric()]
    preprocessed_text = ' '.join(preprocessed_tokens)

    return preprocessed_text
