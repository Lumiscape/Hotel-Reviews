import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from ml_learn.preprocess import preprocess_text

#df = pd.read_csv('./data/df_clean.csv')

#df['review'] = df['review'].apply(preprocess_text)

def train_model(df):

    documents = [TaggedDocument(words=text.split(), tags=[i]) for i, text in enumerate(df['review'])]

    model = Doc2Vec(documents, vector_size=50, window=5, min_count=2, workers=4,epochs=5)

    return model
def search_similar_vectors(text:str,model):

     text_test = preprocess_text(text)
     inferred_vector = model.infer_vector(text_test.split())
     similar_vectors = model.dv.most_similar(positive=[inferred_vector], topn = 20)

     return similar_vectors

def index_vectors(similar_vectors):
    i_vectors = [doc[0] for doc in similar_vectors]
    return i_vectors
