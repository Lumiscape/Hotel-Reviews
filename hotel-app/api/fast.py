from fastapi import FastAPI
from pydantic import BaseModel
from ml_learn.preprocess import preprocess_text
from gensim.models.doc2vec import Doc2Vec
import pandas as pd

app = FastAPI()

# Load the model
model = Doc2Vec.load("path_to_your_model_file")

# Load the dataset
df = pd.read_csv("data/Hotel_Reviews.csv")

@app.get("/search")
def search_hotels(search_query: str):
    # Preprocess the search query
    preprocessed_query = preprocess_text(search_query)

    # Infer the vector representation of the preprocessed query
    query_vector = model.infer_vector(preprocessed_query.split())

    # Find the most similar vectors to the query vector
    similar_vectors = model.dv.most_similar(positive=[query_vector], topn=20)

    # Create an empty DataFrame to store the search results
    results = pd.DataFrame()

    # Retrieve the corresponding rows from the original DataFrame based on the similar vectors
    for i in range(len(similar_vectors)):
        index = similar_vectors[i][0]
        row = df.iloc[index]
        results = results.append(row)

    # Convert the search results DataFrame to a list of dictionaries
    # with each dictionary representing a hotel and its attributes
    return results.to_dict(orient="records")

@app.get("/maps")
def search_maps():
