from fastapi import FastAPI
from pydantic import BaseModel
from ml_learn.preprocess import preprocess_text
from gensim.models.doc2vec import Doc2Vec #Preguntar a los TAs si es necesario importar gensim
import pandas as pd

#Instancia de FastAPI
app = FastAPI()

#Modelo y dataset
app.state.model = Doc2Vec.load("your_model_path")   # ¿Cómo cargamos nuestro modelo entrenado?
                                                    # No es el pre-entrenado?
app.state.df = pd.read_csv("data/Hotel_Reviews.csv")

@app.get("/search")
def search_hotels(search_query: str):
    # Preprocess the search query
    preprocessed_query = preprocess_text(search_query)

    # Infer the vector representation of the preprocessed query
    query_vector = app.infer_vector(preprocessed_query.split()) # Importar como app o model?

    # Find the most similar vectors to the query vector
    similar_vectors = app.dv.most_similar(positive=[query_vector], topn=5) # Importar como app o model?

    # Create an empty DataFrame to store the search results
    results = pd.DataFrame()

    # Retrieve the corresponding rows from the original DataFrame based on the similar vectors
    for i in range(len(similar_vectors)):
        index = similar_vectors[i][0]
        row = app.iloc[index] # Importar como app o df?
        results = results.append(row)

    # Convert the search results DataFrame to a list of dictionaries
    # with each dictionary representing a hotel and its attributes
    return results.to_dict(orient="records")
