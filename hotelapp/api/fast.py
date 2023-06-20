from fastapi import FastAPI
from google.cloud import storage
import pickle
import pandas as pd
import os

try:
    from hotelapp.api.preprocess import preprocess_text
except ImportError:
    from api.preprocess import preprocess_text


# Create a FastAPI app
app = FastAPI()

@app.get("/")
def index():
    return {"message": "Hello, world!"}

# Create a Google Cloud Storage client and specify the name of the bucket
client = storage.Client()
storage_client = storage.Client()
bucket = storage_client.bucket("hotel-review-model")

# Get the absolute path of the current script (api/fast.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the file paths relative to the api directory
model_file_path = os.path.join(current_dir, "model.pkl")
df_file_path = os.path.join(current_dir, "df_clean.csv")

@app.on_event("startup")
async def load_model_and_dataframe():
    print("Loading model.pkl and df_clean.csv")

    # Download the model.pkl file from the bucket
    blob = bucket.blob("model.pkl")
    blob.download_to_filename(model_file_path)

    # Download the df_clean.csv file from the bucket
    blob = bucket.blob("df_clean.csv")
    blob.download_to_filename(df_file_path)

    try:
        # Load your model and dataframe here
        with open(model_file_path, "rb") as f:
            app.state.model = pickle.load(f)

        app.state.df = pd.read_csv(df_file_path)

        print("Model and dataframe loaded successfully")
    except Exception as e:
        print("Failed to load model and dataframe")
        print(e)


@app.get("/search")
def search_hotels(search_query: str, country: str):
    model = app.state.model # Get the model from the state
    df = app.state.df # Get the dataframe from the state
    """Return the list of hotels with their name, address, latitude, and longitude"""
    if df is None:
        return {"error": "DataFrame is not loaded"}  # Return an error response if the DataFrame is not loaded

    # Preprocess the search query
    preprocessed_query = preprocess_text(search_query)

    # Infer the vector representation of the preprocessed query
    query_vector = model.infer_vector(preprocessed_query.split())

    # Find the most similar vectors to the query vector
    similar_vectors = model.dv.most_similar(positive=[query_vector], topn=20)

    # Create a list with the first element of the tuples (which correspond to the index in the DF)
    sv = [doc[0] for doc in similar_vectors]

    # Retrieve the corresponding rows from the original DataFrame based on the similar vectors and make a new DF
    df_filtered = df[df.index.isin(sv)].reindex(sv)

    df_filtered_2 = df_filtered[df_filtered['country'] == country].head(20)

    # Convert the search results of the filtered DataFrame to a list of dictionaries
    # with each dictionary representing a hotel and its attributes
    dict_to_map  =  df_filtered_2.to_dict(orient="records")

    return dict_to_map
