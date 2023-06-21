# Hotel Reviews App

Welcome to the Hotel Reviews App! This application utilizes Gensim's Doc2Vec, FastAPI, nltk.tokenizer, and stopwords to provide users with summarized hotel reviews based on their specified country preference.

## How It Works

1. User Input: The user provides a brief summary of their preferences, alongside their country of preference.

2. Word Tokenization: The user's input is processed through nltk.tokenizer, which breaks down the input into individual words.

3. Stopword Removal: Stopwords, which are common words that do not carry significant meaning in the context of natural language processing, are filtered out from the tokenized words. This helps improve the accuracy of the analysis.

4. Doc2Vec Embeddings: Gensim's Doc2Vec algorithm is used to transform the tokenized words into numerical representations called embeddings. These embeddings capture the semantic meaning of the words and allow for comparison and similarity calculation.

5. Review Similarity Calculation: The embeddings of the user's input are compared with the embeddings of hotel reviews from the specified country. This calculates the similarity between the user's preference and each hotel's summarized reviews.

6. Result Generation: Based on the calculated similarities, a list of hotels is generated, sorted by their closeness to the user's preference. Each hotel includes a summary of its reviews, providing a overview of all the reviews it has received.
