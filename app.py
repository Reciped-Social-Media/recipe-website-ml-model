import pandas as pd
import numpy as np
import csv
import nltk
import joblib
from gensim.models import Word2Vec
from collections import defaultdict
from annoy import AnnoyIndex
from flask import Flask, request, jsonify

nltk.download("wordnet")
nltk.download("punkt")


class TfidfEmbedding(object):
    def __init__(self, model_w2v: Word2Vec):
        self.model_w2v = model_w2v
        self.word_idf_weight = None
        self.vector_size = model_w2v.wv.vector_size

    def fit(self):
        tfidf = joblib.load(
            "./models/tfidf.pkl"
        )  # creates the idf dictionary for all words in the corpus
        max_idf = max(
            tfidf.idf_
        )  # if a word was never seen it is given idf of the max of known idf value
        self.word_idf_weight = defaultdict(
            lambda: max_idf,
            [(word, tfidf.idf_[i]) for word, i in tfidf.vocabulary_.items()],
        )
        return self

    # converts a list of recipes to a list of document vectors
    def transform(self, recipes: list):
        doc_word_vector = self.doc_average_list(recipes)
        return doc_word_vector

    # retruns the document embedding as a weighted average
    def doc_average(self, recipe: list):
        mean = []
        for word in recipe:
            if word in self.model_w2v.wv.index_to_key:
                mean.append(
                    self.model_w2v.wv.get_vector(word) * self.word_idf_weight[word]
                )

        if not mean:
            return np.zeros(self.vector_size)
        else:
            mean = np.array(mean).mean(axis=0)
            return mean

    # returns the full list of all document embeddings
    def doc_average_list(self, recipes):
        return np.vstack([self.doc_average(recipe) for recipe in recipes])


def get_recommendations(
    input,
    N=5,
    chef=False,
) -> list:
    # Input is a list of ingredients if chef is true , else it is a recipe_id
    if chef:
        input.sort()
        input_embedding = fitted_tfidf.transform([input])[0].reshape(1, -1)[0]
    else:
        if recipe_df.index[recipe_df["id"] == input].tolist() == []:
            return []
        index = recipe_df.index[recipe_df["id"] == input].tolist()[0]
        print(index)
        input_embedding = corpus_embeddings[index]
        input_embedding = [float(val) for val in input_embedding]

    return annoy_index.get_nns_by_vector(input_embedding, N)


# Read in annoy index
annoy_index = AnnoyIndex(100, "angular")
annoy_index.load("./models/annoy_index.ann")

# Read recipe dataframe
recipe_df = pd.read_pickle("recipes_df.pkl")

# Read in word2vec model
model_w2v = Word2Vec.load("./models/model_w2v.bin")

# Instanitate embedding model
tfidf = TfidfEmbedding(model_w2v)
fitted_tfidf = tfidf.fit()

# Read in embeddings
print("Loading embeddings...")
with open("embed.csv", "r", newline="") as file:
    reader = csv.reader(file)
    corpus_embeddings = list(reader)
print("Embeddings loaded...")


app = Flask(__name__)


@app.route("/", methods=["GET"])
def welcome():
    return "Hello World!"


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    recipe_input = data.get("input", -1)
    print(recipe_input)
    chef = data.get("chef", -1)
    recommended_ids = []
    if recipe_input != -1:
        for input in recipe_input:
            rec_ids = get_recommendations(input=input, chef=chef)
            reco_ids = [int(recipe_df.loc[id, "id"]) for id in rec_ids]
            if not chef:
                if reco_ids != []:
                    reco_ids = reco_ids[1:2][0]
                    recommended_ids.append([input, reco_ids])
            else:
                recommended_ids.append(reco_ids)
        return jsonify({"recommendations": recommended_ids})
    else:
        return jsonify({"error": "Invalid input query"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=105)
