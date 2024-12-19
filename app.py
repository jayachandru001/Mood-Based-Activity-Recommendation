from flask import Flask, request, redirect, jsonify
import gensim.downloader as api
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine


app = Flask(__name__)

# Load the dataset
dataset_path = "./Mood_Recommendations_Dataset.csv"
data = pd.read_csv(dataset_path)

# Load pre-trained word vectors (GloVe or Word2Vec)
# word_vectors = api.load("glove-wiki-gigaword-100")  # Example: GloVe vectors

# Load pre-installed vectors from a local file
word_vectors = KeyedVectors.load("glove-wiki-gigaword-100.kv", mmap='r')

# Function to compute soft cosine similarity
def soft_cosine_similarity(text1, text2, word_vectors):
    # Tokenize the text (split into words and remove stopwords)
    text1_tokens = text1.lower().split()
    text2_tokens = text2.lower().split()

    # Filter tokens that are in the word vectors' vocabulary
    valid_tokens1 = [token for token in text1_tokens if token in word_vectors]
    valid_tokens2 = [token for token in text2_tokens if token in word_vectors]

    if not valid_tokens1 or not valid_tokens2:
        return 0.0  # Return 0 if no valid tokens for comparison

    # Create the vector representations for each text based on word vectors
    vector1 = sum(word_vectors[token] for token in valid_tokens1) / len(valid_tokens1)
    vector2 = sum(word_vectors[token] for token in valid_tokens2) / len(valid_tokens2)

    # Compute cosine similarity between the two vectors
    similarity = 1 - cosine(vector1, vector2)
    return similarity

# Function to recommend activities using soft cosine similarity
def recommend_activities(user_message, mood, top_n=5):
    # Filter dataset by mood
    mood_data = data[data['Mood'].str.lower() == mood.lower()]
    # Combine Recommendation and Details for text matching
    mood_data['Combined_Text'] = mood_data['Recommendation'] + " " + mood_data['Details']

    # Calculate similarity scores using soft cosine similarity
    similarities = mood_data['Combined_Text'].apply(lambda x: soft_cosine_similarity(user_message, x, word_vectors))

    # Add similarity scores
    mood_data = mood_data.copy()
    mood_data['Similarity'] = similarities

    # Get top N unique recommendations based on similarity
    top_recommendations = (
        mood_data[['Mood', 'Category', 'Recommendation', 'Details', 'Similarity']]
        .drop_duplicates(subset=['Recommendation'])
        .nlargest(top_n, 'Similarity')
    )

    return top_recommendations[['Mood', 'Category', 'Recommendation', 'Details','Similarity']]


@app.route('/mood-recommendations-api', methods= ['POST'])
def moodRecommendation():
    try:
        data = request.get_json()
        user_mood = data['user_mood']
        user_message = data['user_message']
        
        top_activities = recommend_activities(user_message, user_mood)
        top_activities_data = top_activities.to_json(orient='records')
        recomendation_response  = {"result":"Successful", "response_data": top_activities_data[1:-1]}
    except Exception as error:
        recomendation_response = "An Error Occured " + str(error)
        
    return jsonify(recomendation_response)

if __name__ == '__main__':
    # for production use
    app.run(host='0.0.0.0', port=5000, debug=False)
    
    # for developer use
    # app.run(debug=True)