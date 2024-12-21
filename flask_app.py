from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

# Initialize Flask app
app = Flask(__name__)

# Load device, model, and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name =  "sentence-transformers/all-mpnet-base-v2" # "sentence-transformers/all-MiniLM-L6-v2", "google-bert/bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# Load preprocessed data and embeddings
with open('trained_model.pkl', 'rb') as file:
    text_embeddings, data = pickle.load(file)

def compute_embeddings(texts, model, tokenizer, device, batch_size=64):
    """Compute embeddings for given texts using a pre-trained model."""
    embeddings = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

@app.route('/')
def home():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """Provide recommendations based on user input."""
    user_query = request.json.get('query')
    if not user_query:
        return jsonify({"error": "No query provided."}), 400

    # Compute embedding for user query
    query_embedding = compute_embeddings([user_query], model, tokenizer, device)

    # Calculate similarity scores
    similarity_scores = cosine_similarity(query_embedding, text_embeddings)
    top_indices = np.argsort(similarity_scores[0])[::-1][:3]
    recommendations = data.iloc[top_indices]

    # Prepare response with recommendations and scores
    results = []
    for idx, row in recommendations.iterrows():
        results.append({
            "productName": row['productName'],
            "brandName": row['brandName'],
            "price": row['price'],
            "description": row['description'],
            "similarity_score": float(similarity_scores[0][idx])
        })

    return jsonify({"recommendations": results})

if __name__ == '__main__':
    app.run(debug=True)