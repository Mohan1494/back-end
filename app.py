from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import joblib
import os

# Load the vectorizer and the model
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

@app.route('/analyze', methods=['POST'])
def analyze():
    # Get data from the request
    data = request.json
    text = data.get('text')

    if not text:
        return jsonify({'error': 'No input provided'}), 400

    # Preprocess the input using the vectorizer
    transformed_input = vectorizer.transform([text])

    # Predict sentiment using the model
    prediction = model.predict(transformed_input)

    # Assuming the model returns classes like 0 (negative), 1 (neutral), 2 (positive)
    sentiment = ''
    if prediction[0] == 0:
        sentiment = 'negative'
    elif prediction[0] == 1:
        sentiment = 'neutral'
    elif prediction[0] == 2:
        sentiment = 'positive'

    # Return prediction
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
