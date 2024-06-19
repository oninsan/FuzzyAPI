import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import StandardScaler
from category_encoders import BinaryEncoder
from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.stem import WordNetLemmatizer # type: ignore
from nltk.corpus import stopwords # type: ignore
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical # type: ignore
import re
import string

# load the model
model = load_model('inside_out7.pb')

# load the label references
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')

label_encoder = LabelEncoder()
y_train_resampled_encoded = label_encoder.fit_transform(y_train)
y_train_resampled_categorical = to_categorical(y_train_resampled_encoded)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# create a victorizer/tokenizer
vectorizer = TfidfVectorizer(max_features=5000)
vectorizer.fit(X_train['posts'])

api = Flask(__name__)
CORS(api, methods=["POST"], allow_headers=["Content-Type"])

@api.route('/api/insideout/predictemotion', methods=['POST'])
def predict_emotion():
    data = request.get_json()['answers']
    
    # data transformation from client
    answers = []
    for entry in data:
        answers.append(entry['answer'])
    answers_df = pd.DataFrame(answers)
    answers_df.columns = ['posts']
    answers_df.posts = clear_text(answers_df['posts'])
    print(answers_df.posts)
    # Vectorize the answers
    final_tfidf = vectorizer.transform(answers_df['posts'])

    # Predictions
    preds = model.predict(final_tfidf.toarray())
    final_preds = np.argmax(preds, axis=1)
    final_results = []
    results = label_encoder.inverse_transform(final_preds)

    # joy
    count_joy = np.count_nonzero((np.array(results) == 'ESFP') | (np.array(results) == 'ENFP')) / len(results) * 100
    final_results.append({"Joy": count_joy})

    # sadness
    count_sadness = np.count_nonzero((np.array(results) == 'INFP') | (np.array(results) == 'INFJ')) / len(results) * 100
    final_results.append({"Sadness": count_sadness})

    # anger
    count_anger = np.count_nonzero((np.array(results) == 'ENTJ') | (np.array(results) == 'ESTJ')) / len(results) * 100
    final_results.append({"Anger": count_anger})

    # fear
    count_fear = np.count_nonzero((np.array(results) == 'ISFJ') | (np.array(results) == 'ISTJ')) / len(results) * 100
    final_results.append({"Fear": count_fear})

    # disgust
    count_disgust = np.count_nonzero((np.array(results) == 'INTJ') | (np.array(results) == 'ISTP')) / len(results) * 100
    final_results.append({"Disgust": count_disgust})

    print(final_results)
    return jsonify(final_results)

# for clearing all unnecessary charactes and leading spaces
def clear_text(data):
    data_length = []
    cleaned_text = []
    for sentence in tqdm(data):
        sentence = sentence.lower()
        # Remove URLs
        sentence = re.sub(r'https?://[^\s<>"]+|www\.[^\s<>"]+', ' ', sentence)
        # Remove non-alphanumeric characters
        sentence = re.sub(r'[^0-9a-z]', ' ', sentence)
        # Tokenize text
        words = word_tokenize(sentence)
        # Remove stop words and lemmatize
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        # Join words back into a single string
        cleaned_sentence = ' '.join(words)
        data_length.append(len(cleaned_sentence.split()))
        cleaned_text.append(cleaned_sentence)
    return cleaned_text

if __name__ == '__main__':
    api.run(port=5000)