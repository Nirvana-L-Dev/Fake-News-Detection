import streamlit as st
import pickle

# Load pre-trained model and vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Function to clean text (Ensure this function is defined in your script)
def clean_text(text):
    return text.lower()  # Simple example, replace with actual preprocessing

# Function to predict news type
def predict_news(text):
    text = clean_text(text)  # Apply same cleaning function
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    return "Fake News" if prediction == 0 else "Real News"

# Streamlit App UI
st.title("📰 Fake News Detector")

user_input = st.text_area("Enter news article text:")
if st.button("Check"):
    result = predict_news(user_input)
    st.write("Prediction:", result)

from sklearn.feature_extraction.text import TfidfVectorizer

# Assuming you have preprocessed training data
vectorizer = TfidfVectorizer()
vectorizer.fit(training_texts)  # Replace `training_texts` with your data

# Save the vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

VECTOR_PATH = r"C:\Users\APPLE\Documents\Soumya\Data Analyst\Fake News Detection\vectorizer.pkl"

with open(VECTOR_PATH, "rb") as f:
    vectorizer = pickle.load(f)

