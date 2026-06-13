import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the trained model and vectorizer saved from Google Colab
try:
    model = joblib.load('emotion_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
except FileNotFoundError:
    st.error("Error: 'emotion_model.pkl' or 'vectorizer.pkl' not found. Please upload them to your GitHub repository.")

# Text cleaning function
def clean_text(text):
    text = text.lower() # Convert to lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'@[A-Za-z0-9_]+', '', text) # Remove mentions (@user)
    text = re.sub(r'[^\w\s]', '', text) # Remove special characters
    text = " ".join([word for word in text.split() if word not in stop_words]) # Remove stopwords
    return text

# Web Application User Interface (UI) Configuration
st.set_page_config(page_title="AI Emotion Detector", page_icon="🧠", layout="centered")

st.title("🧠 AI-Powered Emotion Detection from Text")
st.write("This machine learning model analyzes text inputs and predicts the underlying emotion expressed in the sentence.")

st.markdown("---")

# User Text Input Area
user_input = st.text_area(
    "Enter text/sentence to analyze:", 
    placeholder="Type something here... (e.g., I am so excited about this project!)",
    height=150
)

# Prediction Button Trigger
if st.button("Predict Emotion 🚀", use_container_width=True):
    if user_input.strip() != "":
        # 1. Clean the input text
        cleaned_text = clean_text(user_input)
        
        # 2. Transform text using the loaded vectorizer
        vec_input = vectorizer.transform([cleaned_text])
        
        # 3. Make prediction using the loaded model
        prediction = model.predict(vec_input)[0]
        
        # 4. Display the result
        st.success(f"### Predicted Emotion: **{prediction.upper()}** 🎯")
    else:
        st.warning("Please enter some text before clicking the predict button! ⚠️")

st.markdown("---")
st.caption("Built with Streamlit • Model trained using Scikit-Learn Logistic Regression")
