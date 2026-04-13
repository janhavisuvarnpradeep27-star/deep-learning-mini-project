import streamlit as st
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Set page config
st.set_page_config(page_title="Sentiment Analysis App", page_icon="🧠", layout="wide")

# Load models and tokenizer
try:
    model_data = joblib.load("models/logistic_model.pkl")
    log_model = model_data['model']
    tfidf_vectorizer = model_data['vectorizer']
except FileNotFoundError:
    st.error("Logistic Regression model not found. Please make sure 'models/logistic_model.pkl' exists.")
    log_model = None
    tfidf_vectorizer = None

try:
    lstm_model = load_model("models/lstm_model.h5")
except (FileNotFoundError, IOError):
    st.error("LSTM model not found. Please make sure 'models/lstm_model.h5' exists.")
    lstm_model = None

try:
    with open("models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
except FileNotFoundError:
    st.error("Tokenizer not found. Please make sure 'models/tokenizer.pkl' exists.")
    tokenizer = None

# This is a placeholder. In a real-world scenario, you would save and load the fitted TfidfVectorizer.
# For this example, we'll create a new one.
# tfidf_vectorizer = TfidfVectorizer(max_features=5000) # Assuming max_features used during training

# --- UI Layout ---
st.title("🧠 Sentiment Analysis App")
st.markdown("This application performs sentiment analysis on user-provided text using either a Logistic Regression or an LSTM model.")

st.markdown("---")

# --- Sidebar for model selection ---
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox("Choose a model for prediction:", ["Logistic Regression", "LSTM"])

# --- SECTION 1: User Input Prediction ---
st.header("🔎 Predict Sentiment from Text")
user_input = st.text_area("Enter your review here:", height=150)

if st.button("Predict Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        if model_choice == "Logistic Regression":
            if log_model and tfidf_vectorizer:
                # This is a simplified approach. For accurate results, the same TF-IDF vectorizer
                # fitted on the training data should be used.
                # Since we are creating a new one, we fit it on the user input.
                # This is not ideal but works for a demonstration.
                try:
                    # In a real app, you would just transform:
                    text_features = tfidf_vectorizer.transform([user_input])
                    # For now, we fit and transform.
                    prediction = log_model.predict(text_features)[0]
                    if prediction == 1:
                        st.success("Positive 😊")
                    else:
                        st.error("Negative 😡")
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

        elif model_choice == "LSTM":
            if lstm_model and tokenizer:
                try:
                    # Tokenize and pad the sequence
                    sequence = tokenizer.texts_to_sequences([user_input])
                    padded_sequence = pad_sequences(sequence, maxlen=100)
                    
                    # Predict
                    prediction_prob = lstm_model.predict(padded_sequence)[0][0]
                    
                    # Display result
                    if prediction_prob > 0.5:
                        st.success("Positive 😊")
                    else:
                        st.error("Negative 😡")
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

st.markdown("---")

# --- SECTION 2: Model Performance ---
st.header("📊 Model Performance")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Accuracy Graph")
    try:
        st.image("results/accuracy_comparison.png", caption="Model Accuracy Comparison", use_column_width='always')
    except FileNotFoundError:
        st.warning("Accuracy graph not found at 'results/accuracy_comparison.png'.")

with col2:
    st.subheader("Confusion Matrix")
    try:
        st.image("results/confusion_matrix_logistic.png", caption="Model Confusion Matrix", use_column_width='always')
    except FileNotFoundError:
        st.warning("Confusion matrix not found at 'results/confusion_matrix_logistic.png'.")

st.markdown("---")

# --- SECTION 3: Data Insights ---
st.header("📈 Data Insights")
st.subheader("Sentiment Distribution in Dataset")
try:
    st.image("results/sentiment_distribution.png", caption="Distribution of Sentiments", use_column_width='always')
except FileNotFoundError:
    st.warning("Sentiment distribution graph not found at 'results/sentiment_distribution.png'.")

st.markdown("---")
st.info("App created by an AI assistant.")
