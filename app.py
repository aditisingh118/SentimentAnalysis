
import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# Download VADER lexicon
nltk.download("vader_lexicon")

# Initialize models
vader = SentimentIntensityAnalyzer()
hf_pipeline = pipeline("sentiment-analysis")  # Hugging Face default (DistilBERT if not changed)

# Load RoBERTa model
tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# Function: VADER
def analyze_vader(text):
    scores = vader.polarity_scores(text)
    if scores["compound"] >= 0.05:
        sentiment = "Positive"
    elif scores["compound"] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, scores

# Function: RoBERTa
def analyze_roberta(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    labels = ["Negative", "Neutral", "Positive"]
    sentiment = labels[probs.argmax()]
    return sentiment, probs.detach().numpy()

# Function: Hugging Face
def analyze_huggingface(text):
    result = hf_pipeline(text)[0]
    return result["label"], result["score"]

# ---------------- Streamlit App ---------------- #
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")
st.title("📊 Sentiment Analysis Project")
st.write("Compare sentiment analysis using **VADER**, **RoBERTa**, and **Hugging Face pipeline**")

# Sidebar
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Choose Method", ["VADER (Bag of Words)", "RoBERTa", "Hugging Face Pipeline"])

# Text Input
st.header("Enter text for analysis")
user_input = st.text_area("Type or paste your text here:", "I love using Streamlit for NLP projects!")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        if choice == "VADER (Bag of Words)":
            sentiment, scores = analyze_vader(user_input)
            st.subheader(f"Sentiment: {sentiment}")
            st.json(scores)

        elif choice == "RoBERTa":
            sentiment, probs = analyze_roberta(user_input)
            st.subheader(f"Sentiment: {sentiment}")
            st.write("Probabilities:", probs)

        elif choice == "Hugging Face Pipeline":
            sentiment, score = analyze_huggingface(user_input)
            st.subheader(f"Sentiment: {sentiment}")
            st.write(f"Confidence: {score:.4f}")
