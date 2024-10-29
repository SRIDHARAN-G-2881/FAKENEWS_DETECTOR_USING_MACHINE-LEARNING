import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Title and description
st.title("Fake News Detection")
st.write("Enter a news article to determine if it's real or fake.")

# Text input
news_article = st.text_area("Enter the news article", "")

# Load the pre-trained model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("fake_news_vectorizer.pkl", "rb"))

# Fake news detection function
def detect_fake_news(article):
    # Vectorize the input article
    article_vector = vectorizer.transform([article])

    # Make a prediction using the trained model
    prediction = model.predict(article_vector)[0]
    
    return prediction

# Detect button
if st.button("Detect"):
    if news_article:
        # Perform fake news detection
        prediction = detect_fake_news(news_article)

        # Display the prediction
        if prediction == 0:
            st.success("The news article is likely real.")
        else:
            st.error("The news article is likely fake.")
    else:
        st.warning("Please enter a news article.")