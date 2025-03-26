import nltk
import streamlit as st
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Step 1: Download dataset
nltk.download("movie_reviews")

# Step 2: Load movie reviews data
docs = [" ".join(movie_reviews.words(fileid)) for fileid in movie_reviews.fileids()]
labels = [1 if fileid.startswith("pos") else 0 for fileid in movie_reviews.fileids()]

# Step 3: Convert text into numerical format
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)

# Step 4: Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Step 5: Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 6: Streamlit Web App
st.title("Sentiment Analysis Web App 🚀")
st.write("Enter a review to analyze its sentiment.")

user_input = st.text_area("Enter your text here:")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        text_vector = vectorizer.transform([user_input])
        prediction = model.predict(text_vector)[0]
        result = "Positive 😊" if prediction == 1 else "Negative 😞"
        st.success(f"Sentiment: {result}")
    else:
        st.warning("Please enter some text!")
