import nltk
import tkinter as tk
from tkinter import messagebox
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Step 1: Download dataset (Only required once)
nltk.download("movie_reviews")

# Step 2: Load movie reviews data
docs = [" ".join(movie_reviews.words(fileid)) for fileid in movie_reviews.fileids()]
labels = [1 if fileid.startswith("pos") else 0 for fileid in movie_reviews.fileids()]  # 1 = Positive, 0 = Negative

# Step 3: Convert text into numbers using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)  # Converts text into word frequency counts

# Step 4: Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Step 5: Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)  # Training the model

# Step 6: Define function to predict sentiment
def predict_sentiment():
    user_text = entry.get()  # Get text from input box
    if not user_text.strip():
        messagebox.showwarning("Input Error", "Please enter some text!")
        return
    
    text_vector = vectorizer.transform([user_text])  # Convert input text to numerical format
    prediction = model.predict(text_vector)[0]
    
    result = "Positive 😊" if prediction == 1 else "Negative 😞"
    messagebox.showinfo("Sentiment Result", f"Sentiment: {result}")

# Step 7: Build GUI with Tkinter
root = tk.Tk()
root.title("Sentiment Analysis")

# UI Elements
tk.Label(root, text="Enter Text for Sentiment Analysis:", font=("Arial", 12)).pack(pady=10)
entry = tk.Entry(root, width=50, font=("Arial", 14))
entry.pack(pady=5)
tk.Button(root, text="Analyze Sentiment", command=predict_sentiment, font=("Arial", 12), bg="blue", fg="white").pack(pady=10)

# Run GUI
root.mainloop()
