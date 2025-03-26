import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

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

print("Data Preprocessing Done! Training samples:", X_train.shape[0], "Testing samples:", X_test.shape[0])

# Step 5: Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)  # Training the model

# Step 6: Test on unseen data
y_pred = model.predict(X_test)  # Predict sentiment of test reviews

# Step 7: Measure accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# --------------------------------------------------
# 🚀 Step 8: Test with Custom Input
def predict_sentiment(text):
    text_vector = vectorizer.transform([text])  # Convert input text to numerical format
    prediction = model.predict(text_vector)[0]
    return "Positive 😊" if prediction == 1 else "Negative 😞"

# Test on custom inputs
print(predict_sentiment("The movie was different,I was sleeping half of the time."))
print(predict_sentiment("the storyline was engaging so it saved the movie The acting was mediocre."))

