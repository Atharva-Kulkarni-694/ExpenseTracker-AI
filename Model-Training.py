# model-Training.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

# Create 'model' directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Sample training data
data = pd.DataFrame({
    'description': ['Grocery store', 'Uber ride', 'Restaurant bill', 'Monthly rent', 'Movie ticket'],
    'category': ['Groceries', 'Transport', 'Food', 'Rent', 'Entertainment']
})

# Vectorize the descriptions
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['description'])
y = data['category']

# Train classifier
model = MultinomialNB()
model.fit(X, y)

# Save model and vectorizer
with open('model/expense_classifier.pkl', 'wb') as f:
    pickle.dump((model, vectorizer), f)

print("Model trained and saved successfully.")

# This code trains a simple Naive Bayes classifier to categorize expenses based on their descriptions.
# You can expand the training data with more examples for better accuracy.