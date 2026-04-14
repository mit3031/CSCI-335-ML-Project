import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

column_names = ['id', 'entity', 'sentiment', 'content']
df = pd.read_csv('data/twitter_training.csv', names=column_names)
df = df.dropna(subset=['content'])

# Cleans the tweets of url's, handles, and other things
print("Cleaning tweets (Task 2.1)...")
def clean_tweet(text):
    text = str(text).lower()
    # Remove URL's
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove @
    text = re.sub(r'\@\w+', '', text)
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    return text


df['cleaned_content'] = df['content'].apply(clean_tweet)

# Training of TF IDF
print("Splitting data and running TF-IDF")
# using 80 percent of the data for training and the rest for testing
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_content'], df['sentiment'], test_size=0.2, random_state=42
)

# Convert text to numbers using TF IDF
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Classic ML models below
print("Training Support Vector Machine")
svm_model = LinearSVC()
svm_model.fit(X_train_vec, y_train)

print("Training Naive Bayes Model")
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

# Print results
print("\n--- Generating Report ---")

# Evaluate SVM
svm_predictions = svm_model.predict(X_test_vec)
print("\nSupport Vector Machine Results:")
print(classification_report(y_test, svm_predictions))

# Evaluate Naive Bayes
nb_predictions = nb_model.predict(X_test_vec)
print("\nNaive Bayes Results:")
print(classification_report(y_test, nb_predictions))