import nltk
from nltk.corpus import gutenberg
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# Start MLflow run
mlflow.start_run()

# Download NLTK resources
nltk.download('gutenberg')
nltk.download('punkt')

# Choose a text by Jane Austen
gutenberg_ids = gutenberg.fileids()
chosen_text_id = 'austen-emma.txt'  # Change to your preferred Jane Austen text

# Load data
def load_data(text_id):
    return gutenberg.raw(text_id)

data = load_data(chosen_text_id)

# Data preprocessing (in this case, simply splitting the text into sentences)
sentences = nltk.sent_tokenize(data)

# Creating example labels (dummy text classification)
# In this case, label sentences containing the name "Emma"
labels = [1 if 'Emma' in sentence else 0 for sentence in sentences]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2, random_state=42)

# Text vectorization using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model training (in this case, a Naive Bayes classifier)
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Predictions on the test set
y_pred = classifier.predict(X_test_tfidf)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)

# Print model accuracy
print(f"Accuracy: {accuracy}")

# Log parameters and metrics in MLflow
mlflow.log_param("chosen_text_id", chosen_text_id)
mlflow.log_metric("accuracy", accuracy)

# End MLflow run
mlflow.end_run()
