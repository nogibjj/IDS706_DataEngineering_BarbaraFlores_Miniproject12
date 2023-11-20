import nltk
from nltk.corpus import gutenberg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import mlflow

# Download NLTK resources
nltk.download('gutenberg')
nltk.download('punkt')

def load_data(text_id):
    return gutenberg.raw(text_id)

def preprocess_data(data):
    sentences = nltk.sent_tokenize(data)
    return sentences

def vectorize_text(text, vectorizer):
    return vectorizer.transform(text)

def predict(model, data):
    return model.predict(data)

if __name__ == "__main__":
    # Replace '<RUN_UUID>' with the actual run UUID that contains your trained model
    run_uuid = '<RUN_UUID>'
    model_path = f"runs:/{run_uuid}/model"

    # Load the model from MLflow
    model = mlflow.sklearn.load_model(model_path)

    # Load data
    chosen_text_id = 'austen-emma.txt'  # Change to your preferred Jane Austen text
    data = load_data(chosen_text_id)

    # Preprocess data
    sentences = preprocess_data(data)

    # Vectorize text using TF-IDF (make sure to use the same vectorizer as during training)
    vectorizer = TfidfVectorizer()
    X_data_tfidf = vectorize_text(sentences, vectorizer)

    # Make predictions
    predictions = predict(model, X_data_tfidf)

    # Print results
    for sentence, prediction in zip(sentences, predictions):
        print(f"Sentence: {sentence[:50]}... Prediction: {prediction}")


