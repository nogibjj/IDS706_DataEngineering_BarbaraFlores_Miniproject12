import nltk
from nltk.corpus import gutenberg
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download('gutenberg')
nltk.download('punkt')
gutenberg_ids = gutenberg.fileids()
chosen_text_id = 'austen-emma.txt'  # Cambia a tu texto de Jane Austen preferido

def load_data(text_id):
    return gutenberg.raw(text_id)

data = load_data(chosen_text_id)

# Preprocesamiento de datos (en este caso, simplemente dividimos el texto en oraciones)
sentences = nltk.sent_tokenize(data)

# Creación de etiquetas de ejemplo (clasificación de texto ficticia)
# En este caso, etiquetamos las oraciones que contienen el nombre "Emma"
labels = [1 if 'Emma' in sentence else 0 for sentence in sentences]

# División de datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2, random_state=42)

# Vectorización de texto utilizando TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Entrenamiento del modelo (en este caso, un clasificador Naive Bayes)
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Predicciones en el conjunto de prueba
y_pred = classifier.predict(X_test_tfidf)

# Evaluación del modelo
accuracy = accuracy_score(y_test, y_pred)

# Imprimir la precisión del modelo
print(f"Accuracy: {accuracy}")

# Lograr métricas en MLflow
import mlflow
with mlflow.start_run():
    mlflow.log_param("chosen_text_id", chosen_text_id)
    mlflow.log_metric("accuracy", accuracy)
