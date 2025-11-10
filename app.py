# Importation des bibliothèques nécessaires
from flask import Flask, render_template, request, session, redirect, url_for

# Bibliothèques standard
import numpy as np
import os
import re

# Pour la gestion des accents
import unicodedata

# Pour la sérialisation des objets
import pickle

# Pandas pour la manipulation des données
import pandas as pd

# TensorFlow et Keras pour le modèle de deep learning
from tensorflow import keras
from tensorflow.keras import layers

# Scikit-learn pour le prétraitement et l'évaluation
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix


# NLTK pour le traitement du langage naturel
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


# Initialisation de l'application Flask
app = Flask(__name__)
app.secret_key = '9bcddfd4d2cb02dd798a9a990f97595cd3e3872c2865e7cf'

# Ressources nécessaires du NLTK
try:
    _ = stopwords.words("french")
    _ = wordnet.synsets("test")
except Exception:
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    nltk.download("punkt")

# Prétraitement du texte
lemmatizer = WordNetLemmatizer()
french_stopwords = set(stopwords.words("french"))

def clean_text(text):
    """Nettoyage basique : minuscules, normalisation des accents, tokenisation, suppression des stopwords, lemmatisation."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Normaliser les accents
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    # Garder lettres, apostrophes et traits d'union
    text = re.sub(r"[^a-z0-9'\-\s]", " ", text)
    # Tokenisation et suppression des stopwords
    tokens = [t for t in re.split(r"\s+", text) if t and t not in french_stopwords]
    # Lemmatisation
    lemma = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(lemma).strip()

# Chargement des données d'entraînement
df = pd.read_csv("training_data.csv")
df.dropna(inplace=True)  # Nettoyage de colonnes vides
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Mélange aléatoire des données

# Récupération des listes questions / réponses
questions = [clean_text(q) for q in df["question"].tolist()]
reponses = df["reponse"].tolist()

# Encodage des labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(reponses)

# Tokenisation
tokenizer_path = "tokenizer.pkl"
if os.path.exists(tokenizer_path):
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
else:
    tokenizer = keras.preprocessing.text.Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(questions)
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)


# Séquences d'entraînement
train_sequences = tokenizer.texts_to_sequences(questions)
max_len = max(len(s) for s in train_sequences) if train_sequences else 20
train_sequences = keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=max_len, padding="post")

# TF-IDF vectorizer pour similarité sémantique 
tfidf_path = "tfidf.pkl"
if os.path.exists(tfidf_path):
    with open(tfidf_path, "rb") as f:
        tfidf = pickle.load(f)
else:
    tfidf = TfidfVectorizer()
    tfidf.fit(questions)
    with open(tfidf_path, "wb") as f:
        pickle.dump(tfidf, f)

# Matrice TF-IDF des questions
tfidf_matrix = tfidf.transform(questions)

# Modèle
def Construction_Model(vocab_size, embed_dim=128, input_length=100, n_classes=None):
    """Crée un modèle simple avec une couche d'embedding et une couche LSTM."""
    inputs = layers.Input(shape=(input_length,))
    # Couche de conversion de séquences en embeddings
    x = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=input_length)(inputs)
    # Couche LSTM pourle traitement séquentiel
    x = layers.LSTM(128)(x)
    # Couche de sortie avec activation softmax pour classification multi-classes
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    # Compilation du modèle
    model = keras.Model(inputs=inputs, outputs=outputs)
    # Compilation du model avec la fonction de perte, l'optimiseur et des metiques
    model.compile(optimizer='adam', #Utilisation de l'optimiseur Adam
                   loss='sparse_categorical_crossentropy', # Fonction de perte
                     metrics=['accuracy']) # metrique de precision
    return model

        # vocab_size = le nombre total de mots connus par le tokenizer
        # embed_dim = dimension des vecteurs d'embedding(par défaut 128)
        # input_length = longueur maximale des séquences d'entrée
        #n_classes = le nombre de classes de sortie (réponses uniques)


model_path = "chatbot_model.h5"
label_encoder_path = "label_encoder.pkl"

if os.path.exists(model_path) and os.path.exists(label_encoder_path):
    print("Chargement du modèle existant...")
    model = keras.models.load_model(model_path)
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
else:
    print("🔧 Entraînement du modèle (nouveau)...")
    vocab_size = len(tokenizer.word_index) + 1
    model = Construction_Model(vocab_size=vocab_size, input_length=max_len, n_classes=len(label_encoder.classes_))
    callbacks = [
        # Arrêt précoce et sauvegarde du meilleur modèle
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', verbose=1)
    ]
    history = model.fit(
        train_sequences,
        encoded_labels,
        epochs=50,
        batch_size=8,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    # Sauvegarde de label_encoder
    with open(label_encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    # Sauvegarde de tokenizer et tfidf
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    with open(tfidf_path, "wb") as f:
        pickle.dump(tfidf, f)
    # Sauvegarde de log d'entraînement
    try:
        pd.DataFrame(history.history).to_csv("training_log.csv", index=False)
    except Exception:
        pass

# Génération de réponse
def semantic_fallback(user_text, threshold=0.75):
    """Retourne la réponse basée sur la similarité sémantique."""
    cleaned = clean_text(user_text)
    v = tfidf.transform([cleaned])
    sims = cosine_similarity(v, tfidf_matrix).flatten()
    best_idx = np.argmax(sims)
    best_score = sims[best_idx]
    if best_score >= threshold:
        return reponses[best_idx], float(best_score)
    return None, float(best_score)

def generate_response(user_text):
    """Génère une réponse en utilisant soit le modèle soit la similarité sémantique."""
    sem_res, sem_score = semantic_fallback(user_text, threshold=0.66)
    if sem_res:
        return sem_res, sem_score, "tfidf"

    cleaned = clean_text(user_text)
    seq = tokenizer.texts_to_sequences([cleaned])
    seq = keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len, padding="post")
    pred = model.predict(seq)
    pred_label = np.argmax(pred, axis=1)[0]
    confidence = float(np.max(pred))
    if confidence < 0.45:
        return "Je ne suis pas sûr de bien comprendre, peux-tu reformuler ?", confidence, "low_conf"
    else:
        return label_encoder.inverse_transform([pred_label])[0], confidence, "model"

# Routes vers l'application web



@app.route('/', methods=['GET', 'POST'])
def index():
    """Gestion de la page d'accueil et des interactions utilisateur."""
    if 'conversation' not in session:
        session['conversation'] = []

    if request.method == 'POST':
        user_input = request.form.get('input', '').strip()
        if user_input:
            response_text, score, source = generate_response(user_input)
            session['conversation'].append({
                'user': user_input,
                'bot': response_text,
                'score': round(score, 3) if isinstance(score, float) else score,
                'source': source
            })
        session.modified = True 

    return render_template('index.html', conversation=session['conversation'])

@app.route('/reset')
def reset():
    """Réinitialise la conversation."""
    session.pop('conversation', None)
    return redirect(url_for('index'))

@app.route('/metrics')
def metrics():
    """Affiche les logs d'entraînement si disponibles."""
    if os.path.exists("training_log.csv"):
        df = pd.read_csv("training_log.csv")
        html = df.to_html(classes="table table-striped", index=False)
        return render_template('metrics.html', table=html)
    return "Aucun log d'entraînement trouvé."


#  AFFICHAGE DES PERFORMANCES DU MODÈLE
@app.route('/performance')
def performance():
    """Affiche accuracy, recall et F1-score dans une page Flask (HTML)."""
    try:
        # Prédictions sur les données d'entraînement
        y_true = encoded_labels
        y_pred_probs = model.predict(train_sequences)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Rapport de classification (en dictionnaire)
        report = classification_report(y_true, y_pred, output_dict=True)

        # Calcul des métriques globales (moyenne pondérée)
        accuracy = accuracy_score(y_true, y_pred) * 100
        recall = report["weighted avg"]["recall"] * 100
        f1 = report["weighted avg"]["f1-score"] * 100

        # Envoi vers la page HTML
        return render_template(
            'performance.html',
            accuracy=round(accuracy, 2),
            recall=round(recall, 2),
            f1=round(f1, 2)
        )

    except Exception as e:
        return f"Erreur lors du calcul des performances : {e}"


@app.route('/check_session')
def check_session():
    """Vérifie l'état de la session."""
    return str(session.get('conversation', []))





# Lancement de l'application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
