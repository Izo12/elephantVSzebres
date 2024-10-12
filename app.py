import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore
from tensorflow.keras.applications.efficientnet import preprocess_input  # type: ignore

# Configuration de la page
st.set_page_config(
    page_title="Classification Éléphant vs Zèbre",
    page_icon="🐘🦓",
    layout="centered",
    initial_sidebar_state="auto",
)

# Fonction pour ajouter du CSS personnalisé
def local_css(css_code):
    st.markdown(f'<style>{css_code}</style>', unsafe_allow_html=True)

# CSS personnalisé
local_css("""
/* Importation de la police depuis Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

/* Application de la police à l'ensemble de l'application */
html, body, [class*="css"]  {
    font-family: 'Roboto', sans-serif;
}

/* Style du titre principal */
h1 {
    color: #2c3e50;
    text-align: center;
    animation: fadeInDown 1s;
}

/* Style du texte */
p, label {
    color: #34495e;
    font-size: 1.1em;
}

/* Animation pour le titre */
@keyframes fadeInDown {
    from {
        opacity: 0;
        transform: translateY(-50px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Bouton personnalisé */
.stButton > button {
    background-color: #3498db;
    color: white;
    padding: 0.5em 2em;
    border-radius: 5px;
    border: none;
    font-size: 1.1em;
    font-weight: bold;
    cursor: pointer;
    transition: background-color 0.3s ease;
}

.stButton > button:hover {
    background-color: #2980b9;
}

/* Style de l'image */
img {
    border: 5px solid #ecf0f1;
    border-radius: 10px;
    margin-top: 20px;
    max-width: 100%;
    height: auto;
    animation: fadeInUp 1s;
}

/* Animation pour l'image */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(50px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Style pour les messages de succès et d'information */
.css-1fv8s86 {
    background-color: #2ecc71 !important;
}

.css-1d391kg {
    background-color: #3498db !important;
}
""")

# Titre de l'application avec emojis
st.markdown("<h1> 🐘 Éléphant vs Zèbre 🦓</h1>", unsafe_allow_html=True)

# Description
st.write("""
         Cette application utilise un modèle de deep learning pour classer les images en tant qu'**Éléphant** ou **Zèbre**.
         Veuillez télécharger une image pour obtenir une prédiction.
         """)

# Chargement du modèle avec mise en cache
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('best_pretrained_model.keras')
    return model

model = load_model()

# Fonction de prédiction
def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img_array = img_to_array(image)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction

# Téléchargement de l'image
file = st.file_uploader("Veuillez télécharger une image", type=["jpg", "jpeg", "png"])

if file is not None:
    image = Image.open(file)

    # Afficher l'image téléchargée avec style
    st.image(image, caption='Image téléchargée', use_column_width=True)

    # Bouton pour lancer la prédiction
    if st.button('Classer l\'image'):
        st.write("Classification en cours...")
        with st.spinner('Analyse de l\'image...'):
            prediction = import_and_predict(image, model)
            probability = prediction[0][0]
            threshold = 0.5
            if probability > threshold:
                predicted_class = 'Zèbre 🦓'
                confidence = probability * 100
            else:
                predicted_class = 'Éléphant 🐘'
                confidence = (1 - probability) * 100
        st.success(f"**Prédiction : {predicted_class}**")
        st.info(f"**Confiance : {confidence:.2f}%**")
else:
    # Afficher un message invitant l'utilisateur à télécharger une image
    st.write("Veuillez télécharger une image en utilisant le bouton ci-dessus.")
