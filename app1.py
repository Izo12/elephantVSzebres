from flask import Flask, render_template, request, redirect, url_for, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'votre_clé_secrète'  # Nécessaire pour utiliser la session
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Charger le modèle pré-entraîné
model = load_model('best_pretrained_model.keras')

# Vérifier le mappage des classes
class_indices = {'elephants': 0, 'zebras': 1}
index_to_class = {v: k for k, v in class_indices.items()}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'Charger l\'image':
            # Vérifier si le fichier est présent dans la requête
            if 'image' not in request.files:
                return 'Pas de fichier téléchargé', 400
            file = request.files['image']
            if file.filename == '':
                return 'Aucun fichier sélectionné', 400
            if file:
                # Enregistrer le fichier de manière sécurisée
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Stocker le chemin de l'image dans la session
                session['image_path'] = 'uploads/' + filename

                # Rendre le template avec l'image affichée, sans prédiction
                return render_template('index.html', image_path=session.get('image_path'))
        elif action == 'Prédire':
            # Récupérer le chemin de l'image depuis la session
            image_path = session.get('image_path')
            if image_path:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(image_path))

                # Charger et prétraiter l'image
                img = load_img(filepath, target_size=(224, 224))
                img_array = img_to_array(img)
                img_array = preprocess_input(img_array)
                img_array = np.expand_dims(img_array, axis=0)

                # Effectuer la prédiction
                prediction = model.predict(img_array)
                probability = prediction[0][0]

                # Interpréter la prédiction
                threshold = 0.5
                if probability > threshold:
                    predicted_class = 'Zèbre'
                    confidence = probability * 100
                else:
                    predicted_class = 'Éléphant'
                    confidence = (1 - probability) * 100

                # Renvoyer le résultat à la page HTML
                return render_template('index.html', prediction=predicted_class, confidence=confidence, image_path=image_path)
            else:
                return 'Aucune image téléchargée pour la prédiction.', 400
        elif action == 'Réinitialiser':
            # Effacer la session et rediriger vers la page d'accueil
            session.clear()
            return redirect(url_for('index'))
    else:
        # Requête GET
        session.clear()
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
