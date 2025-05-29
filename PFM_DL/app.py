import numpy as np
import streamlit as st
import cv2
import json
from keras.models import load_model
import base64
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

image_path = "bk.jpg"  
image_base64 = image_to_base64(image_path)


def preprocess_model_input(image, model_name):
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)

    if model_name == "ResNet50":
        return preprocess_resnet(np.expand_dims(image, axis=0))
    elif model_name == "VGG16":
        return preprocess_vgg(np.expand_dims(image, axis=0))
    else:
        return np.expand_dims(image / 255.0, axis=0)


custom_style = f"""
<style>
/* Arrière-plan */
.stApp {{
    background-image: url("data:image/png;base64,{image_base64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100vh;
}}

/* Centrage vertical et layout global */
.block-container {{
    background-color: rgba(255, 255, 255, 0.10) !important;
    padding: 2rem;
    border-radius: 15px;
    width: 100%;
    max-width: 890px;
    margin: auto;
    # margin-top: 4.2rem;

}}

/* Inputs */
input[type="text"], input[type="number"], textarea {{
    background-color: rgba(255, 255, 255, 0.17) !important;
    color: black !important;
    border: 1px solid #ccc !important;
    border-radius: 8px;
    padding: 10px;
    font-size: 1rem;
}}

/* Selectbox */
.stSelectbox > div {{
    background-color: rgba(255, 255, 255, 0.10) !important;
    color: black !important;
    border-radius: 8px;
    padding: 5px;
}}

/* 🏷️ Labels */
label, .stSelectbox label {{
    color: white !important;
    font-weight: bold;
    margin-bottom: 6px;
    text-shadow: 1px 1px 3px #21381f;
    font-size: 1rem;
}}

/* Titre */
h1 {{
    color: white !important;
    font-weight: bold;
    text-align: center;
    text-shadow: 1px 1px 3px #21381f;
    margin-bottom: 15px;
}}

/*  Bouton prédire */
.stButton > button {{
    background-color: rgba(255, 255, 255, 0.10) !important;
    color: white !important;
    border: none;
    border-radius: 12px;
    font-weight: 900 px;
    padding: 0.8rem 2rem;
    font-size: 2.9rem;  
    letter-spacing: 1.2px;
    text-shadow: 2px 2px 4px #000000;

    box-shadow: 0px 0px 12px rgba(0,0,0,0.3);
    transition: all 0.3s ease;
}}


.stButton > button:hover {{
    background-color: rgba(255, 255, 255, 0.20) !important;
    transform: scale(1.03);
    box-shadow: 0px 0px 16px rgba(0,0,0,0.5);
}}

/* Résultat */
.stMarkdown p, .stMarkdown div, .stSuccess, .stAlert-success, .stText {{
    color: white !important;
    font-size: 1.3rem !important;
    text-shadow: 1px 1px 3px #000000;


}}

/* Nom du fichier uploadé */
span[data-testid="stFileUploaderFilename"] {{
    color: white !important;
    font-weight: bold;
    font-size: 1.2rem;
}}

/* Icône upload */
span[data-testid="stFileUploaderDropzoneIcon"] svg {{
    fill: white !important;
}}

/* Modèle utilisé */
.stMarkdown h3, h2 {{
    color: white !important;
    font-size: 1.5rem !important;
    text-shadow: 1px 1px 3px #000000;

    font-weight: bold;
    margin-bottom: 10px;
}}

/* Marqueur Markdown */
[data-testid="stMarkdownContainer"] p {{
    color: white!important;
    font-size: 1.3rem !important;
    text-shadow: 1px 1px 3px#000000;


}}
.st-emotion-cache-1uixxvy {{
    margin-right: 0.5rem;
    margin-bottom: 0.25rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: white;
}}
.st-emotion-cache-16tyu1 h3 {{
    font-size: 1.75rem;
    padding: 0.75rem 0px 1rem;
    color: white;
}}
.st-emotion-cache-1rpn56r {{
    font-size: 0.875rem;
    line-height: 1.25;
    color: white;
}}
.st-emotion-cache-1gulkj5 {{
    display: flex
;
    -webkit-box-align: center;
    align-items: center;
    padding: 1rem;
    background-color: rgb(255, 255, 255, 0.1) !important;
    border-radius: 0.5rem;
    color: rgb(49, 51, 63);
}}
.st-bc {{
    height: 2.5rem;
    height: 2.5rem;
    background-color: rgba(255, 255, 255, 0.10) !important;
    color: white;
    background-color: rgba(255, 255, 255, 0.10) !important;
    color: white !important;
    border: none;
    border-radius: 12px;
    font-weight: 900 px;
    /* padding: 0.8rem 2rem; */
    /* font-size: 2.9rem; */
}}
.st-cp {{
    width: 1.5rem;
    background-color: white;
}}
.st-emotion-cache-9ycgxx {{
    margin-bottom: 0.25rem;
    COLOR: WHITE;
}}
</style>
"""

     

st.markdown(custom_style, unsafe_allow_html=True)

MODELES = {
    "ResNet50": "model_resnet50.h5",     
    "CNN personnalisé": "cnn_model.h5",
    "VGG16": "vgg_model.h5"
}


try:
    with open("class_names.json", "r") as f:
        CLASS_NAMES = json.load(f)
except FileNotFoundError:
    st.error("Le fichier class_names.json est introuvable.")
    st.stop()

st.title("Détection de maladies des plantes 🌿")


st.markdown(
    "<p class='description'>Chargez l'image et choisissez le modèle !</p>",
    unsafe_allow_html=True
)

selected_model_name = st.selectbox(
    "Choisissez le modèle de prédiction :",
    list(MODELES.keys()),
    index=0  
)

plant_image = st.file_uploader("Choisissez une image au format JPG", type=["jpg", "jpeg"])

submit = st.button('Prédire la maladie')

if submit:
    if plant_image is not None:
        model_path = MODELES[selected_model_name]
        model = load_model(model_path)

        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        st.image(opencv_image, caption='Image chargée', channels="BGR")

        resized_image = cv2.resize(opencv_image, (224, 224))
        normalized_image = resized_image / 255.0
        input_image = preprocess_model_input(opencv_image, selected_model_name)


        prediction = model.predict(input_image)
        predicted_index = int(np.argmax(prediction))
        predicted_class = CLASS_NAMES[predicted_index]

        if '___' in predicted_class:
            plante, maladie = predicted_class.split('___', 1)
            if maladie.lower() == 'healthy':
                maladie = "Aucune (plante saine)"
            else:
                maladie = maladie.replace('_', ' ')
        else:
            plante = predicted_class
            maladie = "Non spécifiée"

        st.success(f"🌿 Modèle utilisé : {selected_model_name}")
        st.subheader("Résultat :")
        st.markdown(f"**Plante** : {plante}")
        st.markdown(f"**Maladie détectée** : {maladie}")
    else:
        st.warning("Veuillez charger une image avant de soumettre.")
