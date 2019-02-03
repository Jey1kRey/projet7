# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 13:13:47 2018

@author: Jérôme
"""


"""-----------------------------------------------------------------------"""
"""

    Ce programme effectue la création de l'API en utilisant le module Flask
    de Python. Il charge le modèle Tf-Idf exécuté dans un autre programme
    ainsi que le modèle de Forêt Aléatoire préalablement entraîné. 
    Ensuite, il demande à l'utilisateur une question, effectue un nettoyage
    de cette question, normalisation du texte et recherche des mots
    pertinents, enfin, retourne les tags possibles.

"""
"""-----------------------------------------------------------------------"""

import tensorflow as tf
import numpy as np
import pickle
import os

from tensorflow import keras
from flask import Flask, request, render_template, url_for, redirect
from PIL import Image
from PIL import ImageFilter
from werkzeug import secure_filename


label_dict = {0:'Lhasa', 1:'Basset', 2:'Doberman', 3:'Pomeranian', 4:'Japanese_spaniel', 5:'German_shepherd', 6:'Labrador_retriever', 7:'Chihuhua', 8:'Saint_Bernard', 9:'Komondor'}

# espace de sauvegarde pour les images chargées par l'utilisateur
sauv_nom = 'static/images/'



# fonction de transformation de l'image avec filtre et redimensionnement pour être conforme au modèle du CNN
def transfo_image(img): 

    img_ouv = Image.open(img)
    img_modif=img_ouv.filter(ImageFilter.SHARPEN)
    img_taille = img_modif.resize((224, 224))
    img = np.array(img_taille)
    return img



# prédiction effectuée avec le CNN issu de VGG16 de keras
def pred_image(img):

    model = tf.keras.models.load_model('model_keras.h5') 
    img_valide = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    
    prediction = model.predict(img_valide)
    
    label_indice = np.argmax(prediction)
    race_predite = label_dict.get(label_indice)

    return race_predite





app = Flask(__name__)


''' options '''

#app.config.from_object('config')

''' page d'accueil '''

@app.route('/')
def home():
    return render_template("image.html")


@app.route('/', methods=['POST'])
def formulaire():
    
    f=request.files.get('file')
    
    sauv_fichier = sauv_nom+ str(secure_filename(f.filename))
        
    f.save(sauv_fichier)


    image_test = transfo_image(sauv_fichier)

    race_predite = pred_image(image_test)
    
    return render_template('image.html',  race=race_predite, imgpath = sauv_fichier)




if __name__ == '__main__':
    app.run()
    
  