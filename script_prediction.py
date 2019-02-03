# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 13:17:22 2018

@author: Jérôme
"""
"""-----------------------------------------------------------------------"""
"""

    Ce programme effectue le test du modèle CNN créé précédemment
    par transfert learning. 
    Il peut effectuer une prédiction sur une image ou sur plusieurs
    avec l'affichage d'un graphe où pour chaque image, la race 
    prédite est indiquée.

"""
"""-----------------------------------------------------------------------"""



import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

from tensorflow import keras
from sklearn.preprocessing import LabelBinarizer
from PIL import Image
from PIL import ImageFilter



img_dir='chiens/test/'
#image = 'chiens/test/Saint_Bernard-n02109525_12937.jpg'
#image = 'chiens/test/Basset-n02088238_3579.jpg'
#image = 'chiens/test/Chihuhua-n02085620_3681.jpg'
#image = 'chiens/test/German_shepherd-n02106662_16418.jpg'
#image = 'chiens/test/komondor-n02105505_1077.jpg'
#image = 'chiens/test/Pomeranian-n02112018_514.jpg'
#image = 'chiens/test/Lhasa-n02098413_932.jpg'
#image = 'chiens/test/Japanese_spaniel-n02085782_1724.jpg'


#images = ['chiens/test/Saint_Bernard-n02109525_12937.jpg','chiens/test/Basset-n02088238_3579.jpg','chiens/test/Chihuhua-n02085620_3681.jpg','chiens/test/German_shepherd-n02106662_16418.jpg','chiens/test/komondor-n02105505_1077.jpg']



label_dict = {0:'Lhasa', 1:'Basset', 2:'Doberman', 3:'Labrador_retriever', 4:'Japanese_spaniel', 5:'German_shepherd', 6:'Pomeranian', 7:'Chihuhua', 8:'Saint_Bernard', 9:'Komondor'}


images = [ img_dir+ x for x in os.listdir(img_dir)]


# définition de la fonction qui transforme l'image pour s'adapter au modèle CNN
def transfo_image(img): 

    img_ouv = Image.open(img)
    img_modif=img_ouv.filter(ImageFilter.SHARPEN)
    img_taille = img_modif.resize((224, 224))
    img = np.array(img_taille)
    return img


# défintition de la fonction qui effectue une transformation de l'image pour l'affichage sur la console
def voir_image(img):

    img_ouv = Image.open(img)
    img_modif=img_ouv.filter(ImageFilter.SHARPEN)
    img_taille = img_modif.resize((224,224))
    return img_taille




# chargement du modèle keras
model = tf.keras.models.load_model('model_keras.h5')  

''' Test de prédiction avec simple affichage textuel dans la console 

img_a_tester = transfo_image(image)
img_a_tester = img_a_tester.reshape(1, img_a_tester.shape[0], img_a_tester.shape[1], img_a_tester.shape[2])

prediction = model.predict(img_a_tester)

label_indice = np.argmax(prediction)

race_predite = label_dict.get(label_indice)
print(race_predite)
'''



# graphe des prédictions sur toutes les images contenus dans le dossier passé en paramètre en début de programme
plt.figure()
i=1
for image in images : 

    img_a_tester = transfo_image(image)
    img_a_tester = img_a_tester.reshape(1, img_a_tester.shape[0], img_a_tester.shape[1], img_a_tester.shape[2])

    prediction = model.predict(img_a_tester)
    label_indice = np.argmax(prediction)
    race_predite = label_dict.get(label_indice)
    image_test = voir_image(image)

    plt.subplot(8,5,i)
    plt.axis('off')
    plt.title(race_predite)
    i+=1
    plt.imshow(image_test)
plt.show()
