# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 13:17:22 2018

@author: Jérôme
"""

"""-----------------------------------------------------------------------"""
"""

    Ce programme effectue du transfert learning en utilisant le modèle
    VGG16, avec keras. On supprime les dernières couches pour les 
    remplacer par des couches adaptées à notre travail, notamment la couche
    de prédiction sur 10 ou 15 races selon la base de données choisies.

"""
"""-----------------------------------------------------------------------"""




import tensorflow as tf
import pickle
import numpy as np

from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras


# création d'un encodeur pour les labels des races de chiens
encodeur=LabelBinarizer()


''' chargement des données entrainement '''
donnees = pickle.load(open('donnees_train.sav', 'rb'))
races = pickle.load(open('race_train.sav', 'rb'))

races_encode = encodeur.fit_transform(races)



''' chargement des données test '''
donnees_test = pickle.load(open('donnees_test.sav', 'rb'))
races_test = pickle.load(open('race_test.sav', 'rb'))

races_test_encode = encodeur.fit_transform(races_test)

x_train = donnees.astype('float32')
x_test = donnees_test.astype('float32')


# chargement du modèle vgg16 sans les couches fully-connected

model = tf.keras.applications.VGG16(weights="imagenet", include_top = False, input_shape=(224,224,3))


# hyperparamètres : variation de la taille du batch pour l'entrainement
taille_batch = 32 
# taille_batch = 16
# taille_batch = 64


# redéfinition des dernièrs couches fully-connected
add_model = tf.keras.models.Sequential()
add_model.add(tf.keras.layers.Flatten(input_shape = model.output_shape[1:]))
add_model.add(tf.keras.layers.Dense(256, activation='relu'))
add_model.add(tf.keras.layers.Dense(10, activation='softmax'))

model = tf.keras.models.Model(inputs=model.input, outputs=add_model(model.output))

for layer in model.layers[:5]:
   layer.trainable = False


# compilation du modèle
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9),metrics=['accuracy'])


model.fit(x_train, races_encode, epochs=5, batch_size=taille_batch)


# sauvegarde du modèle complet et des poids
model.save_weights('model_weights.h5')
model.save('model_keras.h5')
