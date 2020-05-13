# -*- coding: utf-8 -*-

import pandas
import numpy
base = pandas.read_csv('Pybrain-Feedforward-network/credit_data/credit_db.csv')

previsores = base.iloc[:,1:4].values
classe = base.iloc[:,4].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=numpy.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size = 0.25, random_state = 0)

import keras
from keras.models import Sequential
from keras.layers import Dense

classificador = Sequential()

classificador.add(Dense(units=2, activation='relu', input_dim=3))
classificador.add(Dense(units=2, activation='relu'))
classificador.add(Dense(units=1, activation='sigmoid'))

classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento, batch_size=1, nb_epoch=100)

previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.95)

from sklearn.metrics import confusion_matrix, accuracy_score

precisao = accuracy_score(classe_teste, previsoes)
print('precisao: ')
print(precisao)
matriz = confusion_matrix(classe_teste, previsoes)
print(matriz)
