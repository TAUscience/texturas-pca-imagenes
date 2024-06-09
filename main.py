import numpy as np
import numpy as np
import pandas as pd
import csv
import vectorizacion
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Cargar el objetivo (target)
y = np.load('img/olivetti_faces_target.npy')

Xtextu=np.delete(vectorizacion.VECTOR_TEXTURA, 0, axis=1)
Xtextu=np.delete(Xtextu, 0, axis=1)
# Unir las dos matrices de características
X = np.hstack((vectorizacion.VECTOR_PCA, vectorizacion.VECTOR_TEXTURA))
np.savetxt('DATASET.csv', X, delimiter=',', fmt='%.5f')

# Configurar el modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)
print("INICIA")
# Realizar la validación cruzada con 5 particiones (5-fold cross-validation)
scores = cross_val_score(knn, X, y, cv=5)
print("FINALIZA")
# Imprimir los resultados de la validación cruzada
print(f'Scores de la validación cruzada: {scores}')
print(f'Accuracy promedio: {scores.mean()}') 

