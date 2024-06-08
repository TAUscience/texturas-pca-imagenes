import numpy as np
import numpy as np
import pandas as pd
import csv
import vectorizacion
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Cargar el objetivo (target)
y = np.load('img/olivetti_faces_target.npy')

# Acomodar los datos
X = vectorizacion.VECTOR_PCA

# Configurar el modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)
print("INICIA")
# Realizar la validación cruzada con 5 particiones (5-fold cross-validation)
scores = cross_val_score(knn, X, y, cv=5)
print("FINALIZA")
# Imprimir los resultados de la validación cruzada
print(f'Scores de la validación cruzada: {scores}')
print(f'Accuracy promedio: {scores.mean()}')

