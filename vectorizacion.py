import PCAs
import TEXTURAS

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Traer el conjunto de datos
array_3d = np.load('img/olivetti_faces.npy')
"""#################### PROCEDIMIENTO PARA TEXTURAS ####################"""

VECTOR_TEXTURA = TEXTURAS.vector_descriptores_textura(array_3d)
# Guarda el arreglo en un archivo CSV
np.savetxt('TEXTURAs.csv', VECTOR_TEXTURA, delimiter=',', fmt='%.5f')

#################### PROCEDIMIENTO PARA PCA ####################

# Representación del conjunto de datos en la forma de interés (EQ.1.2.1)
array_2d = array_3d.reshape(array_3d.shape[0], -1).T
# Calculo de la media (EQ.1.2.2)
media=PCAs.media(array_2d)
# Desviación respecto a la media (EQ.1.2.3)
deltas_instancias=PCAs.delta_instancias(array_2d,media)
datos_centrados=PCAs.data_centrado_filas_instancias(array_2d,media)
# Matriz de covarianza (EQ.1.2.4)
S_covarianza=PCAs.matriz_covarianza(deltas_instancias)
# Calcular los valores y vectores propios con numPy
print("Calcilando vectores propios")
valores_propios, vectores_propios = np.linalg.eig(S_covarianza)
# Ordenar los vectores propios en función de los valores propios
indices_ordenados = np.argsort(valores_propios)[::-1]
valores_propios_ordenados = valores_propios[indices_ordenados]
vectores_propios_ordenados = vectores_propios[:, indices_ordenados]
# Graficar y obtener n componentes para representar el 85% o más de varianza
k_componentes=PCAs.grafica_scree(valores_propios_ordenados,0.9)
#Obtener los componentes principales de acuerdo al k obtenido
componentes_principales = vectores_propios_ordenados[:, :k_componentes]
# Proyectar los datos
datos_proyectados = np.dot(datos_centrados, componentes_principales)
# Eliminar la parte imaginaria
VECTOR_PCA=np.real(datos_proyectados)

print("Dimensiones de nuevos datos")
print(VECTOR_PCA.shape)

# Guarda el arreglo en un archivo CSV
np.savetxt('PCAs.csv', VECTOR_PCA, delimiter=',', fmt='%.5f')




