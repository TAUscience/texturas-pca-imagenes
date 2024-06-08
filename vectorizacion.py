import PCAs

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




"""#################### PROCEDIMIENTO PARA PCA ####################"""
# Traer el conjunto de datos
array_3d = np.load('img/olivetti_faces.npy')
# Visualizar algunas de las imágenes
num_images_to_display = 3
for i in range(num_images_to_display):
    plt.imshow(array_3d[i], cmap='gray')
    plt.title(f'Imagen {i+1}')
    plt.show()

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







