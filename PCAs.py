import numpy as np
import matplotlib.pyplot as plt

def media(array2d):
    variables,N=array2d.shape
    media=np.zeros(variables)
    for i in range(N):
        media+=array2d[:,i]
    print("Media obtenida")
    return (media/N)

def data_centrado_filas_instancias(array2d,media):
    variables=len(media)
    print(variables)
    datos_centrados=np.copy(array2d.T)
    instancias=len(datos_centrados)
    for j in range(instancias):
        for i in range(variables):
            datos_centrados[j][i]-=media[i]
    return datos_centrados

def delta_instancias(array2d,media): #EQ 1.2.3
    variables,N=array2d.shape
    desviacion=np.zeros((variables,N))
    for i in range(N):
        desviacion[:,i]=array2d[:,i]-media
    print("Desviación respecto a la media obtenida")
    return desviacion
        
def matriz_covarianza(deltas): #EQ 1.2.4
    variables,N=deltas.shape
    matriz_covarianza=np.zeros((variables,variables))
    for i in range(N):
        print(f"Calculando matriz covarianza {i}/{N}")
        delta_interes=deltas[:,i].reshape(-1,1)
        matriz_covarianza+=np.dot(delta_interes,(delta_interes.T))
    print("Matriz covarianza calculada")
    return (matriz_covarianza/N)
    
def grafica_scree(valores_propios_ordenados,varianza_objetivo):
    # Calcular la varianza total
    varianza_total = np.sum(valores_propios_ordenados)

    # Calcular la fracción de varianza explicada por cada componente principal
    fraccion_varianza_explicada = valores_propios_ordenados / varianza_total

    # Calcular la varianza acumulada
    varianza_acumulada = np.cumsum(fraccion_varianza_explicada)

    # Crear un gráfico de barras
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, len(valores_propios_ordenados) + 1), fraccion_varianza_explicada, alpha=0.7, align='center', label='Fracción de Varianza Explicada')
    plt.step(range(1, len(valores_propios_ordenados) + 1), varianza_acumulada, where='mid', label='Varianza Acumulada')
    plt.ylabel('Fracción de Varianza / Varianza Acumulada')
    plt.xlabel('Componentes Principales')
    plt.title('Variabilidad de los Componentes Principales')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Encontrar el número de componentes necesarios para explicar el objetivo% de la varianza
    num_componentes_objetivo = np.argmax(varianza_acumulada >= varianza_objetivo) + 1

    return num_componentes_objetivo


"""a=[[1,10,30],[3,14,36],[0,9,29],[2,10,31]]
a=np.array(a)
print(a)
mediaa=media(a)
print(mediaa)
delta_instanciaas=delta_instancias(a,mediaa)
print(delta_instanciaas)
matriz_covarianzaa=matriz_covarianza(delta_instanciaas)
print(matriz_covarianzaa)
# Calcular los valores y vectores propios con numPy
valores_propios, vectores_propios = np.linalg.eig(matriz_covarianzaa)
# Ordenar los vectores propios en función de los valores propios
indices_ordenados = np.argsort(valores_propios)[::-1]
valores_propios_ordenados = valores_propios[indices_ordenados]
vectores_propios_ordenados = vectores_propios[:, indices_ordenados]
# Graficar y obtener k componentes para representar el 85% o más de varianza
k_componentes=grafica_scree(valores_propios_ordenados,0.85)
#Obtener los componentes principales de acuerdo al k obtenido
componentes_principales = vectores_propios_ordenados[:, :k_componentes]
print(componentes_principales)
print("Dimensiones de componentes P")
print(componentes_principales.shape)

# Proyectar los datos originales en el espacio de los componentes principales
datos_proyectados = np.dot(delta_instanciaas.T, componentes_principales)

# Seleccionar una sola variable para cada instancia (puedes elegir la primera variable en este caso)
vector_caracteristicas = datos_proyectados[:, 0]

# Imprimir el vector de características
print("Vector de características:")
print(vector_caracteristicas)

# Dimensiones del vector de características
print("Dimensiones del vector de características:")
print(vector_caracteristicas.shape)"""

"""# Suponiendo que tienes tu conjunto de datos en una matriz llamada data_array
data_array = np.array([[1.0, 2.0, 3.0,], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

# Paso 1: Calcular la media de cada característica
media_datos = media(data_array)

# Paso 2: Calcular la matriz de desviaciones respecto a la media
deltas = delta_instancias(data_array, media_datos)
datos_centrados=data_centrado_filas_instancias(data_array,media_datos)

# Paso 3: Calcular la matriz de covarianza
matriz_cov = matriz_covarianza(deltas)

# Paso 4: Calcular los vectores y valores propios
valores_propios, vectores_propios = np.linalg.eig(matriz_cov)

# Paso 5: Ordenar los valores propios de mayor a menor
valores_propios_ordenados = np.sort(valores_propios)[::-1]

# Paso 6: Graficar el Scree plot para ayudar a determinar el número de componentes principales a conservar
varianza_objetivo = 0.85  # Por ejemplo, conservar el 85% de la varianza
num_componentes_objetivo = grafica_scree(valores_propios_ordenados, varianza_objetivo)

# Proyectar los datos originales en el subespacio de las primeras k componentes principales
k = num_componentes_objetivo
componentes_principales = vectores_propios[:, :k]
datos_proyectados = np.dot(datos_centrados, componentes_principales)

# Los datos proyectados son ahora tus datos reducidos en dimensionalidad.
print(datos_proyectados)
"""