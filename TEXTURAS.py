import numpy as np
#import matplotlib.pyplot as plt
#from skimage import io, color

def Q1(f, pixel):

    y,x = pixel
    elementos_G = []
    
    # Hacia la derecha

    elemento_G_x = f[y][x]
    elemento_G_y = f[y][x + 1]
    elementos_G.append([elemento_G_y, elemento_G_x])

    # Hacia la izquierda

    elemento_G_x = f[y][x]
    elemento_G_y = f[y][x-1]
    elementos_G.append([elemento_G_y, elemento_G_x])

    elementos_G = np.array(elementos_G)
    
    if np.all(elementos_G == elementos_G[0]):
        return elementos_G[0]
    else:
        return [-1, -1]

def crear_G(f):
    elementos_maximo = np.max(f)
    return np.zeros((elementos_maximo + 1, elementos_maximo + 1))

def padding_negativo(f):
    filas, columnas = f.shape
    matriz_padded = np.full((filas + 2, columnas + 2), -1)
    matriz_padded[1:-1, 1:-1] = f
    return matriz_padded

def co_ocurrenciaG(f):
    f_padded = padding_negativo(f)
    filas, columnas = f.shape
    G = crear_G(f_padded)

    for i in range(1, filas + 1):
        for j in range(1, columnas + 1):
            pixel = [i, j]
            elemento_G = Q1(f_padded, pixel)
            if not np.array_equal(elemento_G, [-1, -1]):
                G[elemento_G[0]][elemento_G[1]] += 1

    return G

def uniformidad(G):
    return np.sum(G**2)

def contraste(G):
    filas, columnas = G.shape
    contraste_suma = 0
    for i in range(filas):
        for j in range(columnas):
            contraste_suma+= (i - j)**2 * G[i, j]
    return contraste_suma

def entropia(G):
    G = np.array(G)
    G = G[G > 0]
    return -np.sum(G * np.log2(G))

def normalizar_G(G):
    total = np.sum(G)
    if total != 0:
        return G / total
    else:
        return G

def homogeneidad(G):
    filas, columnas = G.shape
    homogeneidad_suma = 0
    for i in range(filas):
        for j in range(columnas):
            homogeneidad_suma += G[i, j] / (1 + abs(i - j))
    return homogeneidad_suma

def vector_descriptores_textura(array_3d):

    descriptores = []
    for  imagen_f in array_3d:
        G1 = co_ocurrenciaG(imagen_f*255)
        G1_normalizada = normalizar_G(G1)
        u = uniformidad(G1_normalizada)
        c= contraste(G1_normalizada)
        e = entropia(G1_normalizada)
        h = homogeneidad(G1_normalizada)

        descriptores.append([u,c,e,h])

    descriptores = np.array(descriptores)
    return descriptores


'''
array_3d = np.load('img/olivetti_faces.npy')
#Traer las etiquetas
etiquetas = np.load('img/olivetti_faces_target.npy')

print(len(array_3d))

descriptores = vector_descriptores(array_3d)

print(descriptores[0,:])


imagen = io.imread('img/conjuntoPequenoPruebas/img (1).jpg')
matriz = color.rgb2gray(imagen) * 255

G1 = co_ocurrenciaG(matriz)


G1_normalizada = normalizar_G(G1)


u = uniformidad(G1_normalizada)
c= contraste(G1_normalizada)
e = entropia(G1_normalizada)
h = homogeneidad(G1_normalizada)

print(f"Uniformidad: {u}")
print(f"Contraste: {c}")
print(f"Entropia: {e}")
print(f"Homogeneidad: {h}")

# Plotting the matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot original matrix
axes[0].imshow(matriz, cmap='gray', aspect='auto')
axes[0].set_title('Matriz Original')

# Plot G1 matrix
axes[1].imshow(G1, cmap='gray', aspect='auto')
axes[1].set_title('Matriz G1 Resultante')

plt.tight_layout()
plt.show()



'''