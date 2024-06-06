import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

def Q1(f, pixel):
    filas, columnas = f.shape
    x, y = pixel
    elementos_G = []
    
    # Hacia la derecha
    if x < filas and y < columnas - 1:
        elemento_G_x = f[x][y]
        elemento_G_y = f[x][y + 1]
        elementos_G.append([elemento_G_x, elemento_G_y])
    else:
        elementos_G.append([-1, -1])
    
    # Hacia abajo
    if x < filas - 1 and y < columnas:
        elemento_G_x = f[x][y]
        elemento_G_y = f[x + 1][y]
        elementos_G.append([elemento_G_x, elemento_G_y])
    else:
        elementos_G.append([-1, -1])
    
    # Hacia abajo a la izquierda
    if x < filas - 1 and y > 0:
        elemento_G_x = f[x][y]
        elemento_G_y = f[x + 1][y - 1]
        elementos_G.append([elemento_G_x, elemento_G_y])
    else:
        elementos_G.append([-1, -1])
    
    return elementos_G

def crear_G(f):
    elementos_maximo = np.max(f)
    print(elementos_maximo)
    return np.zeros((elementos_maximo + 1, elementos_maximo + 1))

def padding_negativo(f):
    filas, columnas = f.shape
    matriz_padded = np.full((filas + 2, columnas + 2), -1)
    matriz_padded[1:-1, 1:-1] = f
    return matriz_padded

def iterar_f(f):
    f_padded = padding_negativo(f)
    filas, columnas = f.shape
    G = crear_G(f_padded)

    for i in range(1, filas + 1):
        for j in range(1, columnas + 1):
            pixel = [i, j]
            elementos_G = Q1(f_padded, pixel)
            for elemento_G in elementos_G:
                if elemento_G != [-1, -1]:
                    G[elemento_G[0]][elemento_G[1]] += 1

    return G

imagen = io.imread('img/conjuntoPequenoPruebas/img (0).jpg')
matriz = color.rgb2gray(imagen) * 255

G1 = iterar_f(matriz)

print(G1.shape)

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
