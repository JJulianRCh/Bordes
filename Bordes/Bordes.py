import numpy as np
import matplotlib.pyplot as plt
from skimage import color

# Cargamos la imagen al codigo
imagen_original = plt.imread("Imagen.jpg")
plt.imshow(imagen_original)
plt.show()

# Se coloca un filtro sobre la imagen para que tenga una escala de grises
imagen = color.rgb2gray(imagen_original)
plt.imshow(imagen, cmap='gray')
plt.show()

# Se muestra en pantalla el tamaño de la imagen
print("Tamaño de la imagen: " + str(imagen.shape))

ancho = imagen.shape[0]
largo = imagen.shape[1]

print("Ancho: " + str(ancho))
print("Largo: " + str(largo))

# Imprimir el valor de algunos pixeles
print(imagen[100][200])
print(imagen[300][500])
print(imagen[400][700])

# Iniciamos el tamaño de valores
# que puede almacenar dependiendo de la imagen
dx = np.zeros(shape=(ancho, largo))

# Este ciclo obtendra la derivada punto por punto
for i in range(ancho - 1):
    for j in range(largo - 1):
        # Calcular la derivada parcial con respecto X
        dx[i][j] = abs(imagen[i + 1][j] - imagen[i][j])

plt.imshow(dx, cmap='gray')
plt.show()

# Repetimos procedimiento
dy = np.zeros(shape=(ancho, largo))

for i in range(ancho - 1):
    for j in range(largo - 1):
        # Obtener la derivada parcial con respecto Y
        dy[i][j] = abs(imagen[i][j + 1] - imagen[i][j])

plt.imshow(dy, cmap='gray')
plt.show()

# Se calcula la gradiente
gradiente = np.zeros(shape=(ancho, largo))

for i in range(ancho):
    for j in range(largo):
        gradiente[i][j] = dx[i][j] + dy[i][j]

plt.imshow(gradiente, cmap='gray')
plt.show()

# Se invierte los colores
for i in range(ancho):
    for j in range(largo):
        gradiente[i][j] = 1 - gradiente[i][j]

plt.imshow(gradiente, cmap='gray')
plt.show()

# Dirección inversa
dxi = np.zeros(shape=(ancho, largo))
for i in range(ancho - 1, -1, -1):
    for j in range(largo - 1, -1, -1):
        dxi[i][j] = abs(imagen[i - 1][j] - imagen[i][j])

plt.imshow(dxi, cmap='gray')
plt.show()

dyi = np.zeros(shape=(ancho, largo))

for i in range(ancho - 1, -1, -1):
    for j in range(largo - 1, -1, -1):
        dyi[i][j] = abs(imagen[i][j - 1] - imagen[i][j])

plt.imshow(dyi, cmap='gray')
plt.show()

gradiente2 = np.zeros(shape=(ancho, largo))

for i in range(ancho - 1, -1, -1):
    for j in range(largo - 1, -1, -1):
        gradiente2[i][j] = dxi[i][j] + dyi[i][j]

plt.imshow(gradiente2, cmap='gray')
plt.show()

for i in range(ancho):
    for j in range(largo):
        gradiente2[i][j] = 1 - gradiente2[i][j]

plt.imshow(gradiente2, cmap='gray')
plt.show()

# Direccion diagonal
diagonal = np.zeros(shape=(ancho, largo))

for i in range(ancho - 1):
    for j in range(largo - 1):
        diagonal[i][j] = abs(imagen[i + 1][j + 1] - imagen[i][j])
plt.imshow(diagonal, cmap='gray')
plt.show()

# Obtener la derivada parcial de manera diagonal de derecha a izquierda
diagonal2 = np.zeros(shape=(ancho, largo))

for i in range(ancho - 1):
    for j in range(-1, largo):
        diagonal2[i][j] = abs(imagen[i - 1][j - 1] - imagen[i][j])
plt.imshow(diagonal2, cmap='gray')
plt.show()

gradiente3 = np.zeros(shape=(ancho, largo))

for i in range(ancho):
    for j in range(largo):
        gradiente3[i][j] = diagonal[i][j] + diagonal2[i][j]

plt.imshow(gradiente3, cmap='gray')
plt.show()

for i in range(ancho):
    for j in range(largo):
        gradiente3[i][j] = 1 - gradiente3[i][j]

plt.imshow(gradiente3, cmap='gray')
plt.show()

# Conservar el signo de la derivada parcial
dx = np.zeros(shape=(ancho, largo))

for i in range(ancho - 1):
    for j in range(largo - 1):
        dx[i][j] = imagen[i + 1][j] - imagen[i][j]

plt.imshow(dx, cmap='gray')
plt.show()
