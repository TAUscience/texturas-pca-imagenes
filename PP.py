from PIL import Image, ImageOps

def redim_y_pad(image, size, color=(0, 0, 0)):
    """
    Redimensiona la imagen manteniendo la relación de aspecto y la rellena con un color hasta alcanzar el tamaño deseado.
    
    :param image: Imagen PIL a redimensionar.
    :param size: Tupla (ancho, alto) que representa el tamaño deseado.
    :param color: Color de fondo para el relleno.
    :return: Imagen PIL redimensionada y rellenada.
    """
    # Redimensiona la imagen manteniendo la relación de aspecto
    image.thumbnail(size, Image.Resampling.LANCZOS)
    
    # Crea una nueva imagen con el tamaño deseado y el color de fondo
    new_image = Image.new("RGB", size, color)
    
    # Calcula las coordenadas para centrar la imagen redimensionada
    left = (size[0] - image.size[0]) // 2
    top = (size[1] - image.size[1]) // 2
    new_image.paste(image, (left, top))
    
    return new_image

