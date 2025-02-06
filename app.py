import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Cargar el modelo
model = load_model('fashion_mnist.keras')

# Crear la interfaz de usuario
st.title('Clasificador Fashion MNIST')
st.write('Sube una imagen para clasificarla como una categoría de ropa')

# Subir la imagen
uploaded_file = st.file_uploader('Selecciona una imagen en escala de grises y de 28x28 píxeles', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Abrir la imagen y mostrarla en la interfaz de usuario
    image = Image.open(uploaded_file).convert('L') # convertir rgb a blanco y negro
    image.resize((28, 28))
    image_array = np.array(image) / 255.0 # Normalizar
    # El primer 1 indica que solo hay una imagen, luego las dimensiones y el último 1 indica que solo hay un canal del color
    image_array = image_array.reshape(1, 28, 28, 1)
    st.image(image, caption='Imagen subida', use_column_width=True)

    # Hacer la predicción
    prediction = model.predict(image_array)
    class_names = ['Camiseta', 'Pantalón', 'Jersey', 'Vestido', 'Abrigo', 'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Bota']
    st.write('Predicción:', class_names[np.argmax(prediction)])
