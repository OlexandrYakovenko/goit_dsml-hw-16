# import os

# file_path = r'D:\python game\goit_dsml-hw-16\task1\fashion_mnist_model_con4.keras'
# if os.path.exists(file_path):
#     print("Файл знайдено!")
# else:
#     print("Файл не знайдено. Перевірте шлях та назву файлу.")

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,GlobalMaxPooling2D, Dropout
# from tensorflow.keras import models, layers
# from tensorflow.keras import saving
# from tensorflow.keras.models import load_model

# keras_model = load_model(r'D:\python game\goit_dsml-hw-16\task1\fashion_mnist_model_con5.h5')

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model, model_from_json, Sequential
from tensorflow.keras import optimizers

# Виведення версій TensorFlow та Keras
# st.write(f"TensorFlow version: {tf.__version__}")
# st.write(f"Keras version: {tf.keras.__version__}")
print(f"TensorFlow version: {tf.__version__}")
#print(f"Keras version: {tf.keras.__version__}")

# Завантаження моделей і історій тренувань
@st.cache_resource
def load_data():
    
    string_path_model_conv=r"D:\python game\goit_dsml-hw-16\3\fashion_mnist_model_conv.h5"
    string_path_model_conv_history=r"D:\python game\goit_dsml-hw-16\3\fashion_mnist_model_conv_history.pkl"
    string_path_model_conv_optimizer_state=r"D:\python game\goit_dsml-hw-16\3\optimizer_state_fashion_mnist_model_conv.pkl"
    
    model_conv = load_model(string_path_model_conv)
    with open(string_path_model_conv_history, 'rb') as f:
        history_conv = pickle.load(f)
    with open(string_path_model_conv_optimizer_state, 'rb') as f:
        optimizer_state_conv = pickle.load(f)


    string_path_modified_model_VGG16=r"D:\python game\goit_dsml-hw-16\6\fashion_mnist_model_VGG16.h5"
    # string_path_modified_model_VGG16=r"D:\python game\goit_dsml-hw-16\3\modified_model_VGG16.h5"
    # string_path_modified_model_VGG16=r"D:\python game\goit_dsml-hw-16\3\modified_model_VGG16.pkl"
    #string_path_modified_model_VGG16=r"D:\python game\goit_dsml-hw-16\4\modified_model_VGG16.keras"
    
    # string_path_modified_model_VGG16=r"D:\python game\goit_dsml-hw-16\4\modified_model_VGG16\modified_model_VGG16.json"
    # string_path_modified_model_VGG16_weights=r"D:\python game\goit_dsml-hw-16\4\modified_model_VGG16\modified_model_VGG16_weights1.weights.h5"
    
    string_path_modified_model_VGG16_history=r"D:\python game\goit_dsml-hw-16\6\fashion_mnist_model_VGG16_history.pkl"
    # string_path_modified_model_VGG16=r"D:\python game\goit_dsml-hw-16\4\modified_model_VGG16.pkl"
    # string_path_modified_model_VGG16_history=r"D:\python game\goit_dsml-hw-16\3\modified_model_VGG16_history.pkl"
    # string_path_modified_model_VGG16_history=r"D:\python game\goit_dsml-hw-16\4\modified_model_VGG16_history.pkl"
    
    string_path_modified_model_VGG16_optimizer_state=r"D:\python game\goit_dsml-hw-16\6\optimizer_state_model_vgg_VGG16.pkl"
    # string_path_modified_model_VGG16_optimizer_state=r"D:\python game\goit_dsml-hw-16\3\optimizer_state_modified_model_VGG16.pkl"
    # string_path_modified_model_VGG16_optimizer_state=r"D:\python game\goit_dsml-hw-16\4\optimizer_state_modified_model_VGG16.pkl"
    # model_vgg16 = load_model(string_path_modified_model_VGG16)
    # model_vgg16 = tf.keras.models.load_model(string_path_modified_model_VGG16)
    
    # # load json and create model
    # json_file = open(string_path_modified_model_VGG16, 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # model_vgg16 = model_from_json(loaded_model_json)
    # # load weights into new model
    # model_vgg16.load_weights(string_path_modified_model_VGG16_weights)
    # print("Loaded model from disk")

    # evaluate loaded model on test data

    # model_vgg16 = tf.keras.saving.load_model(string_path_modified_model_VGG16)
    model_vgg16 = load_model(string_path_modified_model_VGG16)
    # model_vgg16 = load_model(string_path_modified_model_VGG16, compile=False)
    # model_vgg16.compile(optimizer=optimizers.RMSprop(learning_rate=2e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # model_vgg16.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # with open(string_path_modified_model_VGG16, 'rb') as file:
    #     model_vgg16 = pickle.load(file)
    
    with open(string_path_modified_model_VGG16_history, 'rb') as f:
        history_vgg16 = pickle.load(f)
    with open(string_path_modified_model_VGG16_optimizer_state, 'rb') as f:
        optimizer_state_vgg16 = pickle.load(f)
    
    # # Діагностичні повідомлення для перевірки форми даних
    # for layer in model_vgg16.layers:
    #     if 'flatten' in layer.name.lower():
    #         print(f"Layer {layer.name} input shape: {layer.input.shape}")
            
    return (model_conv, history_conv, optimizer_state_conv), (model_vgg16, history_vgg16, optimizer_state_vgg16)
    # return (model_conv, history_conv, optimizer_state_conv)
    # return (model_vgg16, history_vgg16, optimizer_state_vgg16)

# conv_data = load_data()
# vgg16_data = load_data()



# Функція для класифікації зображення з моделі Convolutional Neural Network
def classify_image_conv(image, model):
    img = image.resize((28, 28))
    img = img.convert('L')
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32') / 255.0

    prediction = model.predict(img)

    return prediction, np.argmax(prediction)

# Функція для класифікації зображення з моделі VGG16
def classify_image_vgg16(image, model):
    img = image.resize((32, 32))
    img = np.array(img)
    img = img.reshape(1, 32, 32, 3)
    img = img.astype('float32') / 255.0

    prediction = model.predict(img)

    return prediction, np.argmax(prediction)

def main():
    
    (conv_data, vgg16_data) = load_data()
    (model_conv, history_conv, optimizer_state_conv) = conv_data
    (model_vgg16, history_vgg16, optimizer_state_vgg16) = vgg16_data

     # Вибір моделі за допомогою радіо кнопок
    model_choice = st.radio("Виберіть модель для класифікації зображення:",
                            ('Convolutional Neural Network', 'VGG16'))

    st.write('Завантажте зображення для класифікації')

    uploaded_file = st.file_uploader("Виберіть зображення...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        if st.button('Класифікувати'):
            if model_choice == 'Convolutional Neural Network':
                probabilities, predicted_class = classify_image_conv(image, model_conv)
                history = history_conv
            elif model_choice == 'VGG16':
                probabilities, predicted_class = classify_image_vgg16(image, model_vgg16)
                history = history_vgg16

            st.write(f'Ймовірності: {probabilities.flatten()}')
            st.write(f'Передбачений клас: {predicted_class}')

            fig, ax = plt.subplots()
            ax.bar(np.arange(10), probabilities.flatten(), align='center', alpha=0.5)
            ax.set_xticks(np.arange(10))
            ax.set_xticklabels(['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'], rotation=45)
            ax.set_ylabel('Probability')
            ax.set_title('Probabilities for each class')
            st.pyplot(fig)

            # Виведення графіків втрат і точності
            st.subheader(f'Графіки втрат і точності моделі {model_choice}')
            st.line_chart(history['loss'], use_container_width=True)
            st.line_chart(history['accuracy'], use_container_width=True)

# Виклик головної функції
if __name__ == '__main__':
    main()