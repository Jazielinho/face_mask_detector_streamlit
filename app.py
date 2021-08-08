

'''
Autor: Jazielinho
'''


import numpy as np
import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import tensorflow as tf

from typing import List, Tuple


model = None

SHAPE = (224, 224)

color_dict = {1: (0, 255, 0), 0: (0, 0, 255)}
labels_dict = {0: 'no mascara', 1: 'mascara'}


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detecta_rostros(frame: np.array) -> Tuple[List, List]:
    ''' Retorna lista con imágenes de rostros y su ubicación en la imagen original '''
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        frame_gray,
        scaleFactor=1.1,
        minNeighbors=5,
    )

    if len(faces) > 0:
        face_crop = []
        list_ubications = []
        for x, y, w, h in faces:
            face_crop.append(frame[y: y + h, x: x + w])
            list_ubications.append([x, y, w, h])
        return face_crop, list_ubications
    return [], []


def get_model() -> tf.keras.Model:
    base_model = tf.keras.applications.mobilenet.MobileNet(
        include_top=False, 
        weights='imagenet', 
        pooling='max', 
        input_shape=(*SHAPE, 3)
    )
    for layer in base_model.layers:
        layer.W_regularizer = tf.keras.regularizers.l2(1e-3)
        layer.trainable = True
    
    output = base_model.output
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Dropout(0.5)(output)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(output)
    model = tf.keras.models.Model(base_model.input, output)
    return model


def load_model() -> tf.keras.Model:
    ''' Cargando el modelo '''
    global model
    if model is None:
        model = get_model()
        weights_path = tf.keras.utils.get_file(
            'model.h5',
            'https://github.com/Jazielinho/face_mask_detector/blob/master/models/model.h5?raw=true'
        )
        model.load_weights(weights_path)
    return model


model = load_model()


def prepara_imagen_array(img: np.ndarray) -> Tuple[List, List]:
    ''' De la imagen capturara, extrae los rostros y preprocesa '''
    face_crop, list_ubications = detecta_rostros(frame=img)
    face_to_predict = []

    if len(face_crop) > 0:
        for face_ in face_crop:
            img_ = cv2.resize(face_, SHAPE[:2])
            img_ = np.reshape(img_, (1, *SHAPE, 3))
            img_ = tf.keras.applications.mobilenet.preprocess_input(img_)
            face_to_predict.append(img_)
    return face_to_predict, list_ubications


def get_predictions(face_to_predict: List) -> List:
    # Calcula la probabilidad y clase (0 o 1) para una lista de rostros identificados
    global model
    model = load_model()

    list_clases = []
    for face_ in face_to_predict:
        prob = model.predict(face_).ravel()
        list_clases.append(int(prob < 0.5))
    return list_clases


# ==================================================================================================================

class VideoTransformer(VideoTransformerBase):
    # Clase para predecir máscaras de vídeos
    @staticmethod
    def transform_(img: np.array) -> np.array:
        face_to_predict, list_ubications = prepara_imagen_array(img=img)
        list_clases = get_predictions(face_to_predict=face_to_predict)

        if len(list_clases) > 0:

            for enum in range(len(list_clases)):
                x, y, w, h = list_ubications[enum]
                img = cv2.rectangle(img, (x, y), (x + w, y + h), color_dict[list_clases[enum]], 2)
                img = cv2.rectangle(img, (x, y - 40), (x+w, y), color_dict[list_clases[enum]], -2)
                img = cv2.putText(img, labels_dict[list_clases[enum]], (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.75,
                                  (255, 255, 255), 1, cv2.LINE_AA)
        return img, list_clases

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img, list_clases = VideoTransformer.transform_(img=img)
        return img


# ==================================================================================================================
st.title('Detección automática de máscaras')

st.write("Esta aplicación identifica en tiempo real si tiene o no máscara.")
st.write("Para más información puede ir al siguiente enlace: ")

st.write("Para más información: ")

status = st.sidebar.radio("Elija subir imagen o acceder a la camara web", ("Subir imagen", "Camara web"))

if status == "Camara web":
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
else:
    uploaded_file = st.file_uploader("Sube imagen", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        new_image, list_clases = VideoTransformer.transform_(img=image)
        text = f"Hay {len(list_clases)} encontradas, {len([x for x in list_clases if x > 0])} con máscara"
        st.image(new_image, caption=text, use_column_width=True, channels="BGR")