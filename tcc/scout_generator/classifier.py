from keras.models import model_from_json
from keras.preprocessing.image import load_img, ImageDataGenerator, img_to_array
from PIL import Image
from skimage.metrics import structural_similarity
import json
import numpy as np
import os
import cv2

POSSIBLE_RESULTS = {
    0: 'pênalti',
    1: 'falta',
    2: 'bola rolando'
}

def check_threshold(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_lower = np.array([30, 40, 40])
    green_higher = np.array([70, 255, 255])
    mask = cv2.inRange(hsv, green_lower, green_higher)
    green_pixels_amount = image[mask > 0].shape[0]
    if green_pixels_amount/(image.shape[0] * image.shape[1]) < 0.6:
        return True
    return False

def check_similarity(f1, f2):
    first_frame = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    second_frame = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    (score, _) = structural_similarity(first_frame, second_frame, full=True)
    return score

def load_model(path):
    # load json and create model
    json_file = open(path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights('model/model.h5')

    return loaded_model

def get_input_shape(model_json_path):
    with open(model_json_path) as json_file:
        data = json.load(json_file)
        input_shape_array = data['config']['layers'][0]['config']['batch_input_shape']
        return (input_shape_array[1], input_shape_array[2])

def identify_event(model, image=None, path=None):
    input_shape = get_input_shape('model/model.json')
    if image is not None:
        image = cv2.resize(image, (input_shape[1], input_shape[0]))
    elif path is not None:
        image = load_img(path, target_size=(input_shape[0], input_shape[1]))
    else:
        return 'No image or path!'
    preprocessed_image = np.expand_dims(img_to_array(image), axis=0)
    prediction = model.predict(preprocessed_image)
    result = np.argmax(prediction, axis=-1)
    
    return POSSIBLE_RESULTS[result[0]]