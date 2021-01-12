from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, img_to_array
import numpy as np
import os

def preprocess(image_array):
    image_array = (image_array[:, :, :-1])/255
    return image_array.reshape(-1, 200, 200, 3)

def identify_event(event, name):
    # load json and create model
    json_file = open('model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights('model/model.h5')

    # Resize image
    event = event.resize((200, 200))
    event_array = preprocess(img_to_array(event))
    result = loaded_model.predict_classes(event_array)
    
    if result == 0:
        return 'Esse evento é um pênalti'
    elif result == 1:
        return 'Esse evento é uma falta'
    else:
        return 'Esse evento é um escanteio'