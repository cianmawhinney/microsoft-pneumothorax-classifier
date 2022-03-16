"""Script run by an Azure endpoint to the make predictions on images using the model"""

import os
import tensorflow as tf


def init():
    global model

    # AZUREML_MODEL_DIR points to the folder ./azureml-models/$MODEL_NAME/$VERSION
    model_folder = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "outputs")
    model = tf.keras.models.load_model(model_folder)
    print(model)


def run(raw_data):
    print(raw_data)

    # need to work out the format of the data coming into the function (could be in PIL format or a
    # raw PNG maybe?)
    # ... if that decision is up to us - probably just accept a PNG image, then convert it
    # then transform the image into the format the model wants
    # Input is a 3D array with dimensions n,n,3 (1 channel each for RGB and image width n)
    # tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    # then I think we can just do model.predict(image) and return the results????
