"""Script run by an Azure endpoint to the make predictions on images using the model"""

import os
import tensorflow as tf
import numpy as np

from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse
from PIL import Image
import json

# Based on https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-advanced-entry-script#binary-data

def init():
    global model

    # AZUREML_MODEL_DIR points to the folder ./azureml-models/$MODEL_NAME/$VERSION
    model_folder = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "outputs")
    model = tf.keras.models.load_model(model_folder)
    print(model)

@rawhttp
def run(request):
    print(request)

    if request.method == 'POST':
        file_bytes = request.files["image"]
        image = Image.open(file_bytes).convert('RGB')

        new_image_size = (512, 512)
        image = image.resize(new_image_size)
        image_data = tf.keras.preprocessing.image.img_to_array(image)
        
        image_data = image_data.reshape(1, 512, 512, 3)

        predictions = model.predict(image_data).tolist()
        return AMLResponse(json.dumps(predictions), 200)
    else:
        return AMLResponse("Bad Request", 500)
    

    # need to work out the format of the data coming into the function (could be in PIL format or a
    # raw PNG maybe?)
    # ... if that decision is up to us - probably just accept a PNG image, then convert it
    # then transform the image into the format the model wants
    # Input is a 3D array with dimensions n,n,3 (1 channel each for RGB and image width n)
    # tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    # then I think we can just do model.predict(image) and return the results????
