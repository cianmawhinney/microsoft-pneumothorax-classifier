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
    print('Model loaded')
    print(model)


@rawhttp
def run(request):
    print(request)

    if request.method == 'POST':

        results = []

        for filename in request.files:
            # grab the image from the request
            image = Image.open(request.files[filename]).convert('RGB')

            # convert the image into the right sized numpy array for the model to work with
            new_image_size = (512, 512)
            image = image.resize(new_image_size)
            image_data = tf.keras.preprocessing.image.img_to_array(image)

            image_data = image_data.reshape(1, 512, 512, 3)
            image_data /= 255

            # calculate predictions based on the model
            predictions = model.predict(image_data).tolist()

            # TODO: return an object in the format:
            # [
            #  {
            #     score: 0.71,
            #     confidence: 60.5,
            #     pneumothoraxDetected: true
            #   },
            #   ...
            # ]

            results.append(predictions)

        return AMLResponse(json.dumps(results), 200)
    else:
        return AMLResponse("Bad Request", 500)
