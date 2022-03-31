"""Script run by an Azure endpoint to the make predictions on images using the model"""

import os
from tkinter import image_names
import tensorflow as tf
import numpy as np
from numpy import load
from sklearn.metrics import confusion_matrix
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.calibration import calibration_curve

from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse
from PIL import Image
import json

# Based on https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-advanced-entry-script#binary-data

class SigmoidCalibrator:
    def __init__(self, prob_pred, prob_true):
        prob_pred, prob_true = self._filter_out_of_domain(prob_pred, prob_true)
        prob_true = np.log(prob_true / (1 - prob_true))
        self.regressor = LinearRegression().fit(
            prob_pred.reshape(-1, 1), prob_true.reshape(-1, 1)
        )

    def calibrate(self, probabilities):
        return 1 / (1 + np.exp(-self.regressor.predict(probabilities.reshape(-1, 1)).flatten()))

    def _filter_out_of_domain(self, prob_pred, prob_true):
        filtered = list(zip(*[p for p in zip(prob_pred, prob_true) if 0 < p[1] < 1]))
        return np.array(filtered)


class IsotonicCalibrator:
    def __init__(self, prob_pred, prob_true):
        self.regressor = IsotonicRegression(out_of_bounds="clip")
        self.regressor.fit(prob_pred, prob_true)

    def calibrate(self, probabilities):
        return self.regressor.predict(probabilities)

def init():
    global model, sigmoid_calibrator

    # AZUREML_MODEL_DIR points to the folder ./azureml-models/$MODEL_NAME/$VERSION
    model_folder = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "outputs")
    model = tf.keras.models.load_model(model_folder)
    print('Model loaded')
    print(model)

    #Load our previous predictions and labels for calibration
    model_prediction_outputs = load("model_prediction_outputs_vgg_softmax_15.npy")
    actuaL_image_labels = load("actuaL_image_labels_vgg_softmax_15.npy")
    print("Calibration data loaded")

    prob_true, prob_pred = calibration_curve(actuaL_image_labels, model_prediction_outputs, f'Model before calibration')

    #This is the calibrator we use once it has been fed the data
    sigmoid_calibrator = SigmoidCalibrator(prob_pred, prob_true)

@rawhttp
def run(request):
    print(request)

    if request.method == 'POST':
        # grab the image from the request
        file_bytes = request.files["image"]
        image = Image.open(file_bytes).convert('RGB')

        # TODO: Build this step into the model itself
        # convert the image into the right sized numpy array for the model to work with
        new_image_size = (512, 512)
        image = image.resize(new_image_size)
        image_data = tf.keras.preprocessing.image.img_to_array(image)
        image_data = image_data.reshape(1, 512, 512, 3)
        image_data /= 255

        # return the predictions based on the model
        predictions = model.predict(image_data).tolist()

        #return the calibrated predictions instead
        predictions = sigmoid_calibrator.calibrate(predictions)
        return AMLResponse(json.dumps(predictions), 200)
    else:
        return AMLResponse("Bad Request", 500)
