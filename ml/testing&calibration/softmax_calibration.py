import itertools
import os
import sys
import argparse
import subprocess
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.calibration import calibration_curve

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

tf.keras.backend.clear_session()

BATCH_SIZE = 64

IMAGE_SIZE = (512,512)

##Build keras dataset from local files

def build_dataset(subset):
  return tf.keras.preprocessing.image_dataset_from_directory(
      subset,
      label_mode="binary",
      seed=123,
      image_size=IMAGE_SIZE,
      batch_size=1)

#Data preprocessing layers
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
preprocessing_model = tf.keras.Sequential([normalization_layer])

#Setup validationn dataset
val_ds = build_dataset("test/")
valid_size = val_ds.cardinality().numpy()
val_ds = val_ds.unbatch().batch(BATCH_SIZE)
val_ds = val_ds.map(lambda images, labels:
                    (normalization_layer(images), labels))

# AZUREML_MODEL_DIR points to the folder ./azureml-models/$MODEL_NAME/$VERSION
model_folder = ("./model")
model = tf.keras.models.load_model(model_folder)
print('Model loaded')
print(model)

#Evaluate the fine tuned model
# loss, accuracy, recall, precision = model.evaluate(val_ds)
# print('Test accuracy :', accuracy)
# print('Loss : ', loss)
# print('Recall: ', recall)
# print('Precision: ', precision)

model_prediction_outputs_nested = []
model_prediction_outputs = []
actuaL_image_labels_nested = []
actuaL_image_labels = []
print("Generating predictions, might take a while!")
for x,y in val_ds:
  prediction = model.predict(x).tolist()
  print("Prediction = " + str(prediction))
  print("Label = " + str(y.numpy()))
  labels = y.numpy()
  for label in labels:
    actuaL_image_labels.append(label[0])
  model_prediction_outputs_nested.append(prediction)
  actuaL_image_labels_nested.append(y)

print("Actual image labels nested = " + str(actuaL_image_labels_nested))
print("Actual image labels = " + str(actuaL_image_labels))

for batch in model_prediction_outputs_nested:
  for ele in batch:
      model_prediction_outputs.append(ele[1])

model_prediction_outputs = np.array(model_prediction_outputs)
actuaL_image_labels = np.array(actuaL_image_labels)

print("Model_prediction_outputs = " + str(model_prediction_outputs))
print("Actual_image_labels = " + str(actuaL_image_labels))

prob_true, prob_pred = calibration_curve(actuaL_image_labels, model_prediction_outputs, n_bins=10)
sigmoid_calibrator = SigmoidCalibrator(prob_pred, prob_true)
isotonic_calibrator = IsotonicCalibrator(prob_pred, prob_true)
    
sigmoid_calibrated = sigmoid_calibrator.calibrate(model_prediction_outputs)
isotonic_calibrated = isotonic_calibrator.calibrate(model_prediction_outputs)

print("Sigmoid calibrated = " + str(sigmoid_calibrated))
print("Isotonic calibrated = " + str(isotonic_calibrated))

def _get_accuracy(y, preds):
  return np.mean(np.equal(y.astype(np.bool), preds >= 0.5))

print("Score after sigmoid = " + str(_get_accuracy(actuaL_image_labels, sigmoid_calibrated)))
print("Score after isotonic = " + str(_get_accuracy(actuaL_image_labels, sigmoid_calibrated)))


