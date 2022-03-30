import itertools
import os
import sys
import argparse
import subprocess
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


arg_data_folder = ""
arg_batch_size = ""

parser = argparse.ArgumentParser(description="Test training model script.")
parser.add_argument("--data-folder", action="store", required=True)
parser.add_argument("--batch-size", type=int, action="store", required=True)
args = parser.parse_args()

if args.data_folder:
  arg_data_folder=args.data_folder
  print(arg_data_folder)
else:
  print("Data folder not specified")

if args.batch_size:
  arg_batch_size=args.batch_size
  print(arg_batch_size)
else:
  print("Batch size not specified")

tf.keras.backend.clear_session()

BATCH_SIZE = arg_batch_size

IMAGE_SIZE = (512,512)

##Build keras dataset from local files

def build_dataset(subset):
  return tf.keras.preprocessing.image_dataset_from_directory(
      arg_data_folder + "/png_images/" + subset,
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

os.environ["MODEL_NAME"] = "vggbased_trans_zoom_tf"
os.environ["VERSION"] = "1"

# AZUREML_MODEL_DIR points to the folder ./azureml-models/$MODEL_NAME/$VERSION
model_folder = ("./model")
model = tf.keras.models.load_model(model_folder)
print('Model loaded')
print(model)


model_prediction_outputs = []
actuaL_image_labels_nested = []
actuaL_image_labels = []
print("Generating predictions, might take a while!")
for x,y in val_ds:
  prediction = model.predict(x).tolist()
  print("Prediction = " + str(prediction))
  model_prediction_outputs.append(prediction)
  actuaL_image_labels_nested.append(y)

def conf_matrix(test_y,predict_y):
  labels = [0,1]
  plt.figure(figsize=(8,6))
  C = confusion_matrix(test_y, predict_y)
  tn, fp, fn, tp = C.ravel()
  Precision = tp / (tp + fp)
  Recall = tp / (tp + fn)
  F1 = (2* Precision * Recall) / (Precision + Recall)
  sns.heatmap(C, annot=True, fmt='d')
  plt.xlabel('Predicted Class')
  plt.ylabel('Original Class')
  plt.title('Confusion matrix')
  plt.show()
  return Precision, Recall, F1

for batch2 in actuaL_image_labels_nested:
    for ele2 in batch2:
      actuaL_image_labels.append(ele2)

print("Calculating threshold")
threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# threshold_list = [0.5]
for threshold in threshold_list:
  val_pred_label = []
  for batch in model_prediction_outputs:
    for ele in batch:
      if ele[0] > threshold:
        val_pred_label.append(1)
      else:
        val_pred_label.append(0)

  print("actual_image_labels length = " + str(len(actuaL_image_labels)))
  print("val_pred_label length = " + str(len(val_pred_label)))

  print("\n\n" + "="*20 + "  Threshold Value = " + str(threshold) + "  " + "="*20)
  Precision, Recall, F1 = conf_matrix(actuaL_image_labels, val_pred_label)
  print("Precision = {:.4f}, Recall = {:.4f}, F1 = {:.4f} for Threshold = {}".format(Precision, Recall, F1, threshold))



