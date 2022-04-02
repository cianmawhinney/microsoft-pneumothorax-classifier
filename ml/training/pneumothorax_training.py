import itertools
import os
import sys
import argparse
import subprocess

import matplotlib.pylab as plt
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("Keras Version:", keras.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

#Parse the arguments sent to the job

arg_data_folder = ""
arg_batch_size = ""

parser = argparse.ArgumentParser(description="Test training model script.")
parser.add_argument("--data-folder", action="store", required=True)
parser.add_argument("--batch-size", type=int, action="store", required=True)
parser.add_argument("--first-layer-neurons", type=int, action="store", required=False)
parser.add_argument("--second-layer-neurons", type=int, action="store", required=False)
parser.add_argument("--learning-rate", type=float, action="store", required=False)
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
      labels="inferred",
      label_mode="categorical",
      seed=123,
      image_size=IMAGE_SIZE,
      batch_size=1)

train_ds = build_dataset("train/")
class_names = tuple(train_ds.class_names)
train_size = train_ds.cardinality().numpy()
train_ds = train_ds.unbatch().batch(BATCH_SIZE)
train_ds = train_ds.repeat()

#Data preprocessing layers
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
preprocessing_model = tf.keras.Sequential([normalization_layer])

#Do / don't do data augmentation
do_data_augmentation = True
if do_data_augmentation:
  # preprocessing_model.add(
  #   tf.keras.layers.experimental.preprocessing.RandomRotation(5))
  preprocessing_model.add(
    tf.keras.layers.experimental.preprocessing.RandomTranslation(0, 0.2))
  preprocessing_model.add(
    tf.keras.layers.experimental.preprocessing.RandomTranslation(0.2, 0))
  # Like the old tf.keras.preprocessing.image.ImageDataGenerator(),
  # image sizes are fixed when reading, and then a random zoom is applied.
  # If all training inputs are larger than image_size, one could also use
  # RandomCrop with a batch size of 1 and rebatch later.
  preprocessing_model.add(
      tf.keras.layers.experimental.preprocessing.RandomZoom(0.2, 0.2))
  #preprocessing_model.add(
  #    tf.keras.layers.experimental.preprocessing.RandomFlip(mode="horizontal"))

#Mapping the images and their corresponding labels
train_ds = train_ds.map(lambda images, labels:
                        (preprocessing_model(images), labels))

#Setup validationn dataset
val_ds = build_dataset("test/")
valid_size = val_ds.cardinality().numpy()
val_ds = val_ds.unbatch().batch(BATCH_SIZE)
val_ds = val_ds.map(lambda images, labels:
                    (normalization_layer(images), labels))

# Get our base model and set it as untrainable
model_vgg19 = VGG19(weights = "imagenet", include_top=False, input_shape = (512,512,3))
for i in model_vgg19.layers:
  i.trainable=False

print("Building model with Vgg19 imagenet weights")

#Create our classificationn head on top of the base model
model = model_vgg19.output
model = Conv2D(32, (3, 3))(model)
model = (Activation('relu'))(model)
model = (MaxPool2D(pool_size=(2, 2)))(model)
model = Flatten()(model)
model = Dense(256, activation="relu")(model)
model= tf.keras.layers.Dropout(0.2)(model)
model = Dense(128, activation="relu")(model)
output_layer = Dense(2, activation="softmax")(model)
model_2 = Model(model_vgg19.input,output_layer)

#Print layers of our model
model_2.summary()

#Create checkpoints for model
filepath="./outputs"
model_checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_recall',  verbose=1, save_best_only=True, mode='max')

earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0.001, mode='max', patience=4, verbose=1, restore_best_weights=True)

class myCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
      super(myCallback, self).__init__()
      self.threshold = threshold

    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_recall') > self.threshold):
          print("Trainning Stopped. Val Recall = {} crossed threshold = {}".format(logs.get('val_recall'), self.threshold)) 
          self.model.stop_training = True

#Callback for tensorboard output         
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#Optimiser, precision and recall stats
adam = tf.keras.optimizers.Adam(lr=0.0002)
precision = tf.keras.metrics.Precision(name='precision')
recall = tf.keras.metrics.Recall(name='recall')

callback_list = [model_checkpoint, myCallback(threshold=0.99),tensorboard_callback] #earlystop,

#Compile our first model
model_2.compile(loss = "categorical_crossentropy", optimizer=adam, metrics=["accuracy", recall, precision])

steps_per_epoch = train_size // BATCH_SIZE
validation_steps = valid_size // BATCH_SIZE

#Do our first run of fitting the model
history = model_2.fit(train_ds, epochs=12,verbose=1,validation_data=val_ds,batch_size=64,callbacks=callback_list,steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

#Evaluate the fine tuned model
loss, accuracy, recall, precision = model_2.evaluate(val_ds)
print('Test accuracy :', accuracy)
print('Loss : ', loss)
print('Recall: ', recall)
print('Precision: ', precision)

#Set the top few layers of our base model to be trainable
model_vgg19.trainable = True
fine_tune_at = len(model_vgg19.layers) - 6
for layer in model_vgg19.layers[:fine_tune_at]:
  layer.trainable = False

#Recompile the model
adam = tf.keras.optimizers.Adam(lr=0.00002)
model_2.compile(loss = "categorical_crossentropy", optimizer=adam, metrics=["accuracy", recall, precision])
model_2.summary()

fine_tune_epochs = 12
initial_epochs = 12
total_epochs =  initial_epochs + fine_tune_epochs

#Perform the next fit of the model
history_fine = model_2.fit(train_ds,
                         epochs=total_epochs,
                         verbose=1,
                         initial_epoch=history.epoch[-1],
                         validation_data=val_ds,
                         batch_size=64,
                         callbacks=callback_list,
                         steps_per_epoch=steps_per_epoch, 
                         validation_steps=validation_steps)

#Evaluate the fine tuned model
loss, accuracy, *anything_else = model_2.evaluate(val_ds)
print('Test accuracy :', accuracy)
print('Loss : ', loss)
print('Other: ', anything_else)

