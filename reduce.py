
# python reduce.py -d ./datasets/nasd_input.csv -q ./datasets/nasd_query.csv -od ./datasets/enc_input.csv -oq ./datasets/enc_query.csv

# -------------------------------------------------------------------------------------------------------

# Parameters

DATASET_LOCATION   = '' # Input dataset Path 
QUERYSET_LOCATION  = '' # Query dataset Path
ENC_INPUT_LOCATION = '' # Encoded Input dataset Path
ENC_QUERY_LOCATION = '' # Encoded Query dataset Path

SPLIT_PERCENT = 0.8 # The percentage on which the dataset time series gets split into train and test sets

MODEL_SAVE_PATH = r'./PreTrained_Models/Conv/'
LOAD = True
SAVE = False

# Hyperparameters
EPOCHS = 4
BATCHSIZE = 100
WINDOW = 10

# -------------------------------------------------------------------------------------------------------

# Imports
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, optimizers, losses, metrics, models, regularizers

import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sklearn.preprocessing import MinMaxScaler, RobustScaler
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# -------------------------------------------------------------------------------------------------------

# Parse the Args
parser = ArgumentParser(prog='read_args')
parser.add_argument("--dataset_location",  "-d", type=str, required=True)
parser.add_argument("--queryset_location", "-q", type=str, required=True)
parser.add_argument("--enc_dataset_location", "-od", type=str, required=True)
parser.add_argument("--enc_queryset_location", "-oq", type=str, required=True)

args = parser.parse_args()

DATASET_LOCATION   = args.dataset_location
QUERYSET_LOCATION  = args.queryset_location
ENC_INPUT_LOCATION = args.enc_dataset_location
ENC_QUERY_LOCATION = args.enc_queryset_location
IN_FILES = [DATASET_LOCATION, QUERYSET_LOCATION]
OUT_FILES = [ENC_INPUT_LOCATION, ENC_QUERY_LOCATION]

# -------------------------------------------------------------------------------------------------------

# Split the given array into sub-arrays of size window with the window+1 value as its label
def windowSet(arr, window):
  data = []

  for i in range(window, len(arr)):
    sub = []

    for j in range(window, 0, -1):
      sub.append( arr[i-j] )

    data.append( sub )

  return np.array(data).reshape( len(data), WINDOW )

# Split the given array into non-overlapping window-sized sub-arrays
def nonCoverWin(arr, window):
  data = []

  for i in range(0, len(arr), window):
    sub = []

    for j in range(0, window):
      sub.append( arr[i+j] )

    data.append( sub )

  return np.array(data).reshape( len(data), WINDOW )

# Returns a list containing the files in the given directory
def getModels(path):
  return os.listdir(path)

# Load a model from the given file
def loadModel(file):
  return keras.models.load_model(file)

# -------------------------------------------------------------------------------------------------------

# Load the input files
df = []
df.append( pd.read_csv(DATASET_LOCATION,  sep='\t', header=None, index_col=0) )
df.append( pd.read_csv(QUERYSET_LOCATION, sep='\t', header=None, index_col=0) )

# Split the dataset in windows
dataset = []
for dts in df:
  tmp = []
  for i, item in enumerate(dts.iterrows()):
    arr = np.asarray( item[1:][0] )
    tmp.append( windowSet(arr, WINDOW) )
  dataset.append(tmp)

# -------------------------------------------------------------------------------------------------------
encoder = []

# Check if there is a model stored:
pre_trained = getModels(MODEL_SAVE_PATH)

if LOAD and pre_trained:
  print("\n (i) Loading pretrained model")
  # If there is a pre-trained model stored => Load it
  encoder = loadModel( MODEL_SAVE_PATH + pre_trained[0]) 
else:
  print("\n (i) Creating and Training new model")
  # Train a new model using the provided dataset

  # Scale the data values to [0,1]
  sc = MinMaxScaler(feature_range = (0, 1))

  first = True
  train_arr = []
  test_arr  = []

  # Split the dataset into train and test sets and normalize it
  for i, stock in enumerate(dataset[0]):

    tmp_train, tmp_test = np.split(stock, [round(len(stock)*SPLIT_PERCENT)])
    #print(tmp_train.shape, tmp_test.shape)

    if not first:
      train_arr = np.concatenate((train_arr, sc.fit_transform(tmp_train)) )
      test_arr  = np.concatenate(( test_arr, sc.transform(tmp_test )) )
    else:
      train_arr = sc.fit_transform(tmp_train)
      test_arr  = sc.transform(tmp_test)
      first = False

# -------------------------------------------------------------------------------------------------------

  # Create a model - 1D Convolutional autoencoder

  input_window = layers.Input(shape=(WINDOW,1))

  x = layers.Conv1D(16, 3, activation="relu", padding="same")(input_window)
  x = layers.MaxPooling1D(2, padding="same")(x)
  x = layers.Conv1D(1, 3, activation="relu", padding="same")(x)
  encoder = layers.MaxPooling1D(2, padding="same", name='encoder')(x)

  x = layers.Conv1D(1, 3, activation="relu", padding="same")(encoder)
  x = layers.UpSampling1D(2, name='up1')(x) 
  x = layers.Conv1D(16, 2, activation='relu', name='conv4')(x)
  x = layers.UpSampling1D(2, name='up3')(x) # 10 dims
  decoder = layers.Conv1D(1, 4, activation='sigmoid', padding='same', name='output')(x) # 10 dims


  autoencoder = models.Model(input_window, decoder)
  autoencoder.summary()

  autoencoder.compile(optimizer='adam', loss='mae')

  # Train the model
  autoencoder.fit(train_arr, train_arr, epochs=EPOCHS, batch_size=BATCHSIZE, shuffle=True) # , validation_data=(stock, stock)

  # Test the model
  autoencoder.evaluate(test_arr,test_arr, batch_size=BATCHSIZE)

  # Extract the Encoder from the trained model
  encoder = models.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)
  encoder.compile(optimizer='adam', loss='mae')

  # Save it as a file
  if SAVE:
    encoder.save(MODEL_SAVE_PATH + 'C.model')

# -------------------------------------------------------------------------------------------------------

print(" (i) Encoding Data...")

# For each file
for i, dft in enumerate(df):

  # Create the correct output file
  f = open(OUT_FILES[i], "w")
  print(" (i) Processing: " + IN_FILES[i])

  # For each existing stock
  for index, stock in enumerate( dft.iterrows() ):
    # Write its name
    f.write( (df[i]).index[index] + "\t" )

    # Get its original values
    vals = np.asarray( stock[1:][0] )

    # Break them in windows
    win = nonCoverWin(vals, WINDOW)

    # Pass them through the encoder
    enc = encoder.predict(win)

    # Write the resulting values in the csv
    for v in enc:
      f.write( str(v[0][0]) + "\t" + str(v[1][0]) + "\t" + str(v[2][0]) + "\t")

    f.write("\n")

  f.close()
  print("     Exported to file: " + OUT_FILES[i])
