
#python forecast.py -d ./datasets/nasdaq2007_17.csv -n 6

# -------------------------------------------------------------------------------------------------------

# Parameters

DATASET_LOCATION=''       # Path to the stock proces dataset
SPLIT_PERCENT = 0.8       # The percentage on which the dataset time series gets split into train and test sets
TIME_SERIES_AMOUNT = 5    # The amount of time series that will be used to train the model
LOAD = True               # If 'True', load any saved models
SAVE = False              # If 'True', save the models after training 

MODEL_SAVE_PATH = r'/home/pigeon/Downloads/PreTrained_Models'
RET_SEQUENCES_FLAG = True

# Hyperparameters
SINGL_EPOCHS = 7
MULTI_EPOCHS = 5
BATCHSIZE = 100
WINDOW = 7
LAYER_SIZE = 20
DROPOUT_PERCENT = 0.1

# -------------------------------------------------------------------------------------------------------

# Imports
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, optimizers, losses, metrics

import re
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

parser = ArgumentParser(prog='read_args')
parser.add_argument("--dataset_location_arg","-d", type=str, required=True)
parser.add_argument("--time_series_amount_arg", "-n", type=int, required=True)

args = parser.parse_args()

DATASET_LOCATION = args.dataset_location_arg
TIME_SERIES_AMOUNT = args.time_series_amount_arg

# -------------------------------------------------------------------------------------------------------

# Read the input and load it into a dataframe
df=pd.read_csv(DATASET_LOCATION, sep='\t', header=None, index_col=0)

# -------------------------------------------------------------------------------------------------------

# Check input arg value
if TIME_SERIES_AMOUNT <= 0 :
  print("(!) Fatal error: Illegal 'TIME_SERIES_AMOUNT' value provided.")
  exit()
elif TIME_SERIES_AMOUNT > len(df):
  print("(!) Error: 'TIME_SERIES_AMOUNT' value surpasses available time series amount.")
  print("    Using total available time series as 'TIME_SERIES_AMOUNT'...")
  TIME_SERIES_AMOUNT = len(df)

# -------------------------------------------------------------------------------------------------------

print("\n (i) Splitting the dataset")

# Split the dataset into train and test lists
train_arr = []
test_arr  = []

for i, item in enumerate(df.iterrows()):
  arr = np.asarray( item[1:][0] )

  tup = np.split(arr,[ round(arr.size*SPLIT_PERCENT) ])
  train_arr.append(tup[0])
  test_arr.append(tup[1])

# -------------------------------------------------------------------------------------------------------

# Split the given array into sub-arrays of size window with the window+1 value as its label
def windowSet(arr, window):
  data = []
  labels = []

  for i in range(window, len(arr)):
    sub = []

    for j in range(window, 0, -1):
      sub.append( arr[i-j] )

    data.append( sub )
    labels.append( arr[i] )

  return ( np.array(data) , np.array(labels) )

# -------------------------------------------------------------------------------------------------------

# Scale and split each stock time series and store them in lists 

# Used to Normalize the data by scaling it to [0,1]
sc = RobustScaler()

train_array_X = []
train_array_Y = []
test_array_X = []
test_array_Y = []

# For every Stock time series in the dataset
for i in range(len(train_arr) ):

  #sc.fit( np.concatenate( (train_arr[i], test_arr[i])).reshape(-1, 1) )

  # Scale the data values to [0,1]
  scaled_train = sc.fit_transform( train_arr[i].reshape(-1, 1) )
  scaled_test = sc.transform( test_arr[i].reshape(-1, 1) )

  # Split the data to window-sized arrays and their true-next as their corresponding label
  tmp_data_trn, tmp_labels_trn = windowSet( scaled_train, WINDOW )
  tmp_data_tst, tmp_labels_tst = windowSet( scaled_test, WINDOW )

  # Store them in the lists
  train_array_X.append(tmp_data_trn)
  train_array_Y.append(tmp_labels_trn)
  test_array_X.append(tmp_data_tst)
  test_array_Y.append(tmp_labels_tst)

# -------------------------------------------------------------------------------------------------------

# Returns a list containing the files in the given directory
def getModels(path):
  return os.listdir(path)

singl_trained_list = getModels(MODEL_SAVE_PATH+'/Single')
multi_trained_list = getModels(MODEL_SAVE_PATH+'/Multi')

# -------------------------------------------------------------------------------------------------------

# Load a model from the given file
def loadModel(file):
  return keras.models.load_model(file)

# Create, compile and return a model 
def newModel():

   # Initialize the model
  model = keras.Sequential()

  # Build the model
  model.add(layers.LSTM(units = WINDOW, return_sequences=RET_SEQUENCES_FLAG, input_shape=(WINDOW,1) ))
  model.add(layers.Dropout(DROPOUT_PERCENT))

  model.add(layers.LSTM(units = LAYER_SIZE, return_sequences=RET_SEQUENCES_FLAG))
  model.add(layers.Dropout(DROPOUT_PERCENT))

  model.add(layers.LSTM(units = LAYER_SIZE, return_sequences=RET_SEQUENCES_FLAG))
  model.add(layers.Dropout(DROPOUT_PERCENT))

  model.add(layers.LSTM(units = LAYER_SIZE, return_sequences=RET_SEQUENCES_FLAG))
  model.add(layers.Dropout(DROPOUT_PERCENT))

  model.add(layers.LSTM(units = LAYER_SIZE))
  model.add(layers.Dropout(DROPOUT_PERCENT))

  model.add(layers.Dense(units = 1))

  # Compile the model
  model.compile(optimizer = 'adam', loss = 'mean_squared_error' )

  return model

# Get '-n' unique random indexes in the given range
def getRandIndexes(amount, range_end):
  selected = []

  i=0;
  while( i < amount):
    index = random.randint(0, range_end-1)
    if index not in selected:
      selected.append(index)
      i+=1

  return selected

# Get the saved model indexes by removing all non-numerical chars 
def extractIndexes(fname):
  return int(re.sub("[^0-9]", "", fname))

# -------------------------------------------------------------------------------------------------------
indexes = []

# < Single Stock Mode: >
# Train a single Model per stock and use it to make predictions
# Train and predict only "-n" amount of stocks

# We start by checking if there are any supported models stored
singl_trained_model = []

if LOAD and singl_trained_list:
  print(" (i) Loading Single-Stock pretrained models")
  # If there are pre-trained models stored => Load them
  loaded = 0
  for file in singl_trained_list:
    singl_trained_model.append( loadModel( MODEL_SAVE_PATH + '/Single/' + file) )
    loaded+=1
    if loaded == TIME_SERIES_AMOUNT:
      break
else:
  print(" (i) Creating and Training Single-Stock models")
  # If there aren't any models stored => Create and train the required amount of models

  # Choose '-n' random indexes
  indexes = getRandIndexes(TIME_SERIES_AMOUNT, len(train_arr))

  # For each chosen stock
  for i in indexes:
    # Create a model
    model = newModel()

    print("\n (i) Training single-stock model: ", df.index[i], " : ", i)

    # Train it using only the selected random stock
    model.fit(train_array_X[i], train_array_Y[i], epochs=SINGL_EPOCHS, batch_size=BATCHSIZE)

    # Save the model in a file
    if SAVE:
      model.save(MODEL_SAVE_PATH + '/Single' + '/A_' + str(i) + '.model')

    # And store it
    singl_trained_model.append(model)

# < Multiple Stock Mode: >
# Train a single model using all the available stock prices
# Make predictions for "-n" stocks only

# Check if there is a model already stored
multi_trained_model = []

if LOAD and multi_trained_list:
  print("\n (i) Loading Multi-Stock pretrained model")
  multi_trained_model.append( loadModel( MODEL_SAVE_PATH + '/Multi/' + multi_trained_list[0]) )
else:
  print("\n (i) Creating and Training Multi-Stock model")

  # Create a model
  model = newModel()

  all_data   = []
  all_labels = []

  # Train it using all the available stocks
  for i in range( len(train_array_X) ):
    if i == 0:
      all_data   = train_array_X[i]
      all_labels = train_array_Y[i]
    else:
      all_data   = np.concatenate( (all_data,   train_array_X[i]))
      all_labels = np.concatenate( (all_labels, train_array_Y[i]))

  model.fit(all_data, all_labels, epochs=MULTI_EPOCHS, batch_size=BATCHSIZE)

  # Save the model in a file
  if SAVE:
    model.save(MODEL_SAVE_PATH + '/Multi' + '/A.model')

  multi_trained_model.append(model)

# -------------------------------------------------------------------------------------------------------

# In case the program loaded pre-trained models => Get their indexes
if not indexes:
  loaded = 0
  for fname in singl_trained_list:
    indexes.append( extractIndexes(fname) )
    loaded+=1
    if loaded == TIME_SERIES_AMOUNT:
      break

# If the user has requested more single models than are currently loaded => Error
if LOAD and (len(singl_trained_model) < TIME_SERIES_AMOUNT) :
  print(" (!) Error: User requested " + str(TIME_SERIES_AMOUNT) + " models, but only " + str(len(singl_trained_model)) + " saved models were found.")
  print("     Please re-run the program using '-n " + str(len(singl_trained_model)) + "', or train more models")
  exit()

# For each selected stock => Make a prediction using each model
for i, index in enumerate(indexes):
  print("\n (" + str(i+1) + ") Making predictions for stock: ", df.index[index], " : ", index)

  # Make predictions
  single_predicted = singl_trained_model[i].predict( np.concatenate( (train_array_X[index], test_array_X[index]) ))
  multi_predicted  = multi_trained_model[0].predict( np.concatenate( (train_array_X[index], test_array_X[index]) ))
  real = np.concatenate( (train_array_Y[index], test_array_Y[index]) )

  # Scale the time series values back to their original values
  sc.fit( train_arr[index].reshape(-1, 1) )

  real   = sc.inverse_transform(real)
  single = sc.inverse_transform(single_predicted)
  multi  = sc.inverse_transform(multi_predicted)

  # Plot the result
  title = str(df.index[index]) + " Stock Price Prediction"
  plt.plot(range(len(real)), real,   color = 'red'  , label = "Real Stock Price")
  plt.plot(range(len(single)), single, color = 'green', label = "Single-Predicted Stock Price")
  plt.plot(range(len(multi)), multi,  color = 'blue' , label = "Multi-Predicted Stock Price")
  plt.axvline(x=round(arr.size*SPLIT_PERCENT), color = 'pink')
  plt.title(title)
  plt.xlabel('Time Units')
  plt.ylabel('Stock Price')
  plt.legend()
  plt.show()

# -------------------------------------------------------------------------------------------------------

print("\n (i) Evaluating Multi-Model: ")
# Use evaluate() to get the average loss
average_loss = 0
for i in range( len(test_array_X) ):
  average_loss += multi_trained_model[0].evaluate(test_array_X[i], test_array_Y[i], batch_size=len(test_array_X[i]))

print("\n (i) Average evaluate() loss: ", average_loss/len(test_array_X), "\n" )
