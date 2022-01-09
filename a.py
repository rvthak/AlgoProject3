
# Parameters
MODEL_LOCATION=''         # Path to a pre-trained stored model
DATASET_LOCATION=''       # Path to the stock proces dataset
SPLIT_PERCENT = 0.6       # The percentage on which the dataset time series gets split into train and test sets
TIME_SERIES_AMOUNT = 4    # The amount of time series that will be used to train the model
MODEL_EXPORT_FILE_PATH='A.model'

# Hyperparameters
EPOCHS = 5
BATCHSIZE = 10
WINDOW = 10
LAYER_SIZE = 50
RET_SEQUENCES_FLAG = True
DROPOUT_PERCENT = 0.1

# -------------------------------------------------------------------------------------------------------

# Imports
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, optimizers, losses, metrics

import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sklearn.preprocessing import MinMaxScaler

# -------------------------------------------------------------------------------------------------------

parser = ArgumentParser(prog='read_args')
parser.add_argument("--dataset_location_arg","-d", type=str, required=True)
parser.add_argument("--time_series_amount_arg", "-n", type=int, required=True)
parser.add_argument("--model_location_arg","-m",type=str)

args = parser.parse_args()

DATASET_LOCATION = args.dataset_location_arg
TIME_SERIES_AMOUNT = args.time_series_amount_arg

if args.model_location_arg is not None:
    MODEL_LOCATION=args.model_location_arg

# -------------------------------------------------------------------------------------------------------

# Mount the google drive to have access to the files
# from google.colab import drive
# drive.mount('/content/gdrive')

#python a.py -d ~/Downloads/nasdaq2007_17.csv -n 4 -m A.model

# Read the input file and create a dataframe
#Location = r'/content/gdrive/MyDrive/AlgoProject/datasets/nasdaq2007_17.csv'
#Location = r'/content/gdrive/MyDrive/datasets/nasdaq2007_17.csv'
#Location = r'c:/Users/giann/Downloads/nasdaq2007_17.csv'
#Location = r'~/Downloads/nasdaq2007_17.csv'
Location = DATASET_LOCATION

df=pd.read_csv(Location, sep='\t', header=None, index_col=0)

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

# Split the dataset into train and test lists
train_arr = []
test_arr  = []

for i, item in enumerate(df.iterrows()):
  arr = np.asarray( item[1:][0] )
  tup = np.split(arr, [ round(arr.size*SPLIT_PERCENT) ])

  train_arr.append(tup[0])
  test_arr.append(tup[1])

  # Keep only the requested amount of time series
  if( i == TIME_SERIES_AMOUNT-1 ):
    break

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
sc = MinMaxScaler(feature_range = (0, 1))

train_array_X = []
train_array_Y = []
test_array_X = []
test_array_Y = []

# For every Stock time series in the dataset
for i in range(len(train_arr) ):

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

# If a pre-trained model was provided => Load and use it
if MODEL_LOCATION != '':
  model = keras.models.load_model(MODEL_LOCATION)

# Else Create a new model and train it
else:

  # Initialize the model
  model = keras.Sequential()

  # Build the model
  model.add(layers.LSTM(units = WINDOW, return_sequences=RET_SEQUENCES_FLAG, input_shape=(WINDOW,1) ))
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

  # Train the model using each individual Stock time series 
  for i in range( len(train_array_X) ):
    print("\n (i) Training using time series: ", df.index[i], " : " , i)
    model.fit(train_array_X[i], train_array_Y[i], epochs=EPOCHS, batch_size=BATCHSIZE)

  # Save the model
  model.save(MODEL_EXPORT_FILE_PATH)

# -------------------------------------------------------------------------------------------------------

# Choose a random stock to predict
index = random.randint( 0, len(train_arr)-1 )
print("\n (i) Making predictions for stock: ", df.index[index], "\n" )

# Make a prediction for its future price
predicted_stock_price = model.predict( test_array_X[index] )

# Scale the time series values back to their original values
real = sc.inverse_transform(test_array_Y[index])
predicted = sc.inverse_transform(predicted_stock_price)

# Plot the result
title = str(df.index[index]) + " Stock Price Prediction"
plt.plot(range(1450), real, color = 'red', label = "Real Stock Price")
plt.plot(range(1450), predicted, color = 'blue', label = "Predicted Stock Price")
plt.title(title)
plt.xlabel('Time Units')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# -------------------------------------------------------------------------------------------------------

print("\n (i) Evaluating Model: ")
# Use evaluate() to get the average loss
average_loss = 0
for i in range( len(test_array_X) ):
  average_loss += model.evaluate(test_array_X[i], test_array_Y[i], batch_size=BATCHSIZE)

print("\n (i) Average evaluate() loss: ", average_loss/len(test_array_X), "\n" )
