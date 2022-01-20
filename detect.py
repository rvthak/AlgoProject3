
# python detect.py -d ./datasets/nasdaq2007_17.csv -n 6 -mae 3

# -------------------------------------------------------------------------------------------------------

# Parameters
MODEL_LOCATION=''         # Path to a pre-trained stored model
DATASET_LOCATION=''       # Path to the stock proces dataset
SPLIT_PERCENT = 0.8       # The percentage on which the dataset time series gets split into train and test sets
TIME_SERIES_AMOUNT = 10   # The amount of time series that will be used to train the model

MODEL_SAVE_PATH = r'./PreTrained_Models/Anomaly/'
LOAD = True
SAVE = False

# Hyperparameters
EPOCHS = 5
BATCHSIZE = 100
WINDOW = 7
LAYER_SIZE = 20
DROPOUT_PERCENT = 0.1
THRESHOLD = 6

# -------------------------------------------------------------------------------------------------------

# Imports
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, optimizers, losses, metrics

import sys
import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# -------------------------------------------------------------------------------------------------------
#Parse the command line arguments using ArgumentParser
parser = ArgumentParser(prog='read_args')
parser.add_argument("--dataset_location_arg","-d", type=str, required=True)
parser.add_argument("--time_series_amount_arg", "-n", type=int, required=False)
parser.add_argument("--mae_arg", "-mae", type=float, required=True)
parser.add_argument("--query", "-q", type=str, required=False)

args = parser.parse_args()

DATASET_LOCATION = args.dataset_location_arg
THRESHOLD = args.mae_arg

if args.time_series_amount_arg is not None:
  TIME_SERIES_AMOUNT = args.time_series_amount_arg

if args.query is not None:
  QUERY = args.query

if args.time_series_amount_arg and args.query:
  print( " (!) Error: Missing arguments: You need to use either '-n' or '-q'.")
  exit()

# -------------------------------------------------------------------------------------------------------

# Read the input and load it into a dataframe
df=pd.read_csv(DATASET_LOCATION, sep='\t', header=None, index_col=0)

# -------------------------------------------------------------------------------------------------------

# Handle error input regarding TIME_SERIES_AMOUNT argument
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

# -------------------------------------------------------------------------------------------------------

# Split the given array into subsequences
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
# Returns a list containing the files in the given directory
def getModel(path):
  return os.listdir(path)

trained_list = getModel(MODEL_SAVE_PATH)
# -------------------------------------------------------------------------------------------------------

# Load a model from the given file
def loadModel(file):
  return keras.models.load_model(file)

# Create, compile and return a model 
def newModel():

  # Initialize the model
  model = keras.Sequential()

  # Build the model
  model.add(keras.layers.LSTM(units=LAYER_SIZE, input_shape=(WINDOW,1)))

  model.add(keras.layers.Dropout(rate=DROPOUT_PERCENT))

  model.add(keras.layers.RepeatVector(n=1))

  model.add(keras.layers.LSTM(units=LAYER_SIZE, return_sequences=True))

  model.add(keras.layers.Dropout(rate=DROPOUT_PERCENT))

  model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=1)))

  # Compile the model
  model.compile(loss='mae', optimizer='adam')

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

# Search the given dataframe for the given name index
def findIndex(df, name):
  for i, item in enumerate(df.index):
    if item == name:
      return i
  return -1


def findAnomalies(model, index):
  print("\n (i) Making predictions for stock: ", df.index[index], " : ", index, "\n" )

  # Make a prediction for its future price
  predicted_stock_price = model.predict( test_array_X[index] )
  predicted_stock_price = predicted_stock_price.reshape(predicted_stock_price.shape[0],1)

  sc.fit( train_arr[index].reshape(-1, 1) )

  # Scale the time series values back to their original values
  real = sc.inverse_transform(test_array_Y[index])
  predicted = sc.inverse_transform(predicted_stock_price)

  # Plot the result
  title = str(df.index[index]) + " Stock Price Prediction"
  plt.plot(range(len(real)), real, color = 'red', label = "Real Stock Price")
  plt.plot(range(len(predicted)), predicted, color = 'blue', label = "Predicted Stock Price")
  plt.title(title)
  plt.xlabel('Time Units')
  plt.ylabel('Stock Price')
  plt.legend()
  plt.show()

  predicted_stock_price = predicted_stock_price.reshape(predicted_stock_price.shape[0],1,1)

  #Build a dataframe that contains the loss and the anomalies
  test_mae_loss = np.mean(np.abs(real[WINDOW:] - predicted[WINDOW:]), axis=1)
  test_score_df = pd.DataFrame(index=(range(WINDOW, len(test_array_X[index]))))
  test_score_df['loss'] = test_mae_loss
  test_score_df['threshold'] = THRESHOLD
  test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
  test_score_df['close'] = real[WINDOW:] 

  #Plot the result
  plt.plot(range(len(test_mae_loss)), test_mae_loss, color = 'blue', label = "loss") 
  plt.axhline(y=THRESHOLD, color = 'red', label = "Threshold")
  plt.legend()
  plt.show()

  #Create a dataframe of the extreme values detected
  anomalies = test_score_df[test_score_df.anomaly == True]

  plt.plot(real, color='blue', label='close price')
  sns.scatterplot(anomalies.index, anomalies.close, color='red', label='anomaly')
  plt.legend()
  plt.show()

# -------------------------------------------------------------------------------------------------------
# Check if there is a model already stored
trained_model = []

if LOAD and trained_list:
  print(" (i) Loading Multi-Stock pretrained model")
  trained_model.append( loadModel( MODEL_SAVE_PATH + trained_list[0]) )
else:
  print(" (i) Creating and Training Multi-Stock model")
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

  model.fit(all_data, all_labels, epochs=EPOCHS, batch_size=BATCHSIZE)
  trained_model.append(model)

  if SAVE:
    # Save it as a file
    model.save(MODEL_SAVE_PATH + 'B.model')

# -------------------------------------------------------------------------------------------------------

# Make a prediction only for the given query
if args.query is not None:

  index = findIndex(df, QUERY)
  if index == -1:
    print(" (!) Error: The requested stock does not exist in the input file.")
    exit()
  else:
    findAnomalies(trained_model[0], index)
    exit()
  
else:
  # Choose '-n' random indexes
  indexes = getRandIndexes(TIME_SERIES_AMOUNT, len(train_arr))

  for index in indexes:
    findAnomalies(trained_model[0], index)

# -------------------------------------------------------------------------------------------------------

print("\n (i) Evaluating Model: ")
average_loss = 0
for i in range( len(test_array_X) ):
  average_loss += trained_model[0].evaluate(test_array_X[i], test_array_Y[i], batch_size=len(test_array_X[i]))

print("\n (i) Average evaluate() loss: ", average_loss/len(test_array_X), "\n" )
