import numpy as np
import tensorflow as tf
import random as rn
import numpy as np

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from keras.layers import Input, Dense,LSTM,RepeatVector,GRU,Dropout,Reshape
from keras.layers import*
from keras.models import Model
from keras.models import Sequential
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from deap import base, creator, tools, algorithms
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
import matplotlib.pyplot as plt
from math import sqrt
import random
import warnings
warnings.simplefilter("ignore", DeprecationWarning)


def parser(x):
	return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[:,0][-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = []
	for x in X:
   		 new_row = new_row+[i for i in x]
            
	new_row.append(value) 
	new_row_2 = np.array(new_row)
	new_row_2 = new_row_2.reshape((1,new_row_2.shape[0]))
	inverted = scaler.inverse_transform(new_row_2)
	return inverted[0, -1]


def create_dataset(dataset,features, look_back=1):
	dataset = np.insert(dataset,[0]*look_back,0)    
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	dataY= np.array(dataY)        
	dataY = np.reshape(dataY,(dataY.shape[0],features))
	dataset = np.concatenate((dataX,dataY),axis=1)  
	return dataset



# convert series to supervised learning
def series_to_supervised(data,features, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	x = np.zeros(features)
	for i in range(n_in):
		data = np.insert(data,x,0)
	data = data.reshape(int(data.shape[0]/features),features) 
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(batch_size, X.shape[0], X.shape[1])
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

def SMAPE(A, F):
    return 100/len(A) * np.sum(np.abs(F - A) / (np.abs(A) + np.abs(F)))


# compute RMSPE
def RMSPE(x,y):
	 return np.sqrt(np.mean(np.square(((x - y) / x))))*100




def run(): 

	units = 20

	batch_size = 219

	dropout = 0.1

	seq_len = 30

	epochs_pre = 917

	epochs_finetune = 244

	window_size = 0

	features = 8


	series = read_csv('pollution.csv', header=0, index_col=0)	
	raw_values = series.values

	# integer encode wind direction
	encoder = LabelEncoder()
	raw_values[:,4] = encoder.fit_transform(raw_values[:,4])

	# transform data to be stationary
	diff = difference(raw_values, 1)


	dataset = diff.values
	dataset = create_dataset(dataset,features,window_size)

	# frame as supervised learning
	reframed = series_to_supervised(dataset,features, seq_len, 1)
	drop = [i for  i in  range(seq_len*features+1,((seq_len+1)*features))]
	reframed.drop(reframed.columns[drop], axis=1, inplace=True)
	reframed = reframed.values

	# split into train and test sets
	train_size = 365*24*4
	train, test = reframed[0:train_size], reframed[train_size:]

	
	
	# transform the scale of the data
	scaler, train_scaled, test_scaled = scale(train, test)

	


	# split into input and outputs
	x_train,y_train = train_scaled[:,0:-1],train_scaled[:,-1]
	x_test,y_test = test_scaled[:,0:-1],test_scaled[:,-1]

	# reshape input to be 3D [samples, timesteps, features]
	x_train = x_train.reshape(x_train.shape[0],seq_len,features)
	x_test = x_test.reshape(x_test.shape[0],seq_len,features)

	print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


	# print('\nstart pretraining')
	# print('===============')

	# # train AE
	timesteps = x_train.shape[1]
	input_dim = x_train.shape[2]
	# AE = Sequential()
	# AE.add(CuDNNLSTM(units,batch_input_shape=( batch_size,timesteps, input_dim),stateful = False))
	# AE.add(RepeatVector(timesteps))
	# AE.add(CuDNNLSTM(input_dim,stateful = False,return_sequences = True))

	# AE.compile(loss='mean_squared_error', optimizer='Adam')

	# AE.fit(x_train, x_train,
	#                          epochs = epochs_pre,
	#                          batch_size = batch_size,
	#                          shuffle = True,
	#                          verbose = 1
	#                          )


	# trained_encoder = AE.layers[0]
	# weights = AE.layers[0].get_weights()



	# # Fine-turning
	# print('\nFine-turning')
	# print('============')

	# #build finetuning model
	# model = Sequential()
	# model.add(trained_encoder)
	# model.layers[-1].set_weights(weights)
	# model.add(Dropout(dropout))
	# model.add(Dense(1))

	# model.compile(loss='mean_squared_error', optimizer='Adam')

	# model.fit(x_train, y_train, epochs=epochs_finetune, batch_size = batch_size, verbose = 1,shuffle=True)

	# # save trained model
	# model.save('1layer.h5')

	model = load_model('1layer.h5')

	# redefine the model in order to test with one sample at a time (batch_size = 1)
	new_model = Sequential()
	new_model.add(CuDNNLSTM(units,batch_input_shape=( 1,timesteps, input_dim),stateful = False))
	new_model.add(Dropout(dropout))
	new_model.add(Dense(1))

	# copy weights
	old_weights = model.get_weights()
	new_model.set_weights(old_weights)



	# forecast the entire training dataset to build up state for forecasting
	print('Forecasting Training Data')   
	predictions_train = list()
	for i in range(len(y_train)):
		# make one-step forecast
		X = x_train[i]
		y= y_train[i]
		yhat = forecast_lstm(new_model, 1, X)
		# invert scaling
		yhat = invert_scale(scaler, X, yhat)
		# invert differencing
		yhat = inverse_difference(raw_values, yhat, len(raw_values)-i)
		# store forecast
		predictions_train.append(yhat)
		expected = raw_values[:,0][ i+1 ] 
		#print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

	# report performance
	rmse_train = sqrt(mean_squared_error(raw_values[:,0][1:len(train_scaled)+1], predictions_train))
	print('Train RMSE: %.5f' % rmse_train)
	# #report performance using RMSPE
	# RMSPE_train = RMSPE(raw_values[:,0][1:len(train_scaled)+1],predictions_train)
	# print('Train RMSPE: %.5f' % RMSPE_train)
	MAE_train = mean_absolute_error(raw_values[:,0][1:len(train_scaled)+1], predictions_train)
	print('Train MAE: %.5f' % MAE_train)
	# MAPE_train = MAPE(raw_values[:,0][1:len(train_scaled)+1], predictions_train)
	# print('Train MAPE: %.5f' % MAPE_train)
	SMAPE_train = SMAPE(raw_values[:,0][1:len(train_scaled)+1], predictions_train)
	print('Train SMAPE: %.5f' % SMAPE_train)

	# forecast the test data
	print('Forecasting Testing Data')
	predictions_test = list()
	for i in range(len(y_test)):
	    # make one-step forecast
		X = x_test[i]
		y= y_test[i]
		yhat = forecast_lstm(new_model, 1, X)
		# invert scaling
		yhat = invert_scale(scaler, X, yhat)
		# invert differencing
		yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
		# store forecast
		predictions_test.append(yhat)
		expected = raw_values[:,0][len(train) + i + 1]
		#print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

	# report performance using RMSE
	rmse_test = sqrt(mean_squared_error(raw_values[:,0][-len(test_scaled):], predictions_test))
	print('Test RMSE: %.5f' % rmse_test)
	#report performance using RMSPE
	# RMSPE_test = RMSPE(raw_values[:,0][-len(test_scaled):], predictions_test)
	# print('Test RMSPE: %.5f' % RMSPE_test)
	MAE_test = mean_absolute_error(raw_values[:,0][-len(test_scaled):], predictions_test)
	print('Test MAE: %.5f' % MAE_test)
	# MAPE_test = MAPE(raw_values[:,0][-len(test_scaled):], predictions_test)
	# print('Test MAPE: %.5f' % MAPE_test)
	SMAPE_test = SMAPE(raw_values[:,0][-len(test_scaled):], predictions_test)
	print('Test SMAPE: %.5f' % SMAPE_test)

	#predictions = np.concatenate((predictions_train,predictions_test),axis=0)
	
	# line plot of observed vs predicted
	fig, ax = plt.subplots(1)
	ax.plot(raw_values[:,0][-80:],'mo-', label='original',linewidth = 2 )
	ax.plot(predictions_test[-80:] ,'co-', label='predictions',linewidth = 2)
	#ax.axvline(x=len(train_scaled)+1,color='k', linestyle='--')
	ax.legend(loc='upper right')
	ax.set_title('PM2.5 hourly concentration prediction from 28/12/2014 to 31/12/2014')
	ax.set_ylabel('PM2.5 concentration')
	plt.show()


run()






