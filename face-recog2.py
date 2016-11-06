import os
import pandas as pd
import pickle
import numpy as np
from numpy import *
from six.moves import xrange 
import tensorflow as tf
import time
dir_path = os.path.dirname(os.path.realpath(__file__))+"/"

FTRAIN=dir_path+"training.csv"
FTEST=dir_path+"test.csv"
FIDLOOKIP=dir_path+"IdLookupTable.csv"
cols_file=dir_path+'cols.pkl'

NUMBER_OF_FILTERS=1
BATCH_SIZE=50
EVAL_BATCH_SIZE=50
VAL_PERCENTAGE=0.2
KEEP_PROB=0.5

MAX_EPOCH=1
EPOCH=0


def load_data(test=False):

	fname=FTEST if test else FTRAIN
	print(fname)
	df=pd.read_csv(fname)
	
	cols=df.columns[:-1]
	
	df["Image"]= df["Image"].apply(lambda im: np.fromstring(im, sep=" ")/255.0)
	
	df=df.dropna()
	
	NUMBER_OF_PIXELS=len(df["Image"].iloc[0])
	number_of_pictures=df.shape[0]
	PIXELS_PER_AXIS=sqrt(NUMBER_OF_PIXELS)
	
	#print(number_of_pictures)
	#print(PIXELS_PER_AXIS)
	if not test:

		df=df.iloc[np.random.permutation(len(df))]
		df.reset_index(drop=True)
		
		y=df[cols].values/96.0

		pickle.dump(cols,open(cols_file,'wb+') )

	else:
		y=None
	
	X = np.vstack(df["Image"])	#of shape (2140, 9216) 
	number_of_pictures=X.shape[0]
	PIXELS_PER_AXIS=sqrt(X.shape[1])
	#print(number_of_pictures)
	#print(PIXELS_PER_AXIS)
	#print(X.shape)
	X=X.reshape(-1,PIXELS_PER_AXIS,PIXELS_PER_AXIS,NUMBER_OF_FILTERS)	#the -1 here just means it's unspecified, you're leaving it up to the computer to determine how many it needs.
	
	return X,y
	
	
x_data,y_data=load_data()
data_size=x_data.shape[0]
VALIDATION_QTY=int(VAL_PERCENTAGE*data_size)


x_val=x_data[:VALIDATION_QTY]
y_val=y_data[:VALIDATION_QTY,:]
x_train=x_data[VALIDATION_QTY:]
y_train=y_data[VALIDATION_QTY:,:]

train_size = y_train.shape[0]

###TF parameters
NUMBER_OF_CLASSES=y_train.shape[1]



PICTURE_SIZE=x_train.shape[1]			#remember that this has 4 dimensions, [-1,96,96,1]
NUMBER_OF_PIXELS=np.power(PICTURE_SIZE,2)		

print('PICTURE_SIZE')
print(PICTURE_SIZE)
print('NUMBER_OF_PIXELS')
print(NUMBER_OF_PIXELS)


def weight_variable(shape):
	initial=tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)
	
def bias_variable(shape):
	initial=tf.constant(0.1,shape=shape)
	return tf.Variable(initial)
def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")
def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

	
def flat_mult(shapes):
	i=1
	flat_dim_so_far=1
	while i<=len(shapes)-1:	#-1 because we don't care for the shape[0], it's just the number of instances
		flat_dim_so_far=flat_dim_so_far*int(shapes[i])
		i+=1
	return flat_dim_so_far
	
def eval_in_batches(x_val_ph,sess,y_val_h,x_val):

	if EVAL_BATCH_SIZE>x_val.shape[0]:
		raise ValueError("your Eval batch size is bigger than the amount of x validation data")
	predictions=np.zeros((x_val.shape[0],NUMBER_OF_CLASSES))
	
	for step in xrange(x_val.shape[0]//EVAL_BATCH_SIZE):
		begin=step*EVAL_BATCH_SIZE
		end=begin+EVAL_BATCH_SIZE
		x_val_batch=x_val[begin:end,...]
		feed_dict={x_val_ph:x_val_batch,
		}
		predictions[begin:end]=sess.run(y_val_h,feed_dict=feed_dict)
		
	#remainder
	remainder=x_val.shape[0]%EVAL_BATCH_SIZE
	x_val_batch=x_val[-remainder:,...]
	predictions[-remainder:]=sess.run([y_val_h],feed_dict=feed_dict)
	
	return predictions
	
def error_mesure(val_result,y_val):
	np.sum(np.power(val_result-y-val,2))/(2*val_result.shape[0])
		
x_train_ph=tf.placeholder(tf.float32,[None,PICTURE_SIZE,PICTURE_SIZE,NUMBER_OF_FILTERS])
x_val_ph=tf.placeholder(tf.float32,[None,PICTURE_SIZE,PICTURE_SIZE,NUMBER_OF_FILTERS])
	
y_train_ph=tf.placeholder(tf.float32,[None,NUMBER_OF_CLASSES])
y_val_ph=tf.placeholder(tf.float32,[None,NUMBER_OF_CLASSES])
	
global_step=tf.Variable(0,trainable=False)



conv1_w=weight_variable([3,3,1,32])
conv1_b=bias_variable([32])

conv2_w=weight_variable([2,2,32,64])
conv2_b=bias_variable([64])


nn2_weights=weight_variable([NUMBER_OF_PIXELS,NUMBER_OF_CLASSES])
nn2_bias=bias_variable([NUMBER_OF_CLASSES])


def model(data,test=False):
	#layer 1
	
	h_conv1=tf.nn.relu(conv2d(data,conv1_w)+conv1_b)
	h_pool1=max_pool_2x2(h_conv1)
	
	#layer 2

	h_conv2=tf.nn.relu(conv2d(h_pool1,conv2_w)+conv2_b)
	h_pool2=max_pool_2x2(h_conv2)
			
	print('h_pool2.get_shape()')		
	print(h_pool2.get_shape())		
	flattened_dimension_for_this_layer=	flat_mult(h_pool2.get_shape())	
	
	print('flattened_dimension_for_this_layer')
	print(flattened_dimension_for_this_layer)

	h_pool2_flat=tf.reshape(h_pool2,[-1,flattened_dimension_for_this_layer])
	
	print('h_pool2_flat.get_shape()')
	print(h_pool2_flat.get_shape())
	
	#layer 3 densely connected layer

	#print('h_pool2_flat.get_shape()')
	#print(h_pool2_flat.get_shape())
	
	global nn1_weights,nn1_bias
	nn1_weights=weight_variable([int(h_pool2_flat.get_shape()[1]) ,NUMBER_OF_PIXELS])
	nn1_bias=bias_variable([NUMBER_OF_PIXELS])


	h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,nn1_weights)+nn1_bias)

	if not test:
		h_fc1=tf.nn.dropout(h_fc1,KEEP_PROB)

	return tf.matmul(h_fc1, nn2_weights) + nn2_bias	

	
	
y_train_h=model(x_train_ph)	
y_val_h=model(x_val_ph,test=True)

print('y_train_h.get_shape')
print(y_train_h.get_shape())
print('y_train_ph.get_shape()')
print(y_train_ph.get_shape())

error=tf.reduce_mean(tf.reduce_sum(tf.square(y_train_h-y_train_ph),1))
	
regularizers=(tf.nn.l2_loss(nn1_weights)+tf.nn.l2_loss(nn1_bias) +
				tf.nn.l2_loss(nn2_weights)+tf.nn.l2_loss(nn2_bias))
				
error+=1e-7*regularizers

learning_rate=tf.train.exponential_decay(
	1e-3,                      # Base learning rate.
	global_step * BATCH_SIZE,  # Current index into the dataset.
	train_size,                # Decay step.
	0.95,                      # Decay rate.
	staircase=True)
	
train_step=tf.train.AdamOptimizer(learning_rate,0.95).minimize(error,global_step=global_step)

init=tf.initialize_all_variables()
sess=tf.InteractiveSession()
sess.run(init)

loss_train_record=list()
loss_val_record=list()

start_time=time.gmtime()

#early stopping

best_valid= np.inf
best_valid_epoch=0


while EPOCH<MAX_EPOCH:

	shuffled_index=np.random.permutation(train_size)
	x_data=x_data[shuffled_index]
	y_data=y_data[shuffled_index]
	
	for step in xrange(train_size//BATCH_SIZE):
		offset=BATCH_SIZE*step
		
		x_batch=x_data[offset:(offset+BATCH_SIZE),...]
		y_batch=y_data[offset:(offset+BATCH_SIZE)]
	
		print('x_batch.shape')
		print(x_batch.shape)
		print('y_batch.shape')
		print(y_batch.shape)
		feed_dict={	x_train_ph:x_batch,
					y_train_ph:y_batch,
					}
					
		_,loss_train,current_learning_rate=sess.run([train_step,error,learning_rate],feed_dict=feed_dict)
		
	#error=tf.reduce_mean(tf.reduce_sum(tf.square(y_train_h-y_train_ph),1))
	
	val_result=eval_in_batches(x_val_ph,sess,y_val_h,x_val)
	
	loss_valid=error_mesure(val_result,y_val)

	print('EPOCH')
	print(EPOCH)
	print('loss_train')
	print(loss_train.eval())
	print('loss_valid')
	print(loss_valid.eval())

	print('learning rate')
	print(current_learning_rate.eval())
	
	
	EPOCH+=1

print('train finish')

end_time = time.gmtime()
print (time.strftime('%H:%M:%S', start_time))
print (time.strftime('%H:%M:%S', end_time))
