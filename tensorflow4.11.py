#!/usr/bin/env python3
import tensorflow as tf

def runGraph()
	x = tf.Variable(3)
	y = tf.Varaible(4)
	f = x*x*y + y + 2

	# Run graph
	with tf.Session() as sess:
		x.initializer.run()
		y.initializer.run()
		result = f.eval()
		print(result)

def linearReg()
	import numpy as no
	from sklearn.datasets import fetch_california_housing
	import os
	import ssl

	# import data
	housing = fetch_california_housing()
	m, n = housing.data.shape
	housing_data_plus_bias = np.c_[np.ones((m,1)),housing.data]

	#build graph
	X = tf.constant(housing_data_plus_bias,dtype=tf.float32,name="X")
	y = tf.constant(housing.target.reshape(-1,1),dtype=tf.float32,name="y")
	XT = tf.transpose(X)
	theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)),XT),y)

	# Run graph
	with tf.Session() as sess:
		theta_value = theta.evel
		print(theta_value)