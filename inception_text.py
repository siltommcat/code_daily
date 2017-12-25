import tensorflow as tf
import numpy as np
import data_helper

voc_size = 10000
batch_size = 64
s_limit_len = 35
learn_rate = 0.05
n_class = 2
embedding_size = 128
filter_nums ={1:3, 3:3,5:3}

inputs = tf.placeholder(tf.int64,[None, s_limit_len], name="inputs")
labels = tf.placeholder(tf.int64,[None,n_class], name="labels")
keep_prob = tf.placeholder(tf.float32)

embedding_W = tf.Variable(tf.float32, [voc_size, embedding_size], name="embedding_w")
embedding_layer = tf.nn.embedding_lookup(embedding_W,inputs, name="embedding_layer")

#conv1
conv1_w = tf.Variable(tf.truncated_normal([1,embedding_size,1,filter_nums[1]]))
conv1_b = tf.Variable(tf.constant(0.1))
conv1 = tf.relu(tf.nn.conv2d(embedding_layer,conv1_w,[1,1,1,1],padding="VALID"))+conv1_b
#conv3
conv3_1w = tf.Variable(tf.truncated_normal([1,embedding_size,1,2]))
conv3_1b = tf.Variable(tf.constant(0.1))
conv3_1 = tf.relu(tf.nn.conv2d(embedding_layer,conv3_1w,[1,1,1,1],padding="VALID"))+conv3_1b
conv3_3w = tf.Variable(tf.truncated_normal([3,embedding_size,2,4]))
conv3_3b = tf.Variable(tf.constant(0.1))
conv3 = tf.relu(tf.nn.conv2d(conv3_1,conv3_3w, [1,1,1,1],padding="VALID")+ conv3_3b)
#conv5
conv5_3w = tf.Variable(tf.truncated_normal([3,embedding_size,2,4]))
conv5_3b = tf.Variable(tf.constant(0.1))
conv5_3 = tf.relu(tf.nn.conv2d(embedding_layer,conv3_3w, [1,1,1,1],padding="VALID")+ conv5_3b)
conv5_5w = tf.Variable(tf.truncated_normal([5,embedding_size]))
