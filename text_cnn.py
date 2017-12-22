import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
import data_helper

n_class = 2
learning_rate = 0.001
s_limit_len = 10
word_embedding_size = 100
voc_size = 7000
filter_nums = 4
def get_weights(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))
def get_bias(shape):
    return tf.Variable(tf.constant(0.1))

def conv2d(input_x, W):
    return tf.nn.conv2d(input_x,W,strides=[1,1,1,1],padding="VALID")

def maxpooling(x,kszie,strides):
    return tf.nn.max_pool(x,ksize=kszie,strides=strides,padding="VALID")

inputs = tf.placeholder(tf.int32,[None,s_limit_len],name="inputs")
labels = tf.placeholder(tf.int32,[None,n_class],name="label_one-hot")
keep_prob = tf.placeholder(tf.float32,name="keep_prob")

embedding_w = tf.Variable(tf.truncated_normal([voc_size,word_embedding_size],stddev=0.1,dtype=tf.float32))
#这里需要多延展一个维度
embedding_layer = tf.expand_dims(tf.nn.embedding_lookup(embedding_w,inputs),-1)
# convoltional layers


conv_dict = {1:2,3:3,5:4}
filter_types = [1,3,5,7]
filter_types = [1,2,3,3]

# conv1_W = tf.Variable(tf.truncated_normal([1,word_embedding_size,1,1]),name="conv1_w")
# conv1_B = tf.Variable(tf.constant(0.1),name="conv1_b")
# conv1 = tf.nn.relu(tf.nn.conv2d(embedding_layer,conv1_W,[1,1,1,1],padding="SAME")+conv1_B)
# tf.nn.max_pool(conv1,[1,s_limit_len,1,1],[1,1,1,1])
# print("conv1",conv1)


conv1_W = get_weights([1,word_embedding_size,1,1])
conv1_bias = get_bias([1])
conv1 = tf.nn.relu(conv2d(embedding_layer,conv1_W)+conv1_bias)

conv3_W = get_weights([3,word_embedding_size,1,1])
conv3_bias = get_bias([1])
conv3 = tf.nn.relu(conv2d(embedding_layer,conv3_W)+conv3_bias)

conv5_W = get_weights([5,word_embedding_size,1,1])
conv5_b = get_bias([1])
conv5 = tf.nn.relu(conv2d(embedding_layer,conv5_W)+conv5_b)

conv7_W = get_weights([7,word_embedding_size,1,1])
conv7_B = get_bias([1])
conv7 = tf.nn.relu(conv2d(embedding_layer,conv7_W)+conv7_B)

#max_pool_layers
feature_map_1 = maxpooling(conv1, [1,s_limit_len-1+1, 1, 1], [1, 1, 1, 1])
feature_map_3 = maxpooling(conv3, [1,s_limit_len-3+1, 1, 1], [1, 1, 1, 1])
feature_map_5 = maxpooling(conv5, [1,s_limit_len-5+1, 1, 1], [1, 1, 1, 1])
feature_map_7 = maxpooling(conv7, [1,s_limit_len-7+1, 1, 1], [1, 1, 1, 1])

print("feature_map size:",feature_map_1,feature_map_3,feature_map_5,feature_map_7)
pool_outs =  tf.concat([feature_map_1,feature_map_3,feature_map_5, feature_map_7], 3)
print("pool out:",pool_outs)
pool_flat = tf.reshape(pool_outs,[-1,filter_nums])
print("pool flat:",pool_flat)
#full connect layers
h_drop = tf.nn.dropout(pool_flat,keep_prob)

full_W = tf.Variable(tf.truncated_normal([4,n_class],stddev=0.1 ,dtype=tf.float32))
full_B = tf.Variable(tf.constant(0.1,dtype=tf.float32))

outputs = tf.nn.softmax(tf.matmul(h_drop,full_W)+full_B)
pred = tf.argmax(outputs,1)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs,labels=labels))
acc = tf.reduce_mean(tf.cast(tf.equal(pred,tf.argmax(labels,1)),tf.float32))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_x, train_y, words_dict, labels_dict, all_len = data_helper.load("data/train.txt",1000,s_limit_len)
test_x,test_y, testlen =  data_helper.load_test_data("data/test_filter_2.txt",s_limit_len,words_dict,labels_dict)

def test(sess,acc,pred,tes_x,test_y):
    y_pred, acc_test = sess.run([pred,acc],feed_dict={inputs:test_x,labels:test_y,keep_prob:1.0})
    y_true = sess.run(tf.argmax(test_y,1))

    print(metrics.classification_report(y_true,y_pred))



for epoach in range(1000):
    iter = 0
    test(sess,acc,pred,test_x,test_y)
    batchs = data_helper.get_batch(64,train_x,train_y,all_len)
    for [batch_x,batch_y,batch_len] in batchs:
        _,loss_,acc_,pred_list = sess.run([train_op,loss,acc,pred],feed_dict={inputs:batch_x, labels:batch_y,keep_prob:0.5})
        if iter % 50 == 0:
            print(pred_list[:15])
            print("epoach-{0} iter-{1} loss:{2} acc-{3}".format(epoach,iter,loss_,acc_))
        # print(acc_)
        iter += 1