import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data










tf.set_random_seed(1)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
learn_rate = 0.01
training_iters = 100000
batch_size = 128
n_input = 28
n_step = 28
n_hiden_units = 128
n_class = 10

x = tf.placeholder(tf.float32,[None,n_step,n_input],name="x_int")#????why not -->[n_input,n_hidens_units],name="x_in")
y = tf.placeholder(tf.float32,[None,n_class],name="y_out")

weights = {
    'in':tf.Variable(tf.truncated_normal([n_input,n_hiden_units]),dtype=tf.float32),
    'out':tf.Variable(tf.truncated_normal([n_hiden_units,n_class]),dtype=tf.float32)
}

bias = {
    'in_b':tf.Variable(tf.truncated_normal([n_hiden_units]),dtype=tf.float32),
    'out_b':tf.Variable(tf.truncated_normal([n_class]),dtype=tf.float32),
}

def RNN(X,weights,biases):
    print("yes")
    X = tf.reshape(X,[-1,n_input])
    X_in = tf.matmul(X,weights['in'])+biases['in_b']
    X_in = tf.reshape(X_in,[-1,n_step,n_hiden_units])
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hiden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
    results = tf.matmul(final_state[1],weights['out'])+bias['out_b']
    return results

pred = RNN(x,weights,bias)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdagradOptimizer(learn_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    for i in range(3000):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_step, n_input])
        sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys})
        if i % 50 == 0:
            print(sess.run(accuracy, feed_dict={
                x: batch_xs,
                y: batch_ys,}))