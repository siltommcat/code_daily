import tensorflow  as tf
import numpy as np
import pandas as pd
from sklearn import  metrics
import data_helper
class text_cnn():
    def __init__(self, n_class, learning_rate, s_limit_len, voc_size, embedding_size, filters, filter_nums,word2vec=None):
        self.s_limit_len = s_limit_len
        self.n_class = n_class
        self.inputs = tf.placeholder(tf.int64,[None,self.s_limit_len],name = "inputs")
        self.y = tf.placeholder(tf.int64,[None,self.n_class])
        self.keep_prob = tf.placeholder(tf.float32)
        if word2vec is None:
            embedding_w = tf.Variable(tf.truncated_normal([voc_size, embedding_size], stddev= 0.1))
        else:
            embedding_w = word2vec

        embedding_layer = tf.nn.embedding_lookup(embedding_w, self.inputs)
        #conv2d 要求图片是三维的x,y,z(chanel)所以要加一个维度，实际大小不变，也就是说tf.reshape也是可以的操作，但是如果是reshape的话要注意大小
        embedding_layer = tf.expand_dims(embedding_layer,-1)

        #conv and pooling
        pool_arr = []
        filter_sum = 0
        for filter_size,filter_nums in zip(filters,filter_nums):
            conv_w = tf.Variable(tf.truncated_normal([filter_size, embedding_size,1,filter_nums],stddev=0.1))
            conv_b = tf.Variable(tf.constant(0.1))
            conv = tf.nn.conv2d(embedding_layer,conv_w,[1,1,1,1],padding="VALID",name = "conv")
            # conv = tf.nn.conv2d(embedding_layer,conv_w,[1,1,1,1],padding="SAME",name = "conv")
            #　这里不能是same格式，不然这个地方会出现sentence_len*embeddingsize的长度
            conv_out = tf.nn.relu(conv+conv_b)
            print("conv_out:",conv_out)
            pool = tf.nn.max_pool(conv_out,[1,s_limit_len-filter_size+1,1,1],strides=[1,1,1,1],padding="VALID",name = "POOL")
            print("max_pool",pool)
            pool_arr.append(pool)
            filter_sum += filter_nums

        pool_out = tf.concat(pool_arr,3)
        print("reshape_before",pool_out)
        pool_flat = tf.reshape(pool_out,[-1,filter_sum],name = "pool_flat")
        print("reshape_after",pool_out)


        #full connect layer
        fc_w1 = tf.Variable(tf.truncated_normal([filter_sum,2*filter_sum],stddev=0.1),name="fc_w1")
        fc_b1 = tf.Variable(tf.constant(0.1),name="fc_b1")
        fc_output1 = tf.nn.dropout(tf.nn.relu(tf.matmul(pool_flat,fc_w1) + fc_b1),self.keep_prob)

        fc_w2 = tf.Variable(tf.truncated_normal([filter_sum*2,n_class],stddev=0.1),name = "fc_w2")
        fc_b2 = tf.Variable(tf.constant(0.1), name="fc_b2")
        self.fc_output2 = tf.nn.softmax(tf.nn.dropout(tf.matmul(fc_output1,fc_w2) + fc_b2,self.keep_prob))
        print(self.fc_output2)
        self.pred = tf.argmax(self.fc_output2,1)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.fc_output2,labels = self.y))
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.pred,tf.argmax(self.y,1)),tf.float32))
def test(sess,model,test_x,test_y):
    y_pred,test_acc = sess.run([model.pred,model.acc],feed_dict={model.inputs:test_x,model.y:test_y,model.keep_prob:1.0})
    print("------------test acc:{0}--------------".format(test_acc))
    y_true = sess.run(tf.argmax(test_y,1))
    print(metrics.classification_report(y_true,y_pred))
def train(model,epoachs,train_x,train_y,all_len,test_x,test_y):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    #
    for epoach in range(epoachs):
        iter = 0
        test(sess,model,test_x,test_y)
        batchs = data_helper.get_batch(64, train_x, train_y, all_len)
        for batch_x,batch_y, batch_len in batchs:
            sess.run(model.train_op,feed_dict={model.inputs:batch_x,model.y:batch_y,model.keep_prob:0.8})
            y_pred,train_acc, train_loss = sess.run([model.pred,model.acc,model.loss],feed_dict={model.inputs:batch_x,model.y:batch_y,model.keep_prob:1.0})
            if iter % 100 == 0:
                print("{0} epoach {1} iters run train acc: {2} train loss:{3}".format(epoach,iter,train_acc,train_loss))
                if iter % 200 == 0:
                    print("pred value:",y_pred)
            iter += 1


if __name__ == "__main__":
    s_limit_len = 10
    cnn = text_cnn(2,0.005,10,5000,128,[1,3,5,7],[1,2,3,3])
    train_x, train_y, words_dict, labels_dict, all_len = data_helper.load("data/train.txt", 1000, s_limit_len)
    test_x, test_y, testlen = data_helper.load_test_data("data/test_filter_2.txt", s_limit_len, words_dict, labels_dict)
    # batchs = data_helper.get_batch(64,train_x,train_y,all_len)
    train(cnn,200,train_x,train_y,all_len,test_x,test_y)

    pass