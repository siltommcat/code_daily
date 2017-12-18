import tensorflow as tf
import numpy as np
import data_helper
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as birnn
voc_size = 10000
batch_size = 64
seqlen = 35
learn_rate = 0.05
n_class = 2
embedding_size = 100
train_x,train_y,words_dict,labels_dict,seqlen_all = data_helper.load("train.txt",10000,35)
one_hot_label = tf.one_hot(train_y,n_class)
test_x,test_y,seqlen_test = data_helper.load_test_data("test_filter_2.txt",seqlen,words_dict,labels_dict)
# seqlen_all = np.array(seqlen_all)*10
inputs = tf.placeholder(tf.int64,[None,seqlen],name="seq_inputs")
outputs = tf.placeholder(tf.int64,[None,2],name= "outputs")
seqlen_hdr = tf.placeholder(tf.int64,[None])
W_embedding = tf.Variable(tf.random_uniform(shape=[voc_size, embedding_size]))
embedding = tf.nn.embedding_lookup(W_embedding,inputs)
print("embding",embedding)
#embedding shape(35,100)
fwcell = LSTMCell(embedding_size)
bwcell = LSTMCell(embedding_size)
#seqlen这里应该是一个batchsize的长度，应该是一个tensor
out_bilstm, final_state = birnn(fwcell,bwcell,inputs = embedding,sequence_length=seqlen_hdr,dtype=tf.float32)
#attention取得是output做的特征，不是state-----个人理解，英文实在捉鸡
# out_bilstm = tf.concat(out_bilstm,axis=2)
#这里使用最后的状态作为特征，需要注意的是lstm单元最后与gru单元不同，最后包含的是单元状态m和因层输出h两个变量
sentence_embedding = tf.concat([k.h for k in final_state],1)

# W1 = tf.Variable(tf.truncated_normal([out_bilstm.get_shape()[1].value, 128], stddev=0.1))
# b1 = tf.Variable(tf.constant(0., shape=[128]))
# hidden = tf.nn.tanh(tf.matmul(out_bilstm, W1) + b1)
f_W = tf.Variable(tf.truncated_normal([200,n_class],stddev=0.01),dtype=tf.float32)
f_B = tf.Variable(tf.truncated_normal([n_class]),dtype=tf.float32)
out_full = tf.matmul(sentence_embedding,f_W) + f_B

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_full,labels=outputs))
# print("loss", loss)
# print("test", tf.reduce_mean([1,2], 0))
train_op = tf.train.AdagradOptimizer(learn_rate).minimize(loss)

final_out = tf.nn.softmax(out_full)
pred = tf.argmax(final_out,1)
acc = tf.reduce_mean(tf.cast(tf.equal(pred,tf.argmax(outputs,1)),tf.float32))
print("out_bi",out_bilstm)
print("state:",sentence_embedding)
print("fw:",fwcell)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
def test(sess,acc,test_x,test_y,test_len):
    acc_test = sess.run(acc,feed_dict={inputs:test_x,outputs:test_y,seqlen_hdr:test_len})
    print(acc_test)
for epoach in range(200):
    batchs = data_helper.get_batch(64,train_x,train_y,seqlen_all)
    cnt = 0
    for batch_x,batch_y, batch_len in batchs:

        # batch_label_oh = sess.run(tf.one_hot(batch_y,n_class))
        _,train_loss,train_acc = sess.run([train_op,loss,acc],feed_dict={inputs:batch_x,outputs:batch_y,seqlen_hdr:batch_len})
        print("train_size",cnt,"train_acc:",train_acc)
        print("loss:",train_loss)
        # print(len(train_x))
        cnt+=256
        # print(sess.run(pred,feed_dict={inputs:batch_x,outputs:batch_y,seqlen_hdr:batch_len}))
    print("-----test-----",epoach)
    test(sess,acc,test_x,test_y,seqlen_test)
        # print(acc)
# for epoach in range(10):

#其实这里我有不明白的地方就是这个seqlen_hdr到底应该是赋予句子中单词的个数还是应该赋予最后整个句子的维度
# sess.run(out_full,feed_dict={inputs:train_x,seqlen_hdr:seqlen_all})

#为什么shape tensor是不对的返回的是（3,）
# a = sess.run(embedding,feed_dict={inputs:train_x})
# print(a)
# print(np.array(sess.run(embedding,feed_dict={inputs:train_x})).shape)
