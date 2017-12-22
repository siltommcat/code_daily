import tensorflow as tf

import numpy as np
import data_helper
from tensorflow.contrib.rnn import LSTMCell
from sklearn import metrics
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as birnn

voc_size = 10000
batch_size = 64
seqlen = 35
learn_rate = 0.05
n_class = 2
embedding_size = 100
class bilstm_text():
    def __init__(self,voc_size,batch_size,seq_limit_len,n_class,embedding_size,learn_rate):
        #参数设置
        self._voc_size = voc_size
        self._batch_size = batch_size
        self._seqlen_limit = seq_limit_len
        self._n_class = n_class
        self._embedding_size = embedding_size
        self._learn_rate = learn_rate
        #模型操作
        self.inputs = tf.placeholder(tf.int64,[None,seqlen],name="seq_inputs")
        self.outputs = tf.placeholder(tf.int64, [None, 2], name="outputs")
        self.seqlen_hdr = tf.placeholder(tf.int64,[None],name="sentence_len")
        self.W_embedding = tf.Variable(tf.random_uniform(shape=[self._voc_size, self._embedding_size]),name=  "word_embedding_matrix")

        self.embedding = tf.nn.embedding_lookup(self.W_embedding,self.inputs,name="word_emdding_layer")
        out_bilstm,final_state = self.bilstm()
        sentence_embedding = tf.concat([k.h for k in final_state], 1)

        f_W = tf.Variable(tf.truncated_normal([200, n_class], stddev=0.01), dtype=tf.float32)
        f_B = tf.Variable(tf.truncated_normal([n_class]), dtype=tf.float32)
        out_full = tf.matmul(sentence_embedding, f_W) + f_B

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_full, labels=self.outputs))
        # print("loss", loss)
        # print("test", tf.reduce_mean([1,2], 0))
        self.train_op = tf.train.AdagradOptimizer(self._learn_rate).minimize(self.loss)

        final_out = tf.nn.softmax(out_full)
        self.pred = tf.argmax(final_out, 1)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.pred, tf.argmax(self.outputs, 1)), tf.float32))
        #bilstm
    def bilstm(self):
        fwcell = LSTMCell(self._embedding_size)
        bwcell = LSTMCell(self._embedding_size)
        out_bilstm, final_state = birnn(fwcell, bwcell, inputs=self.embedding, sequence_length=self.seqlen_hdr, dtype=tf.float32)
        return out_bilstm,final_state
def test(model,test_x, test_y, seqlen_test):

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    op_acc = model.acc
    op_pred = model.pred

    test_pred, acc_test = sess.run([op_pred, op_acc], feed_dict={model.inputs:test_x,model.outputs:test_y,model.seqlen_hdr:seqlen_test})
    test_y = sess.run(tf.argmax(test_y,1))
    #run之后得到的不是tensor 是narray
    #这个函数必须要是list
    print("test-->all acc:",acc_test)
    print(metrics.classification_report(test_y, test_pred))
    # print(metrics.classification_report(np.array(test_y).tolist(), np.array(test_pred).tolist()))
    # batchs = data_helper.get_batch(test_x,test_y,seqlen_test)
    # for batch_x,batch_y,batch_len in batchs:
    #     test_pred,acc_test = sess.run(op_pred,op_acc,feed_dict={model.inputs:batch_x,model.outputs:batch_y,model.seqlen_hdr:batch_len})

def train():
    train_x, train_y, words_dict, labels_dict, seqlen_all = data_helper.load("train.txt", 10000, 35)
    test_x, test_y, seqlen_test = data_helper.load_test_data("test_filter_2.txt", seqlen, words_dict, labels_dict)
    model = bilstm_text(voc_size,batch_size,seqlen,n_class,embedding_size,learn_rate)
    op_pred = model.pred
    op_loss = model.loss
    op_train = model.train_op
    op_acc = model.acc
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    epoachs = 50
    cnt = 0

    for epoach in range(epoachs):
        batchs = data_helper.get_batch(64, train_x, train_y, seqlen_all)
        for batch_x,batch_y, batch_len in batchs:
            [_,train_acc] = sess.run([op_train,op_acc],feed_dict={model.inputs:batch_x,model.outputs:batch_y,model.seqlen_hdr:batch_len})
            print("{0} epoach {1} iters acc = {2}".format(epoach,cnt,train_acc))
            if cnt % 50 == 0:
                tmp_pred = sess.run(op_pred,feed_dict={model.inputs:batch_x,model.outputs:batch_y,model.seqlen_hdr:batch_len})
                print(tmp_pred)
                test(model, test_x, test_y, seqlen_test)
            cnt += 1
        print("---------test----------------")
        test(model,test_x, test_y, seqlen_test)
if __name__ == "__main__":

    train()

    # train_x,train_y,words_dict,labels_dict,seqlen_all = data_helper.load("train.txt",10000,35)
    # test_x,test_y,seqlen_test = data_helper.load_test_data("test_filter_2.txt",seqlen,words_dict,labels_dict)
    # seqlen_all = np.array(seqlen_all)*10
    # inputs = tf.placeholder(tf.int64,[None,seqlen],name="seq_inputs")
    # outputs = tf.placeholder(tf.int64,[None,2],name= "outputs")
    # seqlen_hdr = tf.placeholder(tf.int64,[None])
    # W_embedding = tf.Variable(tf.random_uniform(shape=[voc_size, embedding_size]))
    # embedding = tf.nn.embedding_lookup(W_embedding,inputs)
    # print("embding",embedding)
    # #embedding shape(35,100)
    # fwcell = LSTMCell(embedding_size)
    # bwcell = LSTMCell(embedding_size)
    # #seqlen这里应该是一个batchsize的长度，应该是一个tensor
    # out_bilstm, final_state = birnn(fwcell,bwcell,inputs = embedding,sequence_length=seqlen_hdr,dtype=tf.float32)
    # #attention取得是output做的特征，不是state-----个人理解，英文实在捉鸡
    # # out_bilstm = tf.concat(out_bilstm,axis=2)
    # #这里使用最后的状态作为特征，需要注意的是lstm单元最后与gru单元不同，最后包含的是单元状态m和因层输出h两个变量
    # sentence_embedding = tf.concat([k.h for k in final_state],1)
    #
    # # W1 = tf.Variable(tf.truncated_normal([out_bilstm.get_shape()[1].value, 128], stddev=0.1))
    # # b1 = tf.Variable(tf.constant(0., shape=[128]))
    # # hidden = tf.nn.tanh(tf.matmul(out_bilstm, W1) + b1)
    # f_W = tf.Variable(tf.truncated_normal([200,n_class],stddev=0.01),dtype=tf.float32)
    # f_B = tf.Variable(tf.truncated_normal([n_class]),dtype=tf.float32)
    # out_full = tf.matmul(sentence_embedding,f_W) + f_B
    #
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_full,labels=outputs))
    # # print("loss", loss)
    # # print("test", tf.reduce_mean([1,2], 0))
    # train_op = tf.train.AdagradOptimizer(learn_rate).minimize(loss)
    #
    # final_out = tf.nn.softmax(out_full)
    # pred = tf.argmax(final_out,1)
    # acc = tf.reduce_mean(tf.cast(tf.equal(pred,tf.argmax(outputs,1)),tf.float32))
    # print("out_bi",out_bilstm)
    # print("state:",sentence_embedding)
    # print("fw:",fwcell)


    # sess = tf.Session()
    # init = tf.global_variables_initializer()
    # sess.run(init)
    # def test(sess,acc,test_x,test_y,test_len):
    #     acc_test = sess.run(acc,feed_dict={inputs:test_x,outputs:test_y,seqlen_hdr:test_len})
    #     print(acc_test)
    # for epoach in range(200):
    #     batchs = data_helper.get_batch(64,train_x,train_y,seqlen_all)
    #     cnt = 0
    #     for batch_x,batch_y, batch_len in batchs:
    #
    #         # batch_label_oh = sess.run(tf.one_hot(batch_y,n_class))
    #         _,train_loss,train_acc = sess.run([train_op,loss,acc],feed_dict={inputs:batch_x,outputs:batch_y,seqlen_hdr:batch_len})
    #         print("train_size",cnt,"train_acc:",train_acc)
    #         print("loss:",train_loss)
    #         # print(len(train_x))
    #         cnt+=256
    #         # print(sess.run(pred,feed_dict={inputs:batch_x,outputs:batch_y,seqlen_hdr:batch_len}))
    #     print("-----test-----",epoach)
    #     test(sess,acc,test_x,test_y,seqlen_test)
            # print(acc)







    #其实这里我有不明白的地方就是这个seqlen_hdr到底应该是赋予句子中单词的个数还是应该赋予最后整个句子的维度
    # sess.run(out_full,feed_dict={inputs:train_x,seqlen_hdr:seqlen_all})

    #为什么shape tensor是不对的返回的是（3,）
    # a = sess.run(embedding,feed_dict={inputs:train_x})
    # print(a)
    # print(np.array(sess.run(embedding,feed_dict={inputs:train_x})).shape)
