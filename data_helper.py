import os
import collections
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
def padding(vec,seqlen):
    return vec[:seqlen]+[0]*max(seqlen-len(vec),0)

#transform 2 dl formation
#label split_words
def int2one_hot(train):
    enc = OneHotEncoder()
    labelset = set(train)
    res = []
    for i in labelset:
        res.append([i])
    enc.fit(res)
    train = np.array(train)
    train = train.reshape([len(train),1])
    res= enc.transform(train).toarray()
    return res
def load(filename,voc_size,seqlen):
    words_dic = {}
    words_set = []
    res_split = []
    all_len = []

    print(os.getcwd())
    with open(filename,"r") as f:
        for line in f.readlines():
            label,words = line.strip().split("\t")
            words = words.split(" ")
            while ("" in words):
                words.remove("")
            words_set.extend(words)
            res_split.append([label,words])

    counter = collections.Counter(words_set)
    tmp_dic = counter.most_common(voc_size-2)
    words_set = [a for a,b in tmp_dic]
    words_dic = dict(zip(words_set,range(2,voc_size)))
    words_dic[" "] = 0
    words_dic["UNK"] = 1
    train_x = []
    train_y = []
    labels = [k for k,h in res_split]
    labels_dict = dict(zip(set(labels),range(len(set(labels)))))
    train_y = [labels_dict[k] for k in labels]
    all_len = [len(k) for j,k in res_split]
    # print(all_len)
    for label,data in res_split:
        # train_x = [words_dic[k] for k in data]
        cur = []
        for i in data:
            if i in words_set:
               cur.append(words_dic[i])
            else:
                cur.append(words_dic["UNK"])
        train_x.append(padding(cur,seqlen))
    train_y = int2one_hot(train_y)
    return train_x,train_y,words_dic,labels_dict,all_len

def load_test_data(filename,limit_len,words_dict,label_dict):
    res_x = []
    res_y = []
    res_len = []
    with open(filename, "r") as f:
        for line in f.readlines():
            label, words = line.strip().split("\t")
            words = words.split(" ")
            while ("" in words):
                words.remove("")
            cur = []
            for k in words:
                if k in words_dict.keys():
                    cur.append(words_dict[k])
                else:
                    cur.append(words_dict["UNK"])
            res_x.append(padding(cur,limit_len))
            res_y.append(label_dict[label])
            res_len.append(len(cur))
    res_y = int2one_hot(res_y)
    return res_x,res_y,res_len


def get_batch(batch_size, data_x, data_y, data_len):
    i = 0
    while i+batch_size < len(data_len):
        # print(i)
        yield data_x[i:i+batch_size],data_y[i:i+batch_size] , data_len[i:i+batch_size]
        i +=  batch_size
    yield data_x[i:], data_y[i:], data_len[i:]

            # return None,None,None
if __name__ == "__main__":
    train_x,train_y,words_dict,labels_dict, all_len = load("test_text.txt",10000,35)
    # test_x,test_y,test_len = load_test_data("test_filter_2.txt",35,words_dict,labels_dict)
    # a= get_batch(5,train_x,train_y,all_len)

    res = int2one_hot(train_y)
    print(res)


