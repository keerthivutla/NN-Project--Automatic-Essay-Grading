    

import pandas as pd
import numpy as np
import tensorflow as tf
import nltk, re, time
from nltk.corpus import stopwords
from string import punctuation
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from collections import namedtuple

from utils import get_data


import pickle

with open("/home/u19159/Project/Attention/embed.pkl",'rb') as f:
    embedding = pickle.load(f)


def build_rnn(n_words, embed_size, batch_size, lstm_size, num_layers, 
              dropout, learning_rate, multiple_fc, fc_units):

    tf.reset_default_graph()

    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.int32, [None, None], name='inputs')

    with tf.name_scope('labels'):
        labels = tf.placeholder(tf.int32, [None,1,13], name='labels')

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    with tf.name_scope("embeddings"):
        embed = tf.nn.embedding_lookup(embedding, inputs)

    print(embed.shape)
    with tf.name_scope("RNN_layers"):
    	layers = [tf.nn.rnn_cell.BasicLSTMCell(lstm_size) for _ in range(num_layers)]
    	cell = tf.nn.rnn_cell.MultiRNNCell(layers)


    with tf.name_scope("RNN_init_state"):
        initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.name_scope("RNN_forward"):
        outputs, final_state = tf.nn.dynamic_rnn(cell, embed,
                                                 initial_state=initial_state)    

    with tf.name_scope("fully_connected"):
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()
        
        dense = tf.contrib.layers.fully_connected(outputs[:, -1],
                                                  num_outputs = fc_units,
                                                  activation_fn = tf.sigmoid,
                                                  weights_initializer = weights,
                                                  biases_initializer = biases)
        dense = tf.contrib.layers.dropout(dense, keep_prob)
        
        if multiple_fc == True:
            dense = tf.contrib.layers.fully_connected(dense,
                                                      num_outputs = fc_units,
                                                      activation_fn = tf.sigmoid,
                                                      weights_initializer = weights,
                                                      biases_initializer = biases)
            dense = tf.contrib.layers.dropout(dense, keep_prob)
    
    with tf.name_scope('predictions'):
        predictions = tf.contrib.layers.fully_connected(dense, 
                                                        num_outputs = 13, 
                                                        activation_fn=tf.sigmoid,
                                                        weights_initializer = weights,
                                                        biases_initializer = biases)
        print(predictions.shape)
        tf.summary.histogram('predictions', predictions)
    
    with tf.name_scope('cost'):
        cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,logits=predictions)


    with tf.name_scope('train'):    
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
    
    merged = tf.summary.merge_all()    

    export_nodes = ['inputs', 'labels', 'keep_prob', 'initial_state', 'final_state','accuracy',
                    'predictions', 'cost', 'optimizer', 'merged']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])
    
    return graph



def get_batches(x, y, batch_size):
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]


def train(model, epochs, log_string):
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        iteration = 0
        iteration2 = 0

        train_writer = tf.summary.FileWriter('/home/u19159/Project/LSTM/lstmLogs/train/{}'.format(log_string), sess.graph)
        valid_writer = tf.summary.FileWriter('/home/u19159/Project/LSTM/lstmLogs/valid/{}'.format(log_string))

        for e in range(epochs):
            state = sess.run(model.initial_state)
            
            train_loss = []
            train_acc = []
            valid_acc = []
            valid_loss = []
            predictions = []
            for _, (x, y) in enumerate(get_batches(x_train, y_train, batch_size), 1):
                feed = {model.inputs: x,
                        model.labels: y[:,None],
                        model.keep_prob: dropout,
                        model.initial_state: state}
                
                summary, loss, acc, state, prediction, _ = sess.run([model.merged, 
                                                         model.cost, 
                                                         model.accuracy, 
                                                         model.final_state, model.predictions,
                                                         model.optimizer], 
                                                        feed_dict=feed)                
                
                train_loss.append(loss)
                train_acc.append(acc)
                predictions.append(prediction)
                
                train_writer.add_summary(summary, iteration)
                
                iteration += 1
            
            avg_train_loss = np.mean(train_loss)
            avg_train_acc = np.mean(train_acc) 

            val_state = sess.run(model.initial_state)
            for x, y in get_batches(x_valid, y_valid, batch_size):
                feed = {model.inputs: x,
                        model.labels: y[:, None],
                        model.keep_prob: 1,
                        model.initial_state: val_state}
                summary, batch_loss, batch_acc, val_state = sess.run([model.merged, 
                                                                      model.cost, 
                                                                      model.accuracy, 
                                                                      model.final_state], 
                                                                     feed_dict=feed)

                iteration2 +=1
                valid_loss.append(batch_loss)
                valid_acc.append(batch_acc)
                valid_writer.add_summary(summary, iteration2)
                        
            print("Epoch: {}/{}".format(e, epochs),
                  "Train Loss: {:.3f}".format(avg_train_loss),
                  "Train Acc: {:.3f}".format(avg_train_acc))

            avg_valid_loss = np.mean(valid_loss)
            avg_valid_acc = np.mean(valid_acc)

            print("Epoch: {}/{}".format(e, epochs),
              "Valid Loss: {:.3f}".format(avg_valid_loss),
              "Valid Acc: {:.3f}".format(avg_valid_acc))

        train_writer.close()
        valid_writer.close()






x_data,y_data,embedding = get_data()

ts = len(x_data) - 200


x_train = x_data[:ts]
y_train =  y_data[:ts]


x_valid = x_data[ts:]
y_valid = y_data[ts:]

n_words = embedding.shape[0]
embed_size = 300
batch_size = 128
lstm_size = 128
num_layers = 2
dropout = 0.5
learning_rate = 0.0001
epochs = 30
multiple_fc = False
fc_units = 256

log_string = 'ru={},fcl={},fcu={}'.format(lstm_size,
                                                  multiple_fc,
                                                  fc_units)
model = build_rnn(n_words = n_words, 
                          embed_size = embed_size,
                          batch_size = batch_size,
                          lstm_size = lstm_size,
                          num_layers = num_layers,
                          dropout = dropout,
                          learning_rate = learning_rate,
                          multiple_fc = multiple_fc,
                          fc_units = fc_units)            

train(model, epochs, log_string)

