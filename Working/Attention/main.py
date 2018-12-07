import pandas as pd
import numpy as np
import tensorflow as tf
import re, time
from string import punctuation
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from collections import namedtuple
from attention import attention
from utils import get_data
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
import pickle


maxSentenceLength = 20
maxWordLength = 50



with open("/home/u19159/Project/Attention/embed.pkl",'rb') as f:
    embedding = pickle.load(f)


def build_rnn(n_words, embed_size, batch_size, lstm_size, num_layers, 
	          dropout, learning_rate, multiple_fc, fc_units):
	tf.reset_default_graph()
	with tf.name_scope('inputs'):
	    inputs = tf.placeholder(tf.int32, [None,None, None], name='inputs')

	with tf.name_scope('labels'):
	    labels = tf.placeholder(tf.int32, [None,1,13], name='labels')

	keep_prob = tf.placeholder(tf.float32, name='keep_prob')

	with tf.name_scope("embeddings"):
	    embed = tf.nn.embedding_lookup(embedding, inputs)

	print("After Embedding",embed.shape)

	with tf.name_scope("RNN_layers"):
		layers = [tf.nn.rnn_cell.LSTMCell(lstm_size,state_is_tuple=True) for _ in range(num_layers)]
		cell = tf.nn.rnn_cell.MultiRNNCell(layers)

	with tf.name_scope("RNN_init_state"):
	    initial_state = cell.zero_state(batch_size, tf.float32)

	embed = tf.reshape(embed,[maxSentenceLength * batch_size,maxWordLength,embed_size])
	print(embed.shape)

	with tf.name_scope("RNN_forward"):
	    outputs, final_state = bi_rnn(cell, cell, embed,dtype=tf.float32)   

	print("RNN OUTPUT SHAPE", outputs[0].shape)

	with tf.name_scope('Attention_layer'):
	    outputs, alphas = attention(outputs, ATTENTION_SIZE, return_alphas=True)
	    tf.summary.histogram('alphas', alphas)
	print("After Attention",outputs.shape)

	outputs = tf.reshape(outputs,[batch_size,-1])
	print(outputs.shape)
	
	with tf.name_scope("fully_connected"):
	    weights = tf.truncated_normal_initializer(stddev=0.1)
	    biases = tf.zeros_initializer()
	    
	    dense = tf.contrib.layers.fully_connected(outputs,
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
	    print("Predictions shape",predictions.shape)
	    tf.summary.histogram('predictions', predictions)

	with tf.name_scope('cost'):
	    cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,logits=predictions)
	    cost = tf.reduce_mean(tf.cast(cost, tf.float32))
	    tf.summary.scalar('cost', cost)

	with tf.name_scope('train'):    
	    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

	with tf.name_scope("accuracy"):
	    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels)
	    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	    tf.summary.scalar('accuracy', accuracy)

	merged = tf.summary.merge_all()    

	export_nodes = ['inputs', 'labels', 'keep_prob', 'initial_state', 'final_state','accuracy',
	                'predictions', 'cost', 'optimizer', 'merged','alphas']
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
		iteration2=0

		train_writer = tf.summary.FileWriter('/home/u19159/Project/Attention/AttLogs/train/{}'.format(log_string), sess.graph)
		valid_writer = tf.summary.FileWriter('/home/u19159/Project/Attention/AttLogs/valid/{}'.format(log_string))

		for e in range(epochs):
			state = sess.run(model.initial_state)

			train_loss = []
			train_acc = []
			val_acc = []
			val_loss = []
			predictions = []
			ans=[]

			for _, (x, y) in enumerate(get_batches(x_train, y_train, batch_size), 1):
			    feed = {model.inputs: x,
			            model.labels: y[:,None],
			            model.keep_prob: dropout,
			            model.initial_state: state}
			    
			    summary, loss, acc, prediction, _ = sess.run([model.merged, 
			                                             model.cost, 
			                                             model.accuracy, 
			                                             model.predictions,
			                                             model.optimizer], 
			                                            feed_dict=feed)                
			    
			    
			    train_loss.append(loss)
			    train_acc.append(acc)
			    train_writer.add_summary(summary, iteration)
			    
			    iteration += 1

			avg_train_loss = np.mean(train_loss)
			avg_train_acc = np.mean(train_acc)

			print("")

			valid_loss = []
			valid_acc = []

			sentence_alpahs = []

			val_state = sess.run(model.initial_state)
			for _, (x, y) in enumerate(get_batches(x_valid, y_valid, batch_size), 1):
				feed = {model.inputs: x,
				        model.labels: y[:, None],
				        model.keep_prob: 1}

				summary, batch_loss, batch_acc, ALPHAS = sess.run([model.merged, 
				                                                      model.cost, 
				                                                      model.accuracy,model.alphas], 
				                                                     feed_dict=feed)

				iteration2 +=1
				valid_loss.append(batch_loss)
				valid_acc.append(batch_acc)
				sentence_alpahs.append(ALPHAS)
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








x_data,y_data,vocab = get_data()
ts = len(x_data)-200
x_train = x_data[:ts]
y_train =  y_data[:ts]

x_valid = x_data[ts:]
y_valid = y_data[ts:]

n_words = embedding.shape[0]
embed_size = 300
batch_size = 128
lstm_size = 128
num_layers = 2

num_batches = x_train.shape[0] // batch_size
dropout = 0.5
learning_rate = 0.0001
epochs = 30
multiple_fc = False
fc_units = 256


HIDDEN_SIZE = 128
ATTENTION_SIZE = 50

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


