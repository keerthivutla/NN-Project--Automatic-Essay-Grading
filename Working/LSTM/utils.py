import re
import os as os
import numpy as np
import itertools
import spacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

import pickle

from sklearn.preprocessing import OneHotEncoder
def load_training_data(training_path, essay_set=1):
    training_df = pd.read_csv(training_path, delimiter='\t')
    resolved_score = training_df[training_df['essay_set'] == essay_set]['domain1_score']
    essay_ids = training_df[training_df['essay_set'] == essay_set]['essay_id']
    essays = training_df[training_df['essay_set'] == essay_set]['essay']
    essay_list = []
    for idx, essay in essays.iteritems():
        essay = clean_str(essay)
        essay_list.append(tokenize(essay))
        # print(tokenize(essay))
    return essay_list, resolved_score.tolist(), essay_ids.tolist()
    

def tokenize(sent):
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    # string = re.sub(r"\s{2,}", " ", string)
    
    return string.strip().lower()

def build_vocab(sentences, vocab_limit):
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    vocabulary = {x: i+1 for i,x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def vectorize_data(data, word_idx, sentence_size):
    E = []
    for essay in data:
        ls = max(0, sentence_size - len(essay))
        wl = []
        for w in essay:
            if w in word_idx:
                wl.append(word_idx[w])
            else:
                wl.append(0)
        wl += [0]*ls
        E.append(wl)
    return E



def sbd_component(doc):
    for i, token in enumerate(doc[:-2]):
        if token.text == '.' and doc[i+1].is_title:
            doc[i+1].sent_start = True
    return doc


def create_embeddings(a):
    embedding = [nlp(key).vector for key,value in a.items()]
    return embedding


def get_data():
    train_data,train_labels,ids = load_training_data('/home/u19159/Project/LSTM/training_set_rel3.tsv')
    a,b = build_vocab(train_data,20000)
    a['0'] = 'UNK'
    max_words_length = 1000
    final_data = np.zeros((len(train_data),max_words_length))
    train_labels = np.array(train_labels)
    train_labels = train_labels.reshape((len(train_data),1))

    for i in range(len(train_data)):
        for j,_ in enumerate(train_data[i]):
            if(j < 1000):
                if(a[ train_data[i][j] ] > len(a)):
                    print(train_data[i][j])
                if(a[train_data[i][j] ] > 0):
                    final_data[i][j] = a[train_data[i][j]]
                else:
                    print(i)

    embedding = np.array(create_embeddings(a))

    with open('/home/u19159/Project/LSTM/LstmEmbed.pkl', 'wb') as embed:
        pickle.dump(embedding,embed)


    train_labels = np.array(train_labels)
    oneHot = np.zeros((len(train_data),13),dtype=np.float32)
    for idx,val in enumerate(train_labels):
        oneHot[idx][val] = 1.0

    return final_data,oneHot,embedding



nlp = spacy.load('en_core_web_md')
nlp.add_pipe(sbd_component, before='parser')
