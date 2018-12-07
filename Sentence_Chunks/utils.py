import re
import os as os
import numpy as np
import itertools
import spacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from sklearn.preprocessing import OneHotEncoder

def zero_pad(X, seq_len):
    return np.array([x[:seq_len] + [0] * max(seq_len - len(x), 0) for x in X])

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
    return string.strip().lower()

def load_training_data(training_path, essay_set=1):
    training_df = pd.read_csv(training_path, delimiter='\t')
    resolved_score = training_df[training_df['essay_set'] == essay_set]['domain1_score']
    essay_ids = training_df[training_df['essay_set'] == essay_set]['essay_id']
    essays = training_df[training_df['essay_set'] == essay_set]['essay']
    essay_list = []
    essay_words = []
    for idx, essay in essays.iteritems():
        essay = clean_str(essay)
        essay_list.append(essay)
        essay_words.append(tokenize(essay))
    return essay_list, resolved_score.tolist(), essay_ids.tolist(),essay_words
    


def get_data():
    #td = 20
    train_data = global_data
    train_labels = global_labels
    max_sen_length = 20
    max_words_length = 50
    final_data = np.zeros((len(train_data),max_sen_length,max_words_length))
    train_labels = np.array(train_labels)
    train_labels = train_labels.reshape((len(train_data),1))
    essay_words = []

    essay_sentences = []
    for i in range(len(final_data)):
    	text = []
    	doc = nlp(train_data[i])
    	for sentence in doc.sents:  
    		text.append(tokenize(sentence.text))
    		essay_words.append(tokenize(sentence.text))

    	essay_sentences.append(text)


    vocab,vocab_inv = build_vocab(essay_words,40000)

    for i in range(len(final_data)):
        text = []
        doc = nlp(train_data[i])
        for sentence in doc.sents:  
            text.append(tokenize(sentence.text))
        
        for j in range(min(max_sen_length,len(text))):
            words_list = text[j]
            
            for w in range(min(max_words_length,len(words_list))):
                temp = words_list[w]

                if(vocab[temp] > len(vocab)):
                    print(temp)
                if(vocab[temp] > 0):
                    final_data[i][j][w] = vocab[temp]
                else:
                    print(i)

    train_labels = np.array(train_labels)
    oneHot = np.zeros((len(train_data),13),dtype=np.float32)
    for idx,val in enumerate(train_labels):
        oneHot[idx][val] = 1.0

    return final_data,oneHot,vocab,essay_sentences



nlp = spacy.load('en_core_web_md')
nlp.add_pipe(sbd_component, before='parser')

global_data,global_labels,ids,essay_words = load_training_data('/home/u19159/Project/Attention/training_set_rel3.tsv')

