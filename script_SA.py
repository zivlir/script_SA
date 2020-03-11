from keras.layers import Activation, Dense, Dropout, Flatten, Conv1D, MaxPooling1D, LSTM
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Embedding
from keras import models
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from collections import Counter
import pandas as pd
import numpy as np
import itertools
import datetime
import pickle
import gensim
import nltk
import time
import re

seq_len = 300
wtvS = 300
wtv_wind = 7
wtv_ep = 32
wtv_min_cnt = 10
gensimwtv_model = gensim.models.Word2Vec.load('/home/ivvan/PycharmProjects/untitled/models/wtv_model_SA.w2v')
words = gensimwtv_model.wv.vocab.keys()

with open('/home/ivvan/PycharmProjects/untitled/models/kmodel_SA_tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1
POS = "positive sentiment"
NEG = "negative sentiment"
NEU = "neutral sentiment"
SENT_POLARITY = (0.4, 0.6)

emb_mtx = np.zeros((vocab_size, wtvS))

for word, i in tokenizer.word_index.items():
  if word in gensimwtv_model.wv:
    emb_mtx[i] = gensimwtv_model.wv[word]

emb_l = Embedding(vocab_size, wtvS, weights=[emb_mtx], input_length=seq_len, trainable=False)

model = Sequential()
model.add(emb_l)
model.add(Dropout(0.5))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model = load_model('/home/ivvan/PycharmProjects/untitled/models/kmodel_SA.h5')

def decode_sentiment(score, with_neu=True):
    if with_neu:
        label = NEU
        if score <= SENT_POLARITY[0]:
            label = NEG
        elif score >= SENT_POLARITY[1]:
            label = POS
        return label
    else:
        return NEG if score < 0.5 else POS

def predict(text, with_neu=True): # requires string input.
    start_at = time.time()
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=seq_len)
    score = model.predict([x_test])[0]
    label = decode_sentiment(score, with_neu=with_neu)
    score = float(score)
    return {print(f"label: {label}, score: {score}, calculation duration: {time.time()-start_at}.")}

print('Welcome!')
print('Script for post toxicity estimation is evaluated. Enter your text below for sentiment estimation and press ENTER.')
post_text = str(input('Your text: '))
print(predict(post_text))
print(f'In terminal type predict(text) and pass string to it for the next prediction. Type file_sanalysis(filepath) to make several predictions.  Local time: {datetime.datetime.now().time()}')

def file_sanalysis(filepath):
    file = open(filepath, 'rt')
    c = 0
    n = 0
    processed_file = file.readlines()
    for text in processed_file:
            n += 1
            print(n,'', text)
            print(predict(text))
            c += 1
            print(f'\n')
    print(f'Total comments analyzed: {c}')

file_sanalysis('../comment2.txt')


