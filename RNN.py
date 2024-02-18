# import necessary libraries
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from collections import defaultdict

from matplotlib import pyplot as plt

from nltk.corpus import brown
from nltk.corpus import treebank
from nltk.corpus import conll2000

import seaborn as sns

from gensim.models import KeyedVectors
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Input
from keras.layers import TimeDistributed
from keras.layers import LSTM, GRU, Bidirectional, SimpleRNN, RNN
from keras.models import Model
from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

if __name__ == '__main__':

    # ### Reading Data
    df = pd.read_csv("Brown_train.txt", sep="\n", header=None)
    n_ex=len(df[0])
    for i in range(n_ex):
        df[0][i] = "^/^ " + df[0][i]
    df[0] = df[0].apply(lambda x: x.split())


    # ### Splitting Data and Labels
    X = []
    Y = []

    for i in range(n_ex):
        x = ["/".join(word.split("/")[:-1]) for word in df[0][i]]
        y = [word.split("/")[-1] for word in df[0][i]]
        X.append(x)
        Y.append(y)

    # ### 5-fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    
    
    word_tokenizer = Tokenizer()
    tag_tokenizer = Tokenizer()
    
    word_tokenizer.fit_on_texts(X)
    X_encoded = word_tokenizer.texts_to_sequences(X)
    
    tag_tokenizer.fit_on_texts(Y)
    Y_encoded = tag_tokenizer.texts_to_sequences(Y)
    lengths = [len(seq) for seq in X_encoded]
    print("Length of longest sentence: {}".format(max(lengths)))
    MAX_SEQ_LENGTH = 100
    X_padded = pad_sequences(X_encoded, maxlen=MAX_SEQ_LENGTH, padding="pre", truncating="post")
    Y_padded = pad_sequences(Y_encoded, maxlen=MAX_SEQ_LENGTH, padding="pre", truncating="post") 
    X, Y = X_padded, Y_padded
    path = './GoogleNews-vectors-negative300.bin.gz'
    word2vec = KeyedVectors.load_word2vec_format(path, binary=True)
    EMBEDDING_SIZE  = 300
    VOCABULARY_SIZE = len(word_tokenizer.word_index) + 1
    embedding_weights = np.zeros((VOCABULARY_SIZE, EMBEDDING_SIZE))
    word2id = word_tokenizer.word_index
    
	# copy vectors from word2vec model to the words present in corpus
    for word, index in word2id.items():
    	try:
    		embedding_weights[index, :] = word2vec[word]
    	except  KeyError:
    		pass  
    print("Embeddings shape: {}".format(embedding_weights.shape))
    Y = to_categorical(Y)
    print(Y.shape)
    TEST_SIZE = 0.2
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=4)
    VALID_SIZE = 0.2
    X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=VALID_SIZE, random_state=4)
    NUM_CLASSES = Y.shape[2]
    rnn_model = Sequential()
    rnn_model.add(Embedding(input_dim     =  VOCABULARY_SIZE,         # vocabulary size - number of unique words in data
                        output_dim    =  EMBEDDING_SIZE,          # length of vector with which each word is represented
                        input_length  =  MAX_SEQ_LENGTH,          # length of input sequence
                        trainable     =  True                    # False - don't update the embeddings
                        ))
    rnn_model.add(SimpleRNN(64, 
              return_sequences=True  # True - return whole sequence; False - return single output of the end of the sequence
              ))
    rnn_model.add(TimeDistributed(Dense(NUM_CLASSES, activation='softmax')))
    rnn_model.compile(loss      =  'categorical_crossentropy',
                  optimizer =  'adam',
                  metrics   =  ['acc'])
    rnn_model.summary()
    rnn_training = rnn_model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_validation, Y_validation))
    plt.plot(rnn_training.history['acc'])
    plt.plot(rnn_training.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc="lower right")
    plt.show()
