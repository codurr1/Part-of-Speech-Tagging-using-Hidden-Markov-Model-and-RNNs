#!/usr/bin/env python
# coding: utf-8

# # HMM and Viterbi Decoding - Yash Malik (2001CS79)

# ### Importing Dependencies

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from collections import defaultdict

# ### HMM Training with Bigram Assumption

initial_state_probs = defaultdict(float)
transition_probs = defaultdict(float)
emmission_probs = defaultdict(float)
new_emmission_probs = defaultdict(float)
tags = ['NOUN', 'VERB', '.', 'ADP', 'DET', 'ADJ', 'ADV', 'PRON', 'CONJ', 'PRT', 'NUM', 'X', '^']
n_states = len(tags)

import math

def Training(X_train, y_train, transition_type):
    bigram_transition_counts = defaultdict(int)
    trigram_transition_counts = defaultdict(int)
    emmission_counts = defaultdict(int)
    state_cnts = defaultdict(int)
    sum_emm_prob = 0
    
    ## State Counts
    for sentence in y_train:
        for state in sentence:
            state_cnts[state] += 1
            
    ## New Word Emmission Probs based on frequency in training data (Heuristic)
    for tag in tags:
        sum_emm_prob += math.log(state_cnts[tag])
    for tag in tags:
        new_emmission_probs[tag] = math.log(state_cnts[tag]) - sum_emm_prob

    ## Bigram Transition Probabilities
    for sentence in y_train:
        for i in range(1, len(sentence)):
            bigram_transition_counts[(sentence[i], sentence[i - 1])] += 1
    for key, count in bigram_transition_counts.items():
        transition_probs[key] = math.log(count) - math.log(state_cnts[key[1]])
        
    ## Trigram Transition Probabilites
    if transition_type == "Trigram":
        for sentence in y_train:
            for i in range(2,len(sentence)):
                trigram_transition_counts[(sentence[i], sentence[i-1], sentence[i-2])] += 1
        for key, count in trigram_transition_counts.items():
            transition_probs[key] = math.log(count) - math.log(bigram_transition_counts[(key[1], key[2])])
    
    ## Emmision Probabilities
    for i in range(len(X_train)):
        for j in range(len(X_train[i])):
            emmission_counts[(X_train[i][j], y_train[i][j])] += 1
    for key, count in emmission_counts.items():
        emmission_probs[key] = math.log(count) - math.log(state_cnts[key[1]])
    
    return


# ### Viterbi Decoding Algorithm

NEG_INF = -10000 # Since log probabilites taken to avoid underflow

def viterbi_decoding(obs_seq, transition_type):
    T = len(obs_seq)
    if transition_type == "Bigram":
        
        # Declare viterbi and backptr matrix (N x T)  
        viterbi = np.full((n_states,T), NEG_INF, dtype=float)
        backptr = np.full((n_states, T), int(-1), dtype=int)
        
        # Initializing starting probability
        viterbi[n_states-1][0] = 0
        
        # Applying Dynamic Programming
        for j in range(1,T):
            for i in range(n_states):
                for k in range(n_states):
                    emp = emmission_probs.get((obs_seq[j],tags[i]),NEG_INF)
                    
                    # If new word found use new word emmission probabilites
                    if emp == NEG_INF: 
                        emp = new_emmission_probs.get(tags[i],NEG_INF)
                    
                    # viterbi[i][j] = max(viterbi[k][j-1] * transition_prob * emmission prob)
                    cur = viterbi[k][j-1] + transition_probs.get((tags[i],tags[k]),NEG_INF) + emp
                    if cur > viterbi[i][j]:
                        viterbi[i][j] = cur  
                        backptr[i][j] = k
        
        # Finding bestpath using the backptr
        bestpath = []
        prob=NEG_INF
        ptr=int(-1)
        for i in range(n_states):
            if viterbi[i][T-1]>prob:
                prob=viterbi[i][T-1]
                ptr=i
        j=T-1
        while ptr > -1:
            bestpath.append(tags[ptr])
            ptr=backptr[ptr][j]
            j -= 1
        bestpath.reverse()
        return bestpath
    
    else:
        
        # Declare viterbi and backptr matrix (N x N x T)      
        viterbi = np.full((n_states, n_states, T), NEG_INF, dtype=float)
        backptr = np.full((n_states, n_states, T), int(-1), dtype=int)
        
        # Initializing starting probabilities using bigram transitions
        for i in range(n_states):
            emp = emmission_probs.get((obs_seq[1],tags[i]),NEG_INF)
            
            # If new word found use new word emmission probabilites
            if emp == NEG_INF:
                emp = new_emmission_probs.get(tags[i],NEG_INF)
            
            cur = transition_probs.get((tags[i],tags[-1]), NEG_INF) + emp
            if cur > viterbi[i][n_states-1][1]:
                viterbi[i][n_states-1][1] = cur  
                
        # Applying Dynamic Programming for Trigram
        for j in range(2,T):
            for i in range(n_states):
                for k in range(n_states):
                    for l in range(n_states):
                        emp = emmission_probs.get((obs_seq[j],tags[i]),NEG_INF)

                        # If new word found use new word emmission probabilites
                        if emp == NEG_INF:
                            emp = new_emmission_probs.get(tags[i],NEG_INF)
                            
                        # viterbi[i][k][j] = max(viterbi[k][l][j-1] * transition_prob * emmission prob)
                        cur = viterbi[k][l][j-1] + transition_probs.get((tags[i],tags[k],tags[l]),NEG_INF) + emp
                        if cur > viterbi[i][k][j]:
                            viterbi[i][k][j] = cur  
                            backptr[i][k][j] = l
        
        # Finding bestpath using the backptr  
        bestpath = []
        prob=NEG_INF
        ptr=(int(-1),int(-1))
        for i in range(n_states):
            for k in range(n_states):
                if viterbi[i][k][T-1]>prob:
                    prob=viterbi[i][k][T-1]
                    ptr=(i,k)
        j=T-1
        while ptr[0] > -1:
            bestpath.append(tags[ptr[0]])
            ptr=(ptr[1],backptr[ptr[0]][ptr[1]][j])
            j -= 1
        bestpath.reverse()
        
        return bestpath


# ### Testing

from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import itertools

def Testing(X_test, y_test, transition_type):
    correct = 0
    total = sum(len(seq) for seq in y_test)
    y_estimates = []
    
    # Finding out Accuracy
    for x_seq, y_seq in zip(X_test, y_test):
        bestpath = viterbi_decoding(x_seq, transition_type)
        y_estimates.append(bestpath)
        correct += sum(1 for pred, true in zip(bestpath, y_seq) if pred == true)
        
    y_test = list(itertools.chain(*y_test)) # Converting into single list
    y_estimates = list(itertools.chain(*y_estimates))
    
    # Building Confusion Matrix
    confusion_matrix = metrics.confusion_matrix(y_test, y_estimates, labels=tags)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = tags)
    cm_display.plot()
    plt.show()
    
    # Showing Classification Report
    print("##########################################################")
    print(transition_type.upper()," CLASSIFICATION REPORT FOR FOLD 1\n")
    print(classification_report(y_test, y_estimates, labels=tags))
    print("##########################################################")
    
    accuracy = correct / total if total > 0 else 0
    return accuracy


# ### PoS Tagging

import multiprocessing # Since Trigram is very slow on cpu

def train_and_test(train_index, test_index, transition_type, X, Y, kf):
    X_train = [X[j] for j in train_index]
    y_train = [Y[j] for j in train_index]
    X_test = [X[j] for j in test_index]
    y_test = [Y[j] for j in test_index]
    
    # Training the Model / Calculating Transition and Emmission Probs
    Training(X_train, y_train, transition_type) 
    testing_accuracy = Testing(X_test, y_test, transition_type)

    return testing_accuracy

def PoSTagging(X, transition_type):
    testing_accuracies = []
    num_splits = kf.get_n_splits(X)
    pool = multiprocessing.Pool(processes=min(num_splits, multiprocessing.cpu_count()))
    
    results = []
    for train_index, test_index in kf.split(X):
        x = train_and_test(train_index, test_index, transition_type, X, Y, kf)
        return x
        result = pool.apply_async(train_and_test, (train_index, test_index, transition_type, X, Y, kf))
        results.append(result)
    pool.close()
    pool.join()
    for result in results:
        testing_accuracy = result.get()
        testing_accuracies.append(testing_accuracy)

    return testing_accuracies


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

    # ### Getting Training and Testing Accuracies
    bigram_testing_accuracies = PoSTagging(X, "Bigram")
    trigram_testing_accuracies = PoSTagging(X, "Trigram") # Might take some time even with multiprocessing
    print(bigram_testing_accuracies)
    print(trigram_testing_accuracies)
    
