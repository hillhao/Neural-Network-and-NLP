##-------------------------------------------------------------
##Module Name:              train_lstm.py
##Department:               
##Function Description:     set the training parameters,load data from csv file, 
##													define lstm network topology, display training processing
##                          verify the model
##--------------------------------------------------------------
##Version       Design       Coding         Simulate          Review          Reldata
##V1.0          YUFENG HAO   YUFENG HAO                                      2017-07-28
##-----------------------------------------------------------------
##Version               Modified History           
##V1.0                  
##--------------------------------Key words--------------------------  

import sys
import os
import time
import numpy as np
from data_preprocessing import *
from datetime import datetime
from lstm import LSTMRNN
import csv
import itertools
import nltk
import operator
import io
import array
from datetime import datetime

SENTENCE_START_TOKEN = "SENTENCE_START"
SENTENCE_END_TOKEN = "SENTENCE_END"
UNKNOWN_TOKEN = "UNKNOWN_TOKEN"

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "100"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "48"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "100"))
NEPOCH = int(os.environ.get("NEPOCH", "100"))
MODEL_OUTPUT_FILE = os.environ.get("MODEL_OUTPUT_FILE")
INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "./data/reddit-comments-2015-08.csv")
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "500"))

if not MODEL_OUTPUT_FILE:
  ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
  MODEL_OUTPUT_FILE = "GRU-%s-%s-%s-%s.dat" % (ts, VOCABULARY_SIZE, EMBEDDING_DIM, HIDDEN_DIM)



x_train, y_train, word_to_index, index_to_word = load_data(INPUT_DATA_FILE, VOCABULARY_SIZE)

# Build model
model = LSTMRNN(VOCABULARY_SIZE, hidden_dim=HIDDEN_DIM, bptt_truncate=-1)
   
# Print SGD step time
t1 = time.time()
model.sgd_step(x_train[10], y_train[10], LEARNING_RATE)
t2 = time.time()
print ("SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.))
print ("Start time is " , datetime.now().isoformat())
sys.stdout.flush()

# We do this every few examples to understand what's going on
def sgd_callback(model, num_examples_seen):
  dt = datetime.now().isoformat()
  loss = model.calculate_loss(x_train[:10], y_train[:10])
  print("\n%s (%d)" % (dt, num_examples_seen))
  print("Loss: %f" % loss)
  generate_sentences(model, 50, index_to_word, word_to_index)
  save_model_parameters_theano(model, MODEL_OUTPUT_FILE)
  print("\n")
  sys.stdout.flush()

for epoch in range(NEPOCH):
  train_with_sgd(model, x_train[:100], y_train[:100], learning_rate=LEARNING_RATE, nepoch=10, decay=0.9, 
    callback_every=PRINT_EVERY, callback=sgd_callback)

