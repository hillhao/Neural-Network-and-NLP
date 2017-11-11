import numpy as np
import theano as theano
import theano.tensor as T
import time
import operator
from data_preprocessing import load_data, load_model_parameters_theano, generate_sentences
from gru_theano import *
import sys


# Load data (this may take a few minutes)
VOCABULARY_SIZE = 8000
X_train, y_train, word_to_index, index_to_word = load_data("data/reddit-comments-2015.csv", VOCABULARY_SIZE)

# Load parameters of pre-trained model
model = load_model_parameters_theano('./data/pretrained.npz')

generate_sentences(model, 100, index_to_word, word_to_index)
