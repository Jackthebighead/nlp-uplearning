import numpy as np
import re
import os
from collections import Counter
from keras.utils import to_categorical
import math

MAX_SEQ_LEN = 30
MIN_FREQ = 5
def read_data_from_file(filename):
	sentence = []
	with open(filename, "r") as fin:
		for line in fin:
			line = re.sub(r"\n", "<eos>", line)
			sentence += line.split(' ')
	return sentence


def build_vocab(sentence):
	"""
	Builds a vocabulary mapping from word to index based on the sentences.
	Remove low frequency words.
	Returns vocabulary.
	"""
	vocabulary = {}
	cnter = Counter(sentence)

	for k in list(cnter):
		if cnter[k] < MIN_FREQ:
			 del cnter[k]
	for word in list(cnter.keys()): 
		vocabulary[word] = len(vocabulary)
	if '<unk>' not in vocabulary.keys():
		vocabulary['<unk>'] = len(vocabulary)
	return vocabulary

def build_input_data(sentence, vocabulary):
	"""
	Maps sentences and labels to vectors based on a vocabulary.
	"""
	unknown_token_id = vocabulary['<unk>']
	vocab = vocabulary.keys()
	vocab_size = len(vocab)
	sentence_id = [ vocabulary[word] if word in vocab else unknown_token_id for word in sentence ]
	
	

	x = []
	y_categorical = []
	y = []

	num_sentences = math.ceil( len(sentence_id)/MAX_SEQ_LEN )
	'''
	The the len of last sentence may be less than MAX_SEQ_LEN, so we pad it using tokens in the begining.
	corpus: a cat sits on the mat
	x(text): a cat sits
	y(text): cat sits on
	'''
	sentence_id += sentence_id[:MAX_SEQ_LEN+1]

	true_data = []
	for i in range( num_sentences ):
		true_data.append( sentence_id[i*MAX_SEQ_LEN:(i+1)*MAX_SEQ_LEN + 1] )
		# x.append( sentence_id[i*MAX_SEQ_LEN:(i+1)*MAX_SEQ_LEN] )
		# y_ = sentence_id[i*MAX_SEQ_LEN+1:(i+1)*MAX_SEQ_LEN+1]
		# y.append(y_)

	true_data = np.array(true_data)
	x = true_data[:, 0:-1]
	y = true_data[:, 1:]
	y = np.expand_dims(y, axis=-1)
	# x = np.array(x)
	# # y_categorical = np.array(y_categorical)
	return x, None, y

def load_data(train_path = "data/ptb.train.txt", valid_path = "data/ptb.valid.txt"):
	# get the data paths

	train_data = read_data_from_file(train_path)
	valid_data = read_data_from_file(valid_path)

	# build vocabulary from training data
	vocabulary = build_vocab(train_data) 
	vocab_size = len(vocabulary)

	# get input data
	x_train, _, y_train = build_input_data(train_data,vocabulary)
	x_valid, _, y_valid = build_input_data(valid_data,vocabulary)

	return x_train, y_train, x_valid, y_valid, vocab_size, vocabulary
# x_train, y_train, x_valid, y_valid, x_test, y_test, vocab_size = load_data()


if __name__ == "__main__":
	x_train, y_train, x_valid, y_valid, vocab_size, vocabulary = load_data(train_path="data/ptb.valid.txt")
	import pdb
	pdb.set_trace()
