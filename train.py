import json
import nltk
#nltk.download('punkt')
import numpy as np
import torch
import torch.nn as nn
from model import Net
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

def bag_of_words(text, words):
	text_words = [stemmer.stem(w.lower()) for w in text]

	bag = np.zeros(len(words))
	for i, w in enumerate(words):
		if w in text:
			bag[i] = 1

	return bag

# Load data
with open('intents.json', 'r') as file:
	data = json.load(file)

words = []
classes = []
pairs = []

for intent in data['intents']:
	if intent['tag'] not in classes:
		classes.append(intent['tag'])

	for pattern in intent['patterns']:
		w = nltk.word_tokenize(pattern)
		words.extend(w)
		pairs.append((w,intent['tag']))

words = [stemmer.stem(w.lower()) for w in words if w != '?']
words = sorted(set(words))
classes = sorted(classes)

# Training data
X, y = [], []

for (pattern, tag) in pairs:
	bag = bag_of_words(pattern, words)
	label = classes.index(tag)
	X.append(bag)
	y.append(label)

X = np.array(X)
y = np.array(y)


# Training model
device = torch.device('cpu')

input_size = len(words)
layer_size = 5
output_size = len(classes)

net = Net(input_size, layer_size, output_size)
