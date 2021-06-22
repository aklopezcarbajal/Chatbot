import torch
from torch.utils.data import Dataset
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
import nltk
import numpy as np

class ChatbotDataset(Dataset):
	def __init__(self, patterns, labels):
		self.patterns = patterns
		self.labels = labels

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, i):
		return self.patterns[i], self.labels[i]


def bag_of_words(text, words):
	text_words = [stemmer.stem(w.lower()) for w in text]

	bag = np.zeros(len(words), dtype=np.float32)
	for i, w in enumerate(words):
		if w in text:
			bag[i] = 1

	return bag