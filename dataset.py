import torch
from torch.utils.data import Dataset

class ChatbotDataset(Dataset):
	def __init__(self, patterns, labels):
		self.patterns = patterns
		self.labels = labels

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, i):
		return self.patterns[i], self.labels[i]
