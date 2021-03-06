import json
import nltk
#nltk.download('punkt')
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import Net
from utils import ChatbotDataset, bag_of_words
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

# Load data
with open('intents.json', 'r') as file:
	data = json.load(file)

words = []
labels = []
pairs = []

for intent in data['intents']:
	if intent['tag'] not in labels:
		labels.append(intent['tag'])

	for pattern in intent['patterns']:
		w = nltk.word_tokenize(pattern)
		words.extend(w)
		pairs.append((w,intent['tag']))

words = [stemmer.stem(w.lower()) for w in words if w != '?']
words = sorted(set(words))
labels = sorted(labels)

# Training data
X, y = [], []

for (pattern, tag) in pairs:
	bag = bag_of_words(pattern, words)
	label = labels.index(tag)
	X.append(bag)
	y.append(label)

X = np.array(X)
y = np.array(y)

dataset = ChatbotDataset(X,y)
train_dataloader = DataLoader(dataset,batch_size=16,shuffle=True)

# Training model
device = torch.device('cpu')
print('[INF] Device:', device)

input_size = len(words)
layer_size = 8
output_size = len(labels)

net = Net(input_size, layer_size, output_size)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=0.01)

epochs = 50

for epoch in range(epochs):
	for (pattern, tag) in train_dataloader:
		tag = tag.to(dtype=torch.long)

		optimizer.zero_grad()

		#forward - backward - optimize
		output = net(pattern)
		loss = criterion(output, tag)
		loss.backward()
		optimizer.step()
	if (epoch+1) % 5 == 0:
		print('[epoch',epoch+1,'] loss: %.3f' %loss.item())

print('Finished Training')

# Save trained model
model_dict = {
			'model_state': net.state_dict(),
			'dimensions': [input_size, layer_size, output_size],
			'data': [words, labels]
}

torch.save(model_dict, 'chatmodel.pth')