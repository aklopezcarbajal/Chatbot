import torch
import nltk
#nltk.download('punkt')
import json
import random
from model import Net
from utils import bag_of_words

device = torch.device('cpu')
print('[INF] Device:', device)

# Load intents file
with open('intents.json', 'r') as f:
	file = json.load(f)

# Load model
model_dict = torch.load('chatmodel.pth')

input_size, layer_size, output_size = model_dict['dimensions']
model_state = model_dict['model_state']

net = Net(input_size, layer_size, output_size)
net.load_state_dict(model_state)
net.eval()

words, labels = model_dict['data']
softmax = torch.nn.Softmax(dim=0)

while True:
	string = input('You: ')
	if string.lower() == 'quit':
		break

	bag = bag_of_words(nltk.word_tokenize(string),words)
	t = torch.from_numpy(bag) # bag of words to tensor
	# Feed input to model
	output = net(t)
	# Guess intent
	val, ind = torch.max(output, dim=0)
	tag = labels[ind.item()]

	p = softmax(output)
	# If the likelihood is greater than 0.5
	if p[ind.item()] > 0.5:
		# Choose a random response
		for intent in file['intents']:
			if tag == intent['tag']:
				responses = intent['responses']
				print('Bot:', random.choice(responses))
	else:
		print('Bot: Sorry, I do not understand...')