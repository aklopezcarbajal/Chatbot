import torch
import nltk
#nltk.download('punkt')
from model import Net
from utils import bag_of_words

device = torch.device('cuda:0' if torch.cuda.is_available() else'cpu')
print('[INF] Device:', device)

# Load model
model_dict = torch.load('chatmodel.pth')

input_size, layer_size, output_size = model_dict['dimensions']
model_state = model_dict['model_state']

net = Net(input_size, layer_size, output_size)
net.to(device)
net.load_state_dict(model_state)
net.eval()

words, labels = model_dict['data']

while True:
	string = input('You:')
	if string.lower() == 'quit':
		break

	bag = bag_of_words(nltk.word_tokenize(string),words)
	t = torch.from_numpy(bag) # bag of words to tensor
	# Feed input to model
	output = net(t)
	# Guess intent
	val, ind = torch.max(output, dim=0)
	tag = labels[ind.item()]
	# Choose a random response
