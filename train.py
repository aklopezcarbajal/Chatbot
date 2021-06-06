import json
import nltk

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