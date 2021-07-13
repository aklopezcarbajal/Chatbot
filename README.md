# PyTorch Chatbot


Simple Chatbot built with PyTorch.
The model is a Feedforward Neural Net with 2 hidden layers.


<img src="/img/chatbot.gif" alt="Chat" width="530" height="300"/>

## Running the application

In the command line, navigate to the folder where main.py and chatbotmodel.pth are located. Then type:
```bash
# run chatbot application
python main.py
```

Type **"quit"** to close the application.

## About the intents

The following intents were used to train the bot:
<li>greeting</li>
<li>name</li>
<li>hobby</li>
<li>joke</li>
<li>cheer</li>
<li>goodbye</li>

If you wish to customize the bot, modify the file `intents.json`
```bash
{"intents":[
	{"tag": "greeting",
		 "patterns": ["Hello", "Hi", "Hi there", "Hey", "Whats up"],
		 "responses": ["Hi friend!", "Hello", "Hello, nice to see you.", "Greetings!", "Good to see you." ]
	},
  ...
}
```
and re-train the model typing in the command line:
```bash
# run training
python train.py
```

