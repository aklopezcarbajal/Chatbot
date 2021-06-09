import torch
import torch.nn as nn

class Net(nn.Module):
	def __init__():
		super().__init__(input_size, layer_size, output_size)
		self.layer1 = nn.Linear(input_size,layer_size)
		self.layer2 = nn.Linear(layer_size,layer_size)
		self.layer3 = nn.Linear(layer_size,output_size)

	def forward(self, x):
		output = nn.ReLU(self.layer1(x))
		output = nn.ReLU(self.layer2(output))
		output = self.layer3(output)

		return output