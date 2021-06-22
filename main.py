import torch
from model import Net

device = torch.device('cuda:0' if torch.cuda.is_available() else'cpu')

# Load model
model_dict = torch.load('chatmodel.pth')

input_size, layer_size, output_size = model_dict['dimensions']
model_state = model_dict['model_state']

net = Net(input_size, layer_size, output_size)
net.to(device)
net.load_state_dict(model_state)
net.eval()