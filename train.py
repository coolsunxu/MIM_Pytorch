
import torch

from models.mim import MIM

a = torch.randn((5,6,1,64,64))
b = torch.randn((5,2,1,64,64))

num_layers = 3
num_hidden = [64,64,64]
filter_size = 5
total_length = a.shape[1]
input_length = a.shape[1]
shape = [5,6,1,64,64]

stlstm = MIM(shape, num_layers, num_hidden, filter_size, total_length, input_length)
		 
new = stlstm(a,b)
print(new[0].shape)
print(new[1])
