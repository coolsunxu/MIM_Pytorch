
import torch
import torch.nn as nn

import sys
sys.path.append("..")
from layers.TensorLayerNorm import tensor_layer_norm

class MIMN(nn.Module):
	def __init__(self, layer_name, filter_size, num_hidden, seq_shape, tln=True, initializer=0.001):
		super(MIMN, self).__init__()
		"""Initialize the basic Conv LSTM cell.
		Args:
			layer_name: layer names for different convlstm layers.
			filter_size: int tuple thats the height and width of the filter.
			num_hidden: number of units in output tensor.
			tln: whether to apply tensor layer normalization.
		"""
		self.layer_name = layer_name # 当前网络层名
		self.filter_size = filter_size # 卷积核大小
		self.num_hidden = num_hidden # 隐藏层大小
		self.layer_norm = tln # 是否归一化
		self.batch = seq_shape[0] # batch_size
		self.height = seq_shape[3] # 图片高度
		self.width = seq_shape[4] # 图片宽度
		self._forget_bias = 1.0 # 遗忘参数
			
		# h_t
		self.h_t = nn.Conv2d(self.num_hidden,
					self.num_hidden * 4,
					self.filter_size, 1, padding=2
					)
					
		# c_t
		self.ct_weight = nn.Parameter(torch.randn((self.num_hidden*2,self.height,self.width)))

		# x
		self.x = nn.Conv2d(self.num_hidden,
					self.num_hidden * 4,
					self.filter_size, 1, padding=2
					)
					
		# oc
		self.oc_weight = nn.Parameter(torch.randn((self.num_hidden,self.height,self.width)))
					
		# bn 
		self.bn_h_concat = tensor_layer_norm(self.num_hidden * 4)
		self.bn_x_concat = tensor_layer_norm(self.num_hidden * 4)

	def init_state(self): # 初始化lstm 隐藏层状态
		shape = [self.batch, self.num_hidden, self.height, self.width]
		return torch.zeros(shape, dtype=torch.float32)

	def forward(self, x, h_t, c_t):
		
		# h c [batch, num_hidden, in_height, in_width]
		
		# 初始化隐藏层 记忆 空间
		
		if h_t is None:
			h_t = self.init_state()
		if c_t is None:
			c_t = self.init_state()
		
		# 1
		h_concat = self.h_t(h_t)
		
		if self.layer_norm:
			h_concat = self.bn_h_concat(h_concat)
		i_h, g_h, f_h, o_h = torch.split(h_concat, self.num_hidden, 1)
		
		# 2 变量 可训练
		ct_activation = torch.mul(c_t.repeat([1,2,1,1]), self.ct_weight)
		i_c, f_c = torch.split(ct_activation, self.num_hidden, 1)

		i_ = i_h + i_c
		f_ = f_h + f_c
		g_ = g_h
		o_ = o_h

		if x is not None:
			# 3 x
			x_concat = self.x(x)
			
			if self.layer_norm:
				x_concat = self.bn_x_concat(x_concat)
			i_x, g_x, f_x, o_x = torch.split(x_concat, self.num_hidden, 1)

			i_ += i_x
			f_ += f_x
			g_ += g_x
			o_ += o_x

		i_ = torch.sigmoid(i_)
		f_ = torch.sigmoid(f_ + self._forget_bias)
		c_new = f_ * c_t + i_ * torch.tanh(g_)

		# 4 变量 可训练
		o_c = torch.mul(c_new, self.oc_weight)

		h_new = torch.sigmoid(o_ + o_c) * torch.tanh(c_new)

		return h_new, c_new # 大小均为 [batch, in_height, in_width, num_hidden]
		
if __name__ == '__main__':
	a = torch.randn((32,64,48,48))
	layer_name = 'stlstm'
	filter_size = 5
	num_hidden_in = 64
	num_hidden = 64
	seq_shape = [32,12,64,48,48]
	tln = True
	
	stlstm = MIMN(layer_name, filter_size, num_hidden,
				 seq_shape, tln)
	
	new_h, new_c = stlstm(a,None,None)
	print(new_h.shape)
	print(new_c.shape)

