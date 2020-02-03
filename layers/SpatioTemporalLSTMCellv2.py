
import math

import torch
import torch.nn as nn

import sys
sys.path.append("..")
from layers.TensorLayerNorm import tensor_layer_norm

class SpatioTemporalLSTMCell(nn.Module): # stlstm 
	def __init__(self, layer_name, filter_size, num_hidden_in, num_hidden,
				 seq_shape, x_shape_in, tln=False, initializer=None):
		super(SpatioTemporalLSTMCell, self).__init__()
		
		"""Initialize the basic Conv LSTM cell.
		Args:
			layer_name: layer names for different convlstm layers.
			filter_size: int tuple thats the height and width of the filter.
			num_hidden: number of units in output tensor.
			forget_bias: float, The bias added to forget gates (see above).
			tln: whether to apply tensor layer normalization
		"""
		self.layer_name = layer_name # 当前网络层名
		self.filter_size = filter_size # 卷积核大小
		self.num_hidden_in = num_hidden_in # 隐藏层输入大小
		self.num_hidden = num_hidden # 隐藏层数量
		self.batch = seq_shape[0] # batch_size
		self.height = seq_shape[3] # 图片高度
		self.width = seq_shape[4] # 图片宽度
		self.x_shape_in = x_shape_in # 通道数
		self.layer_norm = tln # 是否归一化
		self._forget_bias = 1.0 # 遗忘参数
			
		# 建立网络层
		# h
		self.t_cc = nn.Conv2d(self.num_hidden_in,
				self.num_hidden*4, # 网络输入 输出通道数
				self.filter_size, 1, padding = 2 # 滤波器大小 步长 填充方式
				)
				
		# m
		self.s_cc = nn.Conv2d(self.num_hidden_in,
				self.num_hidden*4,  # 网络输入 输出通道数
				self.filter_size, 1, padding = 2 # 滤波器大小 步长 填充方式
				)
				
		# x
		self.x_cc = nn.Conv2d(self.x_shape_in,
				self.num_hidden*4, # 网络输入 输出通道数
				self.filter_size, 1, padding = 2 # 滤波器大小 步长 填充方式
				)
		
		# c 
		self.c_cc = nn.Conv2d(self.num_hidden*2,
				self.num_hidden,  # 网络输入 输出通道数
				1, 1, padding = 0 # 滤波器大小 步长 填充方式
				)
				
		# bn
		self.bn_t_cc = tensor_layer_norm(self.num_hidden*4)
		self.bn_s_cc = tensor_layer_norm(self.num_hidden*4)
		self.bn_x_cc = tensor_layer_norm(self.num_hidden*4)
				
	def init_state(self): # 初始化lstm 隐藏层状态
		return torch.zeros((self.batch, self.num_hidden, self.height, self.width),
						dtype=torch.float32)

	def forward(self, x, h, c, m):
		# x [batch, in_channels, in_height, in_width]
		# h c m [batch, num_hidden, in_height, in_width]
		
		# 初始化隐藏层 记忆 空间
		if h is None:
			h = self.init_state()
		if c is None:
			c = self.init_state()
		if m is None:
			m = self.init_state()
		
		# 计算网络输出
		t_cc = self.t_cc(h)
		s_cc = self.s_cc(m)
		x_cc = self.x_cc(x)
		
		if self.layer_norm:
			# 计算均值 标准差 归一化
			t_cc = self.bn_t_cc(t_cc)
			s_cc = self.bn_s_cc(s_cc)
			x_cc = self.bn_x_cc(x_cc)
		
		# 在第3维度上切分为4份 因为隐藏层是4*num_hidden 
		i_s, g_s, f_s, o_s = torch.split(s_cc, self.num_hidden, 1) # [batch, num_hidden, in_height, in_width]
		i_t, g_t, f_t, o_t = torch.split(t_cc, self.num_hidden, 1)
		i_x, g_x, f_x, o_x = torch.split(x_cc, self.num_hidden, 1)

		i = torch.sigmoid(i_x + i_t)
		i_ = torch.sigmoid(i_x + i_s)
		g = torch.tanh(g_x + g_t)
		g_ = torch.tanh(g_x + g_s)
		f = torch.sigmoid(f_x + f_t + self._forget_bias)
		f_ = torch.sigmoid(f_x + f_s + self._forget_bias)
		o = torch.sigmoid(o_x + o_t + o_s)
		new_m = f_ * m + i_ * g_
		new_c = f * c + i * g
		cell = torch.cat((new_c, new_m),1) # [batch, 2*num_hidden, in_height, in_width]
		
		cell = self.c_cc(cell)
		new_h = o * torch.tanh(cell)

		return new_h, new_c, new_m # 大小均为 [batch, num_hidden, in_height, in_width]
		
if __name__ == '__main__':
	a = torch.rand((32,1,48,48))
	layer_name = 'stlstm'
	filter_size = 5
	num_hidden_in = 64
	num_hidden = 64
	seq_shape = [32,12,1,48,48]
	x_shape_in = 1
	tln = True
	
	stlstm = SpatioTemporalLSTMCell(layer_name, filter_size, num_hidden_in, num_hidden,
				 seq_shape, x_shape_in, tln)
			 
	new_h, new_c, new_m = stlstm(a,None,None,None)
	print(new_h.shape)
	print(new_c.shape)
	print(new_m.shape)
	









