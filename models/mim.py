

import torch
import torch.nn as nn

from layers.SpatioTemporalLSTMCellv2 import SpatioTemporalLSTMCell as stlstm
from layers.MIMBlock import MIMBlock as mimblock
from layers.MIMN import MIMN as mimn
import math


class MIM(nn.Module): # stlstm 
	def __init__(self, shape, num_layers, num_hidden, filter_size,
		total_length=20, input_length=10, tln=True):
		super(MIM, self).__init__()
		
		self.num_layers = num_layers
		self.num_hidden = num_hidden
		self.filter_size = filter_size
		self.total_length = total_length
		self.input_length = input_length
		self.tln = tln
		
		self.gen_images = [] # 存储生成的图片
		self.stlstm_layer = nn.ModuleList() # 存储 stlstm 和 mimblock
		self.stlstm_layer_diff = nn.ModuleList() # 存储 mimn
		self.cell_state = [] # 存储 stlstm_layer 的记忆
		self.hidden_state = [] # 存储 stlstm_layer 的隐藏层输出
		self.cell_state_diff = [] # 存储 stlstm_layer_diff 的记忆
		self.hidden_state_diff = [] # 存储 stlstm_layer_diff 的隐藏层输出
		self.shape = shape # 输入形状
		self.output_channels = shape[-3] # 输出的通道数
		
		for i in range(self.num_layers): # 隐藏层数目
			if i == 0:
				num_hidden_in = self.num_hidden[self.num_layers - 1] # 隐藏层的输入 前一时间段最后一层的输出为后一时间段第一层的输入
			else:
				num_hidden_in = self.num_hidden[i - 1] # 隐藏层的输入
			if i < 1: # 初始层 使用 stlstm
				new_stlstm_layer = stlstm('stlstm_' + str(i + 1),
							  self.filter_size,
							  num_hidden_in,
							  self.num_hidden[i],
							  self.shape,
							  self.output_channels,
							  tln=self.tln)
			else: # 后续层 使用 mimblock
				new_stlstm_layer = mimblock('stlstm_' + str(i + 1),
								self.filter_size,
								num_hidden_in,
								self.num_hidden[i],
								self.shape,
								self.num_hidden[i-1],
								tln=self.tln)
			self.stlstm_layer.append(new_stlstm_layer) # 列表
			self.cell_state.append(None) # 记忆
			self.hidden_state.append(None) # 状态

		for i in range(self.num_layers - 1): # 添加 MIMN
			new_stlstm_layer = mimn('stlstm_diff' + str(i + 1),
								self.filter_size,
								self.num_hidden[i + 1],
								self.shape,
								tln=self.tln)
			self.stlstm_layer_diff.append(new_stlstm_layer) # 列表
			self.cell_state_diff.append(None) # 记忆
			self.hidden_state_diff.append(None) # 状态

		self.st_memory = None # 空间存储
		
		# 生成图片
		self.x_gen = nn.Conv2d(self.num_hidden[self.num_layers - 1],
				 self.output_channels,1,1,padding=0
				 )
		
	def forward(self, images, schedual_sampling_bool):
		
		for time_step in range(self.total_length - 1): # 时间步长
			print('time_step: '+str(time_step))
			
			if time_step < self.input_length: # 小于输入步长
				x_gen = images[:,time_step] # 输入大小为 [batch, in_channel,in_height, in_width]
			else:
				# 掩模 mask
				x_gen = schedual_sampling_bool[:,time_step-self.input_length]*images[:,time_step] + \
						(1-schedual_sampling_bool[:,time_step-self.input_length])*x_gen
						
			preh = self.hidden_state[0] # 初始化状态
			self.hidden_state[0], self.cell_state[0], self.st_memory = self.stlstm_layer[0]( # 使用的是 stlstm 输出 hidden_state[0], cell_state[0], st_memory
				x_gen, self.hidden_state[0], self.cell_state[0], self.st_memory)
			
			# 对于 mimblock
			for i in range(1, self.num_layers):
				print('i: '+str(i))
				if time_step > 0:
					if i == 1:
						self.hidden_state_diff[i - 1], self.cell_state_diff[i - 1] = self.stlstm_layer_diff[i - 1]( # 先求出 mimn
							self.hidden_state[i - 1] - preh, self.hidden_state_diff[i - 1], self.cell_state_diff[i - 1])
					else:
						self.hidden_state_diff[i - 1], self.cell_state_diff[i - 1] = self.stlstm_layer_diff[i - 1]( # 先求出 mimn
							self.hidden_state_diff[i - 2], self.hidden_state_diff[i - 1], self.cell_state_diff[i - 1])
				else:
					self.stlstm_layer_diff[i - 1](torch.zeros_like(self.hidden_state[i - 1]), None, None)
				
				# 接下来计算 mimblock	
				preh = self.hidden_state[i]
				self.hidden_state[i], self.cell_state[i], self.st_memory = self.stlstm_layer[i]( # mimblock
					self.hidden_state[i - 1], self.hidden_state_diff[i - 1], self.hidden_state[i], self.cell_state[i], self.st_memory)
				
			# 生成图像 取最后一层的隐藏层状态
			x_gen = self.x_gen(self.hidden_state[self.num_layers - 1])
			
			self.gen_images.append(x_gen)

		self.gen_images = torch.stack(self.gen_images, dim=1)
		loss_fn = nn.MSELoss()
		loss = loss_fn(self.gen_images,images[:, 1:])
		return [self.gen_images, loss]

		
