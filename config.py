#coding=utf-8
from __future__ import print_function

class DefaultConfig(object):

	root = '/fanjiahui/repositories/pytorch-cifar10'

	input_channels = 3
	batch_size = 128
	use_gpu = True
	num_workers = 2
	print_freq = 100# 每几个batch打印一次信息

	max_epoch = 100
	lr = 0.1
	lr_decay = 0.7
	momentum=0.9
	weight_decay = 5e-4

	debug_file = './debug'
	
	def __init__(self):
		print('User config: ')
		for k, v in self.__class__.__dict__.items():
			if not k.startswith('__'):
				print(k, getattr(self, k))
				