#coding=utf-8
from __future__ import print_function

class DefaultConfig(object):

	root = '/fanjiahui/repositories/pytorch-cifar10'
	load_model_path = None
	# 需要加载模型或者测试时才要给这个参数

	input_channels = 3
	batch_size = 64
	use_gpu = True
	num_workers = 4
	print_freq = 500# 每几个batch打印一次信息
	val_freq = 400# 每几个batch调用一次val，以供调试

	max_epoch = 10
	lr = 0.001
	lr_decay = 0.95
	weight_decay = 1e-4

	debug_file = './debug'
	
	def __init__(self):
		print('User config: ')
		for k, v in self.__class__.__dict__.items():
			if not k.startswith('__'):
				print(k, getattr(self, k))
				