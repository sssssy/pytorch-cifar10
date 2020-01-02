#coding=utf-8
from __future__ import print_function

class DefaultConfig(object):
	model = 'CaviModel'

	root = 'E:/Pytorch Projects/Caricature recognition/IIIT-CFW/IIIT-CFW1.0/'
	load_model_path = None#'/data2/fanjiahui/CaricatureRecognition/test/checkpoints/'
	# 需要加载模型或者测试时才要给这个参数

	input_channels = 3
	batch_size = 64
	use_gpu = True
	num_workers = 4
	print_freq = 20# 每几个batch打印一次信息
	val_freq = 400# 每几个batch调用一次val，以供调试

	max_epoch = 10
	lr = 0.001
	lr_decay = 0.95
	weight_decay = 1e-4
	threshold = 0.5# softmax输出的判别阈值

	debug_file = './debug'

	def parse(self, kwargs):
		'''
		根据字典kwargs更新参数
		'''
		for k, v in kwargs.items():
			if not hasattr(self, k):
				print("\n!!!Warning: opt has no attribute %s!!!\n" %k)
			setattr(self, k, v)

		print('User config: ')
		for k, v in self.__class__.__dict__.items():
			if not k.startswith('__'):
				print(k, getattr(self, k))