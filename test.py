import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torchvision.transforms as transforms
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from config import DefaultConfig

if __name__ == "__main__":

	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	test_image_path = 'test.jpg'
	load_model_path = 'checkpoints/01021643.pth'

	# read image

	if os.path.exists(load_model_path):
		net = torch.load(load_model_path)
	else:
		print("No model")

	if os.path.exists(test_image_path):
		img = Image.open(test_image_path)
	else :
		print("No test image")

	transform = transforms.Compose([
				transforms.Resize(32, interpolation=2),
    			transforms.CenterCrop(32),
    			transforms.ToTensor(),
    			 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

	img = transform(img)
	img = np.array(img)

	x = torch.from_numpy(img)
	x = torch.unsqueeze(x, dim=0)
	print(x.size())

	# test

	net.eval()
	with torch.no_grad():
		output = net(V(x.cuda()))
		_, predicted = torch.max(output,1)
		print(classes[predicted])
