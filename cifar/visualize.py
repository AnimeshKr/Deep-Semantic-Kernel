import pickle
from pylearn2.utils import serial
import sys
from PIL import Image
import math
import numpy as np

def normalize(i1):
	maxi, mini=max(i1), min(i1)
	for i in range(len(i1)):
		i1[i] = float(i1[i]-mini)/float(maxi-mini)
		i1[i] = i1[i] * 255.0
	return i1

model = serial.load(sys.argv[1])
l1= model.layers[0]
w1 = l1.get_weights()

for i in range(w1.shape[1]):
	i1 = w1[:,i]
	i1,i2 = i1[:w1.shape[0]/2],i1[w1.shape[0]/2:]
	im1 = normalize(i1)
	im2 = normalize(i2)
	print 'normalization done!'

	im1 = im1.reshape((math.sqrt(len(im1)),-1)) 
	im2 = im2.reshape((math.sqrt(len(im2)),-1)) 
	img1 = Image.fromarray(im1).convert('RGB')
	img1.save('visualization/im1_'+ str(i) +'.png')
	img2 = Image.fromarray(im2).convert('RGB')
	img2.save('visualization/im2_'+ str(i) +'.png')


