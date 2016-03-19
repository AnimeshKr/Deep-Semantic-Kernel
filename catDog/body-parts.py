import pickle
from pylearn2.utils import serial
import sys
import pylab
from PIL import Image
import math
import numpy as np
import matplotlib.pyplot as plt
print 'loaded dependencies'
def normalize(i1):
	"""
	maxi, mini=max(i1), min(i1)
	if maxi != mini :
  	  for i in range(len(i1)):
		i1[i] = float(i1[i]-mini)/float(maxi-mini)
		i1[i] = i1[i] * 255.0
	return i1
	"""
	for i in range( len(i1) ):
		if i1[i] < 0:
			i1[i] = 0.
	return i1
 
print 'loading model'
data = pickle.load( open( 'Train-body-parts.pkl') )
X = data['X']
X = X/X.max()
X = X*255.
for i in range(X.shape[0]):
	x = X[i]
	#fig, axes = plt.subplots(nrows=256, ncols=3)
	for j in range( 256 ):
		im1 = x[i]
			
		
		img1 = Image.fromarray(im1).convert('RGB')
		img1.save('body-parts/im_'+ str(i) + '_' + str(j)+'.png')
		
		plt.subplot(16, 16, j); pylab.axis('off'); pylab.imshow(img1)

	#fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
	plt.savefig('/home/amit-pc/caffe/python/Final/body-parts/'+str(i+1)+'.eps', format='eps', dpi=1000)	
	plt.clf()

