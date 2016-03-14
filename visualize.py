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
	for i in range( len(i1) ):
		if i1[i] < 0:
			i1[i] = 0.
	return i1
 
w = pickle.load( open('weights.pkl') ) 
act_fc = pickle.load( open( '../pool5/body-part-features.pkl' ) )
act_conv = pickle.load(  open( '../pool5/Train-body-parts.pkl' ) )['X']
x = 3
y = 9
for x in range(12):
	for y in range(12):
		z = 12*x+y
		index = np.argsort(act_fc[z])[-1]
		print index
n = 256*6*6
c = 0

for i in range( 256 ):
	if act_conv[x][i].max() > 0 and act_conv[y][i].max() > 0:
 	 max1 = act_conv[x][i].max()
	 max2 = act_conv[y][i].max()
	 w1 = w2 =np.zeros(36)
	 for j in range( 36 ):
		w1[j] = w[i*36+j][index]*max1
		w2[j] = w[n+i*36+j][index]*max2
	 w1 = normalize(w1)
	 w2 = normalize(w2)
	 max = w1.max()
	 if max < w2.max():
		max = w2.max()
	 for j in range( 36 ):
		w1[j] = w1[j]*255./max
		w2[j] = w2[j]*255./max

	 w1 = w1.reshape((math.sqrt(len(w1)),-1)) 
	 w2 = w2.reshape((math.sqrt(len(w2)),-1)) 
	 img1 = Image.fromarray(w1).convert('RGB')
	 img2 = Image.fromarray(w2).convert('RGB')
	 plt.subplot(256, 2, 2*c ); pylab.axis('off'); pylab.imshow(img1)
	 plt.subplot(256, 2, 2*c+1 ); pylab.axis('off'); pylab.imshow(img2)
	 c += 1
print c
plt.savefig('/home/amit-pc/final_figures/weights-255threshold.eps', format='eps', dpi=1000)	
	
	
	

"""
print 'weights fetched'
print w1.shape
w1 = w1/w1.max()
w1 = w1*255.
indices = [346, 342, 639, 209, 843, 877, 796, 632, 385, 491]
for i in indices:
	W = w1[:,i]
	n = 256*6*6
	#fig, axes = plt.subplots(nrows=256, ncols=3)
	for j in range( 256 ):
		i1 = W[j*6*6:(j+1)*6*6]
		i2 = W[n+j*6*6:n+(j+1)*6*6]
		im1 = normalize(i1)
		im2 = normalize(i2)
		im3 = normalize(i1+i2)
		im4 = normalize(i1-i2)
		print i, j
		print 'normalization done!'
		if im1.max() + im2.max() + im3.max() + im4.max() < 10 :
			continue 	
		im1 = im1.reshape((math.sqrt(len(im1)),-1)) 
		im2 = im2.reshape((math.sqrt(len(im2)),-1)) 
		im3 = im3.reshape((math.sqrt(len(im3)),-1)) 
		im4 = im4.reshape((math.sqrt(len(im4)),-1)) 
		
		img1 = Image.fromarray(im1).convert('RGB')
		#img1.save('visualization/im1_'+ str(i) + '_' + str(j)+'.png')
		img2 = Image.fromarray(im2).convert('RGB')
		#img2.save('visualization/im2_'+ str(i) + '_' + str(j)+'.png')
		img3 = Image.fromarray(im3).convert('RGB')
		#img3.save('visualization/im3_'+ str(i) + '_' + str(j)+'.png')
		img4 = Image.fromarray(im3).convert('RGB')
		#img3.save('visualization/im3_'+ str(i) + '_' + str(j)+'.png')
		
		plt.gray()
		plt.subplot(256, 4, 3*j); pylab.axis('off'); pylab.imshow(img1)
		plt.subplot(256, 4, 3*j+1); pylab.axis('off'); pylab.imshow(img2)
		plt.subplot(256, 4, 3*j+2); pylab.axis('off'); pylab.imshow(img3)
		plt.subplot(256, 4, 3*j+3); pylab.axis('off'); pylab.imshow(img4)

	#fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
	plt.savefig('/home/amit-pc/final_figures/'+str(i)+'threshold.eps', format='eps', dpi=1000)	
	plt.clf()
"""
