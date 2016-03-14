import numpy as np
from glob import glob
import pickle
from random import randint
from sklearn.utils import shuffle

a=pickle.load(open('TRAIN_5000.pkl', 'rb'))
X1 = a['X'] 
Y1 = a['Y']

index=sorted(enumerate(Y1),key=lambda x:x[1])
index=np.array(index)
index=index.transpose()
index=list(index[0])
Y1=Y1[index]
X1=X1[index]
a=[np.where(Y1==i)[0][0] for i in range(2)]
a=np.array(a)
print a
b=a+[500]*2

#x1,y1=X1[a[0]:a[0]+num_images],Y1=[a[0]:a[0]+num_images]
#x2,y2=X1[a[1]:a[1]+num_images],Y1=[a[1]:a[1]+num_images]

for i in range(0,len(a)):
        print reduce(lambda x, y: x + y, Y1[a[i]:b[i]]) / 500.0
        if i==0:
                X_=X1[a[i]:b[i]]
                Y_=Y1[a[i]:b[i]]
        else:
                X_=np.concatenate((X_,X1[a[i]:b[i]]),axis=0)
                Y_=np.concatenate((Y_,Y1[a[i]:b[i]]),axis=0)

print "images used ", X_.shape
x,y=X_,Y_
num=5000
dataX = np.zeros((num, x.shape[1]*2), dtype='float32')
dataY= np.zeros((num),dtype='float32')
kk=0
ll=0
num1=[0]*2
num2=[0]*2
for i in xrange(0,num):
	q=randint(0,x.shape[0]-1)
	r=randint(0,x.shape[0]-1)
	a=np.array(x[q])
	b=np.array(x[r])
	d=y[q]
	e=y[r]
	dataX[i]=np.concatenate([a,b])
	
	if d==e:
		num1[int(d)]=num1[int(d)]+1
		num1[int(e)]=num1[int(e)]+1
		dataY[i]=1.
		ll=ll+1
	else:
		num2[int(d)]=num2[int(d)]+1
		num2[int(e)]=num2[int(e)]+1
		dataY[i]=0.
		kk=kk+1

print num1
print num2
print kk
print ll
print dataX.shape
print dataY.shape
print "len and sum = ",len(dataY), sum(dataY)
print dataY[1]

#for xx in dataX[1]:
#	print xx
dataX,dataY=shuffle(dataX,dataY)

pickle.dump(dict(X=dataX,Y=dataY), open('TEST5000CDpairsfrom1000.pkl', 'wb'))


	

