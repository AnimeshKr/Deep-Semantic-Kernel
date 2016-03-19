import os
import random
import pickle
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=cpu,floatX=float32'
import numpy as np
from random import randint
from itertools import combinations
from sklearn.utils import shuffle

data=pickle.load(open("data/GPCifar10TRAIN_40000.pkl","rb"))
X1,Y1=data['X'],data['Y']
X1=X1.reshape((-1,X1.shape[1]))
#X1,Y1=X1[:2500],Y1[:2500]
############################
index=sorted(enumerate(Y1),key=lambda x:x[1])
index=np.array(index)
index=index.transpose()
index=list(index[0])
Y1=Y1[index]
X1=X1[index] ## now sorted x and y wrt y values


a=[np.where(Y1==i)[0][0] for i in range(10)]
a=np.array(a)
print a
b=a+[12]*10
for i in range(0,len(a)):
        print reduce(lambda x, y: x + y, Y1[a[i]:b[i]]) / 12.0
        if i==0:
                X_=X1[a[i]:b[i]]
                Y_=Y1[a[i]:b[i]]
        else:
                X_=np.concatenate((X_,X1[a[i]:b[i]]),axis=0)
                Y_=np.concatenate((Y_,Y1[a[i]:b[i]]),axis=0)
a=range(len(Y_))
ind=combinations(a,2)
ind=np.array(list(ind))
ind = shuffle(ind)
print "combination generation over"
X_,Y_=shuffle(X_,Y_)
flength=760
xdim=(X_.shape[1])*2
diffX_ = np.zeros(shape=(flength,xdim))
diffY_ = np.zeros(shape=(flength,))

sameX_ = np.zeros(shape=(flength,xdim))
sameY_ = np.zeros(shape=(flength,))
samei=0
diffi=0
num1=[0]*10
num2=[0]*10
for i in ind:
        a=X_[i[0]]
        b=X_[i[1]]
        x= np.concatenate([a,b])
        if Y_[i[0]] == Y_[i[1]] and samei <760:
		num1[int(Y_[i[0]])]=num1[int(Y_[i[0]])]+1
		num1[int(Y_[i[1]])]=num1[int(Y_[i[1]])]+1
                sameY_[samei]=1
                sameX_[samei]=x
                samei=samei+1
        elif Y_[i[0]] != Y_[i[1]] and diffi <760:
		num2[int(Y_[i[0]])]=num2[int(Y_[i[0]])]+1
		num2[int(Y_[i[1]])]=num2[int(Y_[i[1]])]+1
                diffY_[diffi]=0
                diffX_[diffi]=x
                diffi=diffi+1
	if diffi > 760 and samei > 760:
		break


mini=min(diffi,samei)-1
diffX_,diffY_=diffX_[:diffi-1], diffY_[:diffi-1]
sameX_,sameY_=sameX_[:samei-1], sameY_[:samei-1]
sameX_,sameY_=shuffle(sameX_,sameY_)
diffX_,diffY_=shuffle(diffX_,diffY_)
diffX_=diffX_[0:mini]
diffY_=diffY_[0:mini]
sameX_=sameX_[0:mini]
sameY_=sameY_[0:mini]
print "length from combinations" , mini

########################
for i in num1:
	print i
print "diff . "
for i in num2:
	print i

Xtotal=np.concatenate((diffX_,sameX_),axis=0)
Ytotal=np.concatenate((diffY_,sameY_),axis=0)
Xtotal,Ytotal=shuffle(Xtotal,Ytotal)
print "saving......."
data=dict(X=Xtotal,Y=Ytotal)
pickle.dump(data,open("GPCifar10TRAIN_1400from120.pkl","wb"))

data=dict(X=X_,Y=Y_)
pickle.dump(data,open("GPCifar10TRAIN_120.pkl","wb"))


