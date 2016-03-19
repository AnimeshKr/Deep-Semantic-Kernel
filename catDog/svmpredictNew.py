import pickle
import numpy as np
from theano import tensor as T
import theano
from sklearn.utils import shuffle
from pylearn2.utils import serial
from pylearn2.space import VectorSpace
from pylearn2.datasets import DenseDesignMatrix
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import caffe
import time
from sklearn.metrics import classification_report

localtime = time.asctime( time.localtime(time.time()) )
print 'start time', localtime
#model_path='model_catDog50_50Data5000.pkl' # model giving 80% kernel accuracy
model_path='model_catDogData30000from1000.pkl' # model giving 80% kernel accuracy
model = serial.load(model_path)
X=model.get_input_space().make_theano_batch()
Y=model.fprop(X)
#yi=T.argmax(Y,axis=1)
#Y=Y[yi]
Y=Y[0]
f=theano.function([X],Y)

#Dtr=pickle.load(open("fcOPCifar100Train.pkl","rb")) # This has 10 features corresponding to each image
#Dte=pickle.load(open("fcNewOPCifar10Test.pkl","rb")) 
Dtr=pickle.load(open('SVMTEST_10000.pkl'))
Dte=pickle.load(open('TEST_10000.pkl'))

Xt,Yt=Dte['X'],Dte['Y']
Xr,Yr=Dtr['X'],Dtr['Y']

Xt=Xt.reshape(-1,Xt.shape[1])
Xt,Yt=shuffle(Xt,Yt)
Xt = Xt.astype('float32')

Xr=Xr.reshape(-1,Xr.shape[1])
Xr,Yr=shuffle(Xr,Yr)
Xr = Xr.astype('float32')

print "X,Y shape", Xr.shape, Yr.shape
svmXtr,svmYtr=Xr[3000:5000],Yr[3000:5000]
svmXte,svmYte=Xt[:2000],Yt[:2000]

train_len=2000
kernelTr=np.zeros(shape=(train_len,train_len))
a=np.zeros(shape=(train_len,2*svmXtr.shape[1]))
b=np.zeros(shape=(train_len,2*svmXtr.shape[1]))

localtime = time.asctime( time.localtime(time.time()))
print "building training data",localtime
for i in xrange(0,train_len):
	for j in xrange(0,train_len):
		a[j]=np.concatenate([svmXtr[i],svmXtr[j]])
		b[j]=np.concatenate([svmXtr[j],svmXtr[i]])
	sim1=f(a.astype('float32'))  # predicting the kernel output for ith image with all the other images -> (1000,)
	sim2=f(b.astype('float32'))
	kernelTr[i]=(sim1+sim2)/2.0  

#pickle.dump(kernelTr,open('kernelTrainingdataCATdog1000.pkl','wb'))

'''
data=pickle.load(open('kernelTrainingData10_1000.pkl','rb'))
kernelTr=data['X']
svmYtr=data['Y']'''
#kernelTr=pickle.load(open('kernelTrainingData.pkl','rb'))
#svc = SVC(kernel='precomputed')
#print "starting SVM training"
#svc.fit(kernelTr, svmYtr)

pickle.dump(dict(gram=kernelTr,Y=svmYtr),open('svmTrainData30000from1000_2000_2.pkl','wb'))
localtime = time.asctime( time.localtime(time.time()))

#y_pred = svc.predict(kernelTr)
#print 'accuracy score: %0.3f' % accuracy_score(svmYtr, y_pred)
#print(classification_report(svmYtr, y_pred))

print "testing data creation start",localtime

test_len=2000
kernelTe=np.zeros(shape=(test_len,test_len))

a1=np.zeros(shape=(test_len,2*svmXte.shape[1]))
b1=np.zeros(shape=(test_len,2*svmXte.shape[1]))
print "starting SVM testing"
for i in xrange(0,test_len):
	for j in xrange(0,test_len):
		a1[j]=np.concatenate([svmXte[i],svmXtr[j]])
		b1[j]=np.concatenate([svmXtr[j],svmXte[i]])
	sim1=f(a1.astype('float32'))
	sim2=f(b1.astype('float32'))
	kernelTe[i]=(sim1+sim2)/2.0

pickle.dump(dict(gram=kernelTe,Y=svmYte),open('svmTestData30000from1000_2000_2.pkl','wb'))
'''
kernelTr=pickle.load(open("kernelTrainingData10_1000.pkl","rb"))
kernelTe=pickle.load(open("kernelTestData10_1000.pkl","rb"))
'''
C = [0.001,0.01,0.1,1.0,10.0]

for slack in C:
	svc = SVC(C=slack,kernel='precomputed')
	print "starting SVM training"
	svc.fit(kernelTr, svmYtr)
	y_pred = svc.predict(kernelTe)
	print 'For slack parameter C =',slack,'accuracy score: %0.3f' % accuracy_score(svmYte, y_pred), 'for',test_len,'data points.'


'''
kernelTe=pickle.load(open('kernelTestData.pkl','rb'))
print 'test data created.
y_pred = svc.predict(kernelTe)
print 'accuracy score: %0.3f' % accuracy_score(svmYte, y_pred)
print(classification_report(svmYtr, y_pred))
'''
localtime = time.asctime( time.localtime(time.time()))
print 'end time', localtime






