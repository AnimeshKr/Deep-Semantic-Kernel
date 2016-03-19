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
#model_path='modelGPCifar10_10test60000from2000.pkl' # model giving 80% kernel accuracy
model_path="modelGPCifar10_10test40000from120_3.pkl" # model giving 80% kernel accuracy
model = serial.load(model_path)
X=model.get_input_space().make_theano_batch()
Y=model.fprop(X)
f=theano.function([X],Y)

Dte=pickle.load(open('data/GPCifar10TESTSVM_10000.pkl'))
Dtr=pickle.load(open('GPCifar10TRAIN_120.pkl'))

Xr,Yr=Dtr['X'],Dtr['Y']
Xt,Yt=Dte['X'],Dte['Y']

Xr=Xr.reshape(-1,Xr.shape[1])
Xt=Xt.reshape(-1,Xt.shape[1])
Xr = Xr.astype('float32')
Xt = Xt.astype('float32')

#Xt=Xt.reshape(-1,Xt.shape[1])
#Xt = Xt.astype('float32')
for index1 in range(1):
	Xr,Yr=shuffle(Xr,Yr)
	#Xt,Yt=shuffle(Xt,Yt)
	print "X,Y shape", Xr.shape, Yr.shape
	svmXtr,svmYtr=Xr,Yr
	#svmXtr,svmYtr=Xr[3900:4000],Yr[3900:4000]
	svmXte,svmYte=Xt[:1000],Yt[:1000]

	train_len=120
	kernelTr=np.zeros(shape=(train_len,train_len))
	a=np.zeros(shape=(train_len,2*svmXtr.shape[1]))
	b=np.zeros(shape=(train_len,2*svmXtr.shape[1]))

	localtime = time.asctime( time.localtime(time.time()))
	print "building training data",localtime
	for i in xrange(0,train_len):
		for j in xrange(0,train_len):
			a[j]=np.concatenate([svmXtr[i],svmXtr[j]])
			b[j]=np.concatenate([svmXtr[j],svmXtr[i]])
		sim1=f(a.astype('float32'))  
		sim2=f(b.astype('float32'))
		s1=sim1[:,1]
		s2=sim2[:,1]
		kernelTr[i]=(s1+s2)/2.0


	pickle.dump(dict(gramTr=kernelTr,YTr=svmYtr,svmXtr=svmXtr),open('svmTrainData120_2.pkl','wb'))
	localtime = time.asctime( time.localtime(time.time()))


	print "testing data creation start",localtime

	test_len=1000
	kernelTe=np.zeros(shape=(test_len,train_len))

	a1=np.zeros(shape=(train_len,2*svmXte.shape[1]))
	b1=np.zeros(shape=(train_len,2*svmXte.shape[1]))
	print "starting SVM testing"
	for i in xrange(0,test_len):
		for j in xrange(0,train_len):
			a1[j]=np.concatenate([svmXte[i],svmXtr[j]])
			b1[j]=np.concatenate([svmXtr[j],svmXte[i]])
		sim1=f(a1.astype('float32'))
		sim2=f(b1.astype('float32'))
		s1=sim1[:,1]
		s2=sim2[:,1]
		kernelTe[i]=(s1+s2)/2.0


#	pickle.dump(dict(gramTe=kernelTe,YTe=svmYte),open('probsvm/'+str(train_len)+'/svmTest60000from5000Data_'+str(test_len)+'x'+str(train_len)+'_'+str(index1)+'.pkl','wb'))
        pickle.dump(dict(gramTe=kernelTe,YTe=svmYte,svmXte=svmXte),open('svmTestData1000x120_2.pkl','wb'))

	C = [0.001,0.01,0.02,0.04,0.08,0.1, 0.2, 0.5, 0.7, 1.0, 2.0, 5.0, 7.0, 10.0,40.0,100.0]
	#f1=open('probsvm/'+str(train_len)+'/svmTest60000from5000Data_'+str(test_len)+'x'+str(train_len)+'_'+str(index1)+'.txt','w')
	maxi=0
	maxc=0
	for slack in C:
		svc = SVC(C=slack,kernel='precomputed')
		print "starting SVM training"
		svc.fit(kernelTr, svmYtr)
		y_pred = svc.predict(kernelTe)
		acc=accuracy_score(svmYte, y_pred)
		if maxi < acc : 
			maxi=acc
			maxc=slack 
		print 'For slack parameter C =',slack,'accuracy score: %0.3f' % acc, 'for',test_len,'data points.'
		#f1.write( 'For slack parameter C =' + str(slack) + 'accuracy score: ' + str(acc) + 'for' + str(test_len) + 'data points\n' )

	print "maximum is : ", maxi, maxc
localtime = time.asctime( time.localtime(time.time()))
print 'end time', localtime






