import pickle
import numpy as np
#from theano import tensor as T
#import theano
from sklearn.utils import shuffle
#from pylearn2.utils import serial
#from pylearn2.space import VectorSpace
#from pylearn2.datasets import DenseDesignMatrix
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
from sklearn.metrics import classification_report
#import numpy as np


n_samples =2000

Dte=pickle.load(open("/home/amit-pc/mxnet/example/cifar100_Model/AllNew1Nov/EXP3/from5000/probsvm/1000/svmTest60000from5000Data_1000x1000FromKernelTrain_1.pkl","rb"))
Dtr=pickle.load(open("/home/amit-pc/mxnet/example/cifar100_Model/AllNew1Nov/EXP3/from5000/probsvm/1000/svmTrain60000from5000Data_1000FromKernelTrain_1.pkl","rb"))
#Dtr=pickle.load(open("/home/amit-pc/mxnet/example/cifar100_Model/AllNew1Nov/EXP3/from5000/probsvm/2000/svmTrain60000from5000Data_2000_0.pkl","rb"))
#Dte=pickle.load(open("/home/amit-pc/mxnet/example/cifar100_Model/AllNew1Nov/EXP3/from5000/probsvm/2000/svmTest60000from5000Data_1000x2000_0.pkl","rb"))

#Dtr = pickle.load( open( "svmTrainingdataCifar100_10_1000.pkl-2" ) )
#Dte = pickle.load( open("svmTestdataCifar100_10_1000-2.pkl" ) )
print Dtr.keys()
print Dte.keys()
gramTr = Dtr['gramTr']
Ytr = Dtr['YTr']
print gramTr.shape

'''
U, s, V = np.linalg.svd(gramTr, full_matrices=True)
S = np.diag(s)
gramTr = np.dot(U, np.dot(S, V))


print U.shape
print V.shape
print s.shape
'''


gramTe = Dte['gramTe']
Yte = Dte['YTe']
print gramTe.shape
'''print gramTe.shape
eigensTr, Vtr = np.linalg.eig(np.transpose(gramTr))
for i in range(0,n_samples):
	if eigensTr[i]<0:
		eigensTr[i]=0
eigensDTr = np.diagflat(eigensTr)
#gramTr_psd = np.matrix(Vtr)*np.matrix(eigensDTr)*np.matrix(np.linalg.inv(Vtr))
gramTr_psd = np.matrix(np.linalg.inv(Vtr))*np.matrix(eigensDTr)*np.matrix(Vtr)
#gramTe_SVD= np.matrix(gramTe)*np.matrix(Vtr)
#gramTe_SVD = np.matrix(np.transpose(gramTe))*np.matrix(U)*np.matrix(np.linalg.inv(S))
#gramTe_SVD= np.matrix(np.transpose(Vtr))*np.matrix(gramTe)
#print gramTe_SVD.shape
'''
tp=np.transpose(gramTr)
gramTr_psd=gramTr*np.matrix(tp)
gramTe_psd=gramTe*np.matrix(tp)
'''
tp=np.transpose(gramTr)
gramTr_psd=tp*np.matrix(gramTr)
gramTe_psd=gramTe*np.matrix(gramTr)

pickle.dump(dict(gram=gramTr_psd, Ytr= Ytr),open('svmTrainDataPSD120.pkl','wb'))
pickle.dump(dict(gram=gramTe_psd, Yte= Yte),open('svmTestDataPSD120.pkl','wb'))
np.savetxt("svmTrainDataPSD120.csv",gramTr_psd, delimiter=",")
np.savetxt("svmTestDataPSD120.csv",gramTe_psd, delimiter=",")'''
C = [0.001,0.005,0.007,0.009,0.01,0.015, 0.02,0.04,0.05,0.06,0.07,0.08,0.09, 0.1, 0.2, 0.5,0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 5.0, 7.0, 10.0,40.0,100.0]
#C =[1]
for slack in C:
	svc = SVC(C=slack, kernel='precomputed')
	print "starting SVM training"
	svc.fit(gramTr_psd, Ytr)
	y_pred = svc.predict(gramTe_psd)
	print 'For slack parameter C =',slack,'accuracy score: %0.3f' % accuracy_score(Yte, y_pred), 'for',n_samples,'data points.'

pickle.dump(dict(gram=gramTr_psd, Y=Ytr), open('svmTrain60000from5000Data_2000_0_STS.pkl','wb'))
pickle.dump(dict(gram=gramTe_psd, Y=Yte), open('svmTest60000from5000Data_1000x2000_0.pkl','wb'))

