import pickle
import numpy as np
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
from sklearn.metrics import classification_report



#Dtr=pickle.load(open("svmTrainData500.pkl"))
#Dte=pickle.load(open("svmTestData500.pkl","rb"))
Dtr=pickle.load(open("svmTrainData500.pkl"))
Dte=pickle.load(open("svmTestData500.pkl","rb"))

gramTr = Dtr['gramTr']
Ytr = Dtr['YTr']

gramTe = Dte['gramTe']
Yte = Dte['YTe']

gramTr_pick, gramTe_pick = gramTr, gramTe

for i in range(len(gramTr)):
	for j in range(i,len(gramTr)):
		if Ytr[i]==Ytr[j]:
			gramTr_pick[i][j]=max(gramTr[i][j],gramTr[j][i])
			gramTr_pick[j][i]=max(gramTr[i][j],gramTr[j][i])
		else:
			gramTr_pick[i][j]=min(gramTr[i][j],gramTr[j][i])
			gramTr_pick[j][i]=min(gramTr[i][j],gramTr[j][i])

C = [0.0001, 0.0005,0.001,0.01,0.02,0.04,0.05,0.06,0.07, 0.08, 0.1, 0.2, 0.5,0.6, 0.7,0.8, 0.9, 1.0, 2.0, 5.0, 7.0, 10.0,40.0,100.0]
for slack in C:
	svc = SVC(C=slack, kernel='precomputed')
	print "starting SVM training"
	svc.fit(gramTr_pick, Ytr)

	a= gramTe[0]
	b= gramTe[0]
	for i in range(len(gramTe)):
		for j in range(len(gramTr)):
			if(Yte[j]==0):
				a[j]=max(gramTe[i][j], gramTe[j][i])
				b[j]=min(gramTe[i][j], gramTe[j][i])
			else:
				a[j]=min(gramTe[i][j], gramTe[j][i])
				b[j]=max(gramTe[i][j], gramTe[j][i])

		if svc.decision_function(a) > svc.decision_function(b):
			gramTe_pick[i]=a
		else:
			gramTe_pick[i]=b	

	y_pred = svc.predict(gramTe_pick)
	print 'Accuracy score: %0.3f' % accuracy_score(Yte, y_pred), 'for',len(gramTe),'data points.'

'''
C = [0.001,0.01,0.02,0.04,0.05,0.06,0.07, 0.08, 0.1, 0.2, 0.5,0.6, 0.7,0.8, 0.9, 1.0, 2.0, 5.0, 7.0, 10.0,40.0,100.0]
for slack in C:
	svc = SVC(C=slack, kernel='precomputed')
	print "starting SVM training"
	svc.fit(gramTr, Ytr)
	y_pred = svc.predict(gramTe)
	print 'For slack parameter C =',slack,'accuracy score: %0.3f' % accuracy_score(Yte, y_pred), 'for',n_samples,'data points.'
'''


