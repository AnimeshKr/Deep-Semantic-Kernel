import pickle
import numpy as np	
from sklearn.svm import SVC
from sklearn.utils import shuffle
import sys
from sklearn.metrics import accuracy_score

tr= pickle.load(open(sys.argv[1]))
te= pickle.load(open(sys.argv[2]))
Xtr,Ytr=tr['X'], tr['Y']
Xte,Yte=te['X'], te['Y']

Xtr,Xte= np.reshape(Xtr,(Xtr.shape[0],Xtr.shape[1])), np.reshape(Xte, (Xte.shape[0],Xte.shape[1]))
for i in range(1):
	Xtr,Ytr= shuffle(Xtr,Ytr)
	Xte,Yte= shuffle(Xte,Yte)

	dpoints=sys.argv[3]
	print sys.argv[3] 

	xtr,ytr=Xtr[:1000],Ytr[:1000]
	xte,yte=Xte[:1000],Yte[:1000]

	#C= [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10.0, 12.0, 15.0, 20.0, 30.0, 40.0, 50.0, 100.0, 1000.0]
	C= [0.0001, 0.005, 0.001, 0.01, 0.05, 0.1 , 0.7, 0.8, 0.9, 1.0, 100.0, 1000.0]

	for slack in C:
		svc = SVC(C=0.1, gamma=slack, kernel='rbf')
		print "starting SVM training"
		svc.fit(xtr, ytr)
		y_pred = svc.predict(xte)
		print 'For slack parameter C =',slack,'accuracy score: %0.3f' % accuracy_score(yte, y_pred), 'for',dpoints,'data points.'



