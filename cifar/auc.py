import matplotlib.pyplot as plt
from theano import tensor as T
import theano
from sklearn.utils import shuffle
from pylearn2.utils import serial
import pickle
import numpy as np
from sklearn import metrics
def make_Y( x, y ):
	mat = np.zeros( (len(x), len(y)))
	for i in range( len(x) ):
		for j in range( len(y) ):
			if x[i] == y[j] :
				mat[i][j] = 2.
			else: 
				mat[i][j] = 1.
	return mat

def get_auc_score( y, pred, gtype = 'train' ):
	fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
	score = metrics.auc(fpr, tpr)
	plt.plot(fpr, tpr, label='ROC curve %s (area = %0.3f)' % (gtype, score) )
	return score 

f =pickle.load(open('data/GPCifar10TRAIN_60000from1000.pkl'))
x,y= f['X'],f['Y']
xt,yt= x[55000:], y[55000:]

model_path="modelGPCifar10_10test60000from1000.pkl" # model giving 80% kernel accuracy
model = serial.load(model_path)
X=model.get_input_space().make_theano_batch()
Y=model.fprop(X)
f=theano.function([X],Y)


print 'starting feedforward'
#y_pred = np.zeros( len(yt) )
'''
for i in range( len(yt) ):
	y_pred[i] = f(xt[i].astype('float32'))
'''

y_pred = f(xt.astype('float32'))
y_pred= y_pred[:,1]

print y_pred 
print yt
#plt.figure()
print metrics.roc_auc_score(yt, y_pred)
"""
trainfile = 'from5000/probsvm/2000/svmTrain60000from5000Data_2000_0.pkl'
testfile = 'from5000/probsvm/2000/svmTest60000from5000Data_1000x2000_0.pkl'

plt.figure()
data = pickle.load( open( trainfile ) )
print data.keys()
matrix = data[ 'gramTr' ]
trlabels = data[ 'YTr' ]

y_pred = matrix.flatten()
y = make_Y( trlabels, trlabels )
print get_auc_score( y.flatten(), y_pred, gtype = 'train' )


data = pickle.load( open( testfile ) )
print data.keys()
matrix = data[ 'gramTe' ]
telabels = data[ 'YTe' ]
y_pred = matrix.flatten()
y = make_Y( telabels, trlabels )
print y
print matrix
print get_auc_score( y.flatten(), y_pred, gtype = 'test' )

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('')
plt.legend(loc="lower right")
plt.show()

"""
