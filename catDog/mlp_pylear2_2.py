from pylearn2.models import mlp
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.training_algorithms import sgd, learning_rule
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets import DenseDesignMatrix
from pylearn2.train import Train
from pylearn2.train_extensions import best_params
from pylearn2.space import VectorSpace
import pylearn2
import pickle
import numpy as np

def to_one_hot(l):
    out = np.zeros((len(l), 2))
    for n, i in enumerate(l):
        out[n, i] = 1.
    return out


test = pickle.load(open('TEST5000CDpairsfrom1000.pkl', 'rb'))
X_test,Y_test=test['X'], test['Y']
Y_test = to_one_hot(Y_test)

train = pickle.load(open('TRAIN30000from1000.pkl', 'rb'))
x,y=train['X'], train['Y']
y = to_one_hot(y)

print y
print y.shape

X_train,Y_train=x[:20000],y[:20000]
X_valid,Y_valid=x[20000:25000], y[20000:25000]

in_space = VectorSpace(dim=X_train.shape[1])
full = DenseDesignMatrix(X=X_train, y=Y_train)

l1 = mlp.RectifiedLinear(layer_name='l1',
                         sparse_init=12,
                         dim=5000,
                         max_col_norm=1.)

l2 = mlp.RectifiedLinear(layer_name='l2',
                         sparse_init=12,
                         dim=5000,
                         max_col_norm=1.)

l3 = mlp.RectifiedLinear(layer_name='l3',
                         sparse_init=12,
                         dim=5000,
                         max_col_norm=1.)

output = mlp.Softmax(layer_name='y',
                     n_classes=2,
                     irange=.005)

layers = [l1, l2, l3, output]

mdl = mlp.MLP(layers,
              input_space=in_space)

lr = .05
epochs = 1000
trainer = sgd.SGD(learning_rate=lr,
                  batch_size=128,
                  learning_rule=learning_rule.Momentum(.5),
                  # Remember, default dropout is .5
                  cost=Dropout(input_include_probs={'l1': .8},
                               input_scales={'l1': 1.}),
                  termination_criterion=EpochCounter(epochs),
                  monitoring_dataset={'train': full})
'''
watcher = best_params.MonitorBasedSaveBest(
    channel_name='train_y_misclass',
    save_path='saved_clf_v3.pkl')
'''
velocity = learning_rule.MomentumAdjustor(final_momentum=.6,
                                          start=1,
                                          saturate=250)

decay = sgd.LinearDecayOverEpoch(start=1,
                                 saturate=250,
                                 decay_factor=lr*.05)

saveBest = pylearn2.train_extensions.best_params.MonitorBasedSaveBest (
             channel_name= 'test_y_misclass',
             save_path= "model_catDog30000from1000Trying.pkl"
        )

trn = DenseDesignMatrix(X=X_train, y=Y_train)
valid = DenseDesignMatrix(X=X_valid, y=Y_valid)
tst = DenseDesignMatrix(X=X_test, y=Y_test)
trainer.monitoring_dataset={'valid': valid,
                            'train': trn,
			    'test' : tst}
experiment = Train(dataset=full,
                   model=mdl,
                   algorithm=trainer,
                   extensions=[saveBest, velocity, decay])
                   

experiment.main_loop()

