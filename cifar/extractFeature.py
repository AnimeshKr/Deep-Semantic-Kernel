import pickle
import mxnet as mx
import logging
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

########## Starting Definitions ################

def ConvFactory(data, num_filter, kernel, stride=(1,1), pad=(0, 0), name=None, suffix=''):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, name='conv_%s%s' %(name, suffix))
    bn = mx.symbol.BatchNorm(data=conv, name='bn_%s%s' %(name, suffix))
    act = mx.symbol.LeakyReLU(data=bn, act_type='rrelu', name='rrelu_%s%s' %(name, suffix))
    return act

def InceptionFactoryA(data, num_1x1, num_3x3red, num_3x3, num_d3x3red, num_d3x3, pool, proj, name):
    # 1x1
    c1x1 = ConvFactory(data=data, num_filter=num_1x1, kernel=(1, 1), name=('%s_1x1' % name))
    # 3x3 reduce + 3x3
    c3x3r = ConvFactory(data=data, num_filter=num_3x3red, kernel=(1, 1), name=('%s_3x3' % name), suffix='_reduce')
    c3x3 = ConvFactory(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), pad=(1, 1), name=('%s_3x3' % name))
    # double 3x3 reduce + double 3x3
    cd3x3r = ConvFactory(data=data, num_filter=num_d3x3red, kernel=(1, 1), name=('%s_double_3x3' % name), suffix='_reduce')
    cd3x3 = ConvFactory(data=cd3x3r, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), name=('%s_double_3x3_0' % name))
    cd3x3 = ConvFactory(data=cd3x3, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), name=('%s_double_3x3_1' % name))
    # pool + proj
    pooling = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, name=('%s_pool_%s_pool' % (pool, name)))
    cproj = ConvFactory(data=pooling, num_filter=proj, kernel=(1, 1), name=('%s_proj' %  name))
    # concat
    concat = mx.symbol.Concat(*[c1x1, c3x3, cd3x3, cproj], name='ch_concat_%s_chconcat' % name)
    return concat

def InceptionFactoryB(data, num_3x3red, num_3x3, num_d3x3red, num_d3x3, name):
    # 3x3 reduce + 3x3
    c3x3r = ConvFactory(data=data, num_filter=num_3x3red, kernel=(1, 1), name=('%s_3x3' % name), suffix='_reduce')
    c3x3 = ConvFactory(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_3x3' % name))
    # double 3x3 reduce + double 3x3
    cd3x3r = ConvFactory(data=data, num_filter=num_d3x3red, kernel=(1, 1),  name=('%s_double_3x3' % name), suffix='_reduce')
    cd3x3 = ConvFactory(data=cd3x3r, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name=('%s_double_3x3_0' % name))
    cd3x3 = ConvFactory(data=cd3x3, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_double_3x3_1' % name))
    # pool + proj
    pooling = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pool_type="max", name=('max_pool_%s_pool' % name))
    # concat
    concat = mx.symbol.Concat(*[c3x3, cd3x3, pooling], name='ch_concat_%s_chconcat' % name)
    return concat
    
    
def inception(nhidden, grad_scale):
    # data
    data = mx.symbol.Variable(name="data")
    # stage 2
    in3a = InceptionFactoryA(data, 64, 64, 64, 64, 96, "avg", 32, '3a')
    in3b = InceptionFactoryA(in3a, 64, 64, 96, 64, 96, "avg", 64, '3b')
    in3c = InceptionFactoryB(in3b, 128, 160, 64, 96, '3c')
    # stage 3
    in4a = InceptionFactoryA(in3c, 224, 64, 96, 96, 128, "avg", 128, '4a')
    in4b = InceptionFactoryA(in4a, 192, 96, 128, 96, 128, "avg", 128, '4b')
    in4c = InceptionFactoryA(in4b, 160, 128, 160, 128, 160, "avg", 128, '4c')
    in4d = InceptionFactoryA(in4c, 96, 128, 192, 160, 192, "avg", 128, '4d')
    in4e = InceptionFactoryB(in4d, 128, 192, 192, 256, '4e')
    # stage 4
    in5a = InceptionFactoryA(in4e, 352, 192, 320, 160, 224, "avg", 128, '5a')
    in5b = InceptionFactoryA(in5a, 352, 192, 320, 192, 224, "max", 128, '5b')
    # global avg pooling
    avg = mx.symbol.Pooling(data=in5b, kernel=(7, 7), stride=(1, 1), name="global_pool", pool_type='avg')
    # linear classifier
    flatten = mx.symbol.Flatten(data=avg, name='flatten')
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=nhidden, name='fc')
    softmax = mx.symbol.Softmax(data=fc1, name='softmax')
    return softmax

##############  Definitions over ####################

softmax = inception(100, 1.0)
num_round=12
batch_size = 20

train_dataiter = mx.io.ImageRecordIter(
    shuffle=True,
    path_imgrec="../train.rec",
    mean_img="../mean.bin",
    rand_crop=True,
    rand_mirror=True,
    data_shape=(3, 28, 28),
    batch_size=batch_size,
    prefetch_buffer=4,
    preprocess_threads=2)

test_dataiter = mx.io.ImageRecordIter(
    path_imgrec="../test.rec",
    mean_img="../mean.bin",
    rand_crop=False,
    rand_mirror=False,
    data_shape=(3, 28, 28),
    batch_size=batch_size,
    prefetch_buffer=4,
    preprocess_threads=2,
    round_batch=False)

test10_dataiter = mx.io.ImageRecordIter(
    path_imgrec="../test10.rec",
    mean_img="../../../data/cifar/cifar_mean.bin",
    rand_crop=False,
    rand_mirror=False,
    data_shape=(3, 28, 28),
    batch_size=batch_size,
    prefetch_buffer=4,
    preprocess_threads=2,
    round_batch=False)

train10_dataiter = mx.io.ImageRecordIter(
    path_imgrec="../train10.rec",
    mean_img="../../../data/cifar/cifar_mean.bin",
    rand_crop=False,
    rand_mirror=False,
    data_shape=(3, 28, 28),
    batch_size=batch_size,
    prefetch_buffer=4,
    preprocess_threads=2,
    round_batch=False)



model_prefix = "cifar100_stage3"

model3 = mx.model.FeedForward.load(model_prefix, num_round, ctx=mx.gpu())

prob = model3.predict(test_dataiter)
logging.info('Finish predict...')

# Because the iterator pad each batch same shape, we want to remove paded samples here
test_dataiter.reset()
y_batch = []
for _, label in test_dataiter:
    label = label.asnumpy()
    pad = test_dataiter.getpad()
    real_size = label.shape[0] - pad
    y_batch.append(label[0:real_size])
y = np.concatenate(y_batch)

# get prediction label from 
py = np.argmax(prob, axis=1)

#print "predicted value is"
#print py[0:40]

acc1 = float(np.sum(py == y)) / len(y)
#here y is orignal and py is predicted
logging.info('final accuracy = %f', acc1)
#print acc1

internals = softmax.get_internals()

fea_symbol = internals["global_pool_output"]
# here we are extracting global_pool wecan  also use fc1_output or flatten_output

# Make a new model by using an internal symbol. We can reuse all parameters from model we trained before
# In this case, we must set ```allow_extra_params``` to True 
# Because we don't need params of FullyConnected Layer

feature_extractor = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, 
                                         arg_params=model3.arg_params,
                                         aux_params=model3.aux_params,
                                         allow_extra_params=True)
# Predict as normal
print "staring feature extraction"

global_pooling_feature = feature_extractor.predict(test10_dataiter)
print(global_pooling_feature.shape)
test10_dataiter.reset()
y_batch = []
for _, label in test10_dataiter:
    label = label.asnumpy()
    pad = test10_dataiter.getpad()
    real_size = label.shape[0] - pad
    y_batch.append(label[0:real_size])
Y10te = np.concatenate(y_batch)
data=dict(X=global_pooling_feature,Y=Y10te)
#X10te=global_pooling_feature
pickle.dump(data, open('GPCifar10Test.pkl','wb'))

global_pooling_feature = feature_extractor.predict(train_dataiter)
print(global_pooling_feature.shape)
train_dataiter.reset()
y_batch = []
for _, label in train_dataiter:
    label = label.asnumpy()
    pad = train_dataiter.getpad()
    real_size = label.shape[0] - pad
    y_batch.append(label[0:real_size])
Y100tr = np.concatenate(y_batch)
#X100tr=global_pooling_feature
data=dict(X=global_pooling_feature,Y=Y100tr)
pickle.dump(data, open('GPCifar100Train.pkl','wb'))


global_pooling_feature = feature_extractor.predict(train10_dataiter)
print(global_pooling_feature.shape)
train10_dataiter.reset()
y_batch = []
for _, label in train10_dataiter:
    label = label.asnumpy()
    pad = train10_dataiter.getpad()
    real_size = label.shape[0] - pad
    y_batch.append(label[0:real_size])
Y10tr = np.concatenate(y_batch)
#X10tr=global_pooling_feature
data=dict(X=global_pooling_feature,Y=Y10tr)
pickle.dump(data, open('GPCifar10Train.pkl','wb'))

print "feature extraction Two done Two more !"
global_pooling_feature = feature_extractor.predict(test_dataiter)
print(global_pooling_feature.shape)
test_dataiter.reset()
y_batch = []
for _, label in test_dataiter:
    label = label.asnumpy()
    pad = test_dataiter.getpad()
    real_size = label.shape[0] - pad
    y_batch.append(label[0:real_size])
Y100te = np.concatenate(y_batch)
#X100te=global_pooling_feature
data=dict(X=global_pooling_feature,Y=Y100te)
pickle.dump(data, open('GPCifar100Test.pkl','wb'))
'''
X10=np.concatenate((X10tr,X10te),axis=0)
print X10.shape
Y10=np.concatenate((Y10tr,Y10te),axis=0)
print Y10.shape
X100=np.concatenate((X100tr,X100te),axis=0)
print X100.shape
Y100=np.concatenate((Y100tr,Y100te),axis=0)
print Y100.shape
data10=dict(X=X10,Y=Y10)
pickle.dump(data10, open('globlPoolNewOPCifar10.pkl','wb'))
data100=dict(X=X100,Y=Y100)
pickle.dump(data100, open('globlPoolNewOPCifar100.pkl','wb'))'''
