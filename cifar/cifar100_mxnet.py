import pickle
import mxnet as mx
import logging
import numpy as np
print 'import complete'
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

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

softmax = inception(100, 1.0)

batch_size = 30

train_dataiter = mx.io.ImageRecordIter(
    shuffle=True,
    path_imgrec="./train.rec",
    mean_img="./mean.bin",
    rand_crop=True,
    rand_mirror=True,
    data_shape=(3, 28, 28),
    batch_size=batch_size,
    prefetch_buffer=4,
    preprocess_threads=2)

test_dataiter = mx.io.ImageRecordIter(
    path_imgrec="./test.rec",
    mean_img="./mean.bin",
    rand_crop=False,
    rand_mirror=False,
    data_shape=(3, 28, 28),
    batch_size=batch_size,
    prefetch_buffer=4,
    preprocess_threads=2,
    round_batch=False)


num_round = 50
model_prefix = "cifar100"

softmax = inception(100, 1.0)
'''
model = mx.model.FeedForward(ctx=mx.gpu(), symbol=softmax, num_round=num_round,
                             learning_rate=0.05, momentum=0.9, wd=0.0001)

model = pickle.dumps(model)

model.fit(X=train_dataiter,
          eval_data=test_dataiter,
          eval_metric="accuracy",
          epoch_end_callback=mx.callback.Speedometer(batch_size, 100),
          iter_end_callback=mx.callback.do_checkpoint(model_prefix))
'''
#model.save(model_prefix)


# load params from saved model
num_round = 14
model_prefix = "cifar100_stage3"
#model.save(model_prefix)

tmp_model = mx.model.FeedForward.load(model_prefix, num_round)

print "create new model with params"

num_round =1
model_prefix = "cifar100_stage4"
model = mx.model.FeedForward(ctx=mx.gpu(), symbol=softmax, num_round=num_round,
                             learning_rate=0.001, momentum=0.9, wd=0.001,
                             arg_params=tmp_model.arg_params, aux_params=tmp_model.aux_params)

 
model.fit(X=train_dataiter,
          eval_data=test_dataiter,
          eval_metric="accuracy",
          epoch_end_callback=mx.callback.Speedometer(batch_size, 100),
          iter_end_callback=mx.callback.do_checkpoint(model_prefix))

print "Done!"

'''
#model.save(model_prefix)

####################################################

print test_dataiter
#feature extraction
model_prefix="cifar100_stage2"

model3 = mx.model.FeedForward.load(model_prefix, num_round, ctx=mx.gpu())

prob = model3.predict(test_dataiter)
logging.info('Finish predict...')
# Check the accuracy from prediction
#### added code ###
print "printing labels before reset"
for i, label in test_dataiter:
    label = label.asnumpy()
    print label
    if i>40:
        break;


test_dataiter.reset()

print "printing labels after reset"
for _, label in test_dataiter:
    label = label.asnumpy()
    print label
    if i>40:
        break
 
# get label
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

print "predicted value is"
print py[0:40]

acc1 = float(np.sum(py == y)) / len(y)
#here y is orignal and py is predicted
logging.info('final accuracy = %f', acc1)
print acc1
# Predict internal featuremaps
# From a symbol, we are able to get all internals. Note it is still a symbol
internals = softmax.get_internals()
# We get get an internal symbol for the feature.
# By default, the symbol is named as "symbol_name + _output"
# in this case we'd like to get global_avg" layer's output as feature, so its "global_avg_output"
# You may call ```internals.list_outputs()``` to find the target
# but we strongly suggests set a special name for special symbol 
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
global_pooling_feature = feature_extractor.predict(test10_dataiter)
print(global_pooling_feature.shape)
'''

