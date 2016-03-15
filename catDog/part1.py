#place this file is expected to be in {caffe_root}/examples
import numpy as np
import logging
from glob import glob
import matplotlib.image as mpimg
from random import shuffle
import pickle
import matplotlib.pyplot as plt
caffe_root = '../../'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
#MODEL_FILE = '../../models/bvlc_reference_caffenet/deploy.prototxt'
#PRETRAINED = '../../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'


#Fine-tuning dog vs cat
MODEL_FILE = '../../models/finetune_cat_dog/deploy.prototxt'
PRETRAINED = '../../models/finetune_cat_dog/kutta_billi_train_500im_iter_50000.caffemodel'

#note that if this part does not work then manually download prototxt and caffemodel and remove below lines
import os
#if not os.path.isfile(PRETRAINED):
 #   print("Downloading pre-trained CaffeNet model...")
   # ../scripts/download_model_binary.py ../models/bvlc_reference_caffenet
    

def activate(net, im):
    input_image = caffe.io.load_image(im)
    # Resize the image to the standard (256, 256) and oversample net input sized crops.
    #Crop images into the four corners, center, and their mirrored versions.
    input_oversampled = caffe.io.oversample([caffe.io.resize_image(input_image, net.image_dims)], net.crop_dims)
    # 'data' is the input blob name in the model definition, so we preprocess for that input.
    caffe_input = np.asarray([net.transformer.preprocess('data', in_) for in_ in input_oversampled])
    # forward() takes keyword args for the input blobs with preprocessed input arrays.
    predicted = net.forward(data=caffe_input)
    # Activation of all convolutional layers and first fully connected
    feat = net.blobs['pool5'].data[0]
    return feat


def jpg_to_np(basedir, fetch_target):
    logging.getLogger().setLevel(logging.INFO)
    caffe.set_mode_gpu()
    print 'fetching model'
    net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))
    
    files = glob(basedir + '*jpg') 
    if fetch_target:
        print 'fetching target'
        shuffle(files)
        # Sort the files so they match the labels
        target = np.array([1. if 'cat' in f.split("/")[-1] else 0.
                           for f in files],
                          dtype='float32')
    else:
        #Must sort the files for the test sort to assure order!
        files = sorted(files,
                       key=lambda x: int(x.split("/")[-1].split(".")[-2]))
    
    feature_info = activate(net, files[0])
    feature_count = feature_info.shape
    feature_dtype = feature_info.dtype
    shape = (len(files), feature_count[0]) + (feature_count[1], feature_count[2])
    for i in range(len(files)):
	print i , files[i]
    data = np.zeros(shape, dtype=feature_dtype)
    for n, im in enumerate(files):
        data[n, :] = activate(net, im)
        if n % 1000 == 0:
            print 'Reading in image', n
    if fetch_target:
        return data, target, files
    else:
        return data, files

print 'starting program'
#please give path to folder having train images
x, y, filenames = jpg_to_np('../catDogData/body-parts/', fetch_target=True)
pickle.dump(dict(X=x,Y=y), open('Train-body-parts.pkl', 'wb'))
