import os
import sys
import pickle
import csv
import numpy as np
from PIL import Image

if len(sys.argv) < 3:
    print "usage: convert cifar.pkl output_root_folder_path output_list_path"
    exit(-1)

cifar = pickle.load(file(sys.argv[1]))
root = sys.argv[2]
try:
    os.mkdir(root)
except:
    print "%s exists, ignore"

fo = csv.writer(open(sys.argv[3], "w"), lineterminator='\n', delimiter='\t')

data = cifar['data']
label = cifar['fine_labels']
path = cifar['filenames']

sz = data.shape[0]

for i in xrange(sz):
    if i % 1000 == 0:
        print i
    img = data[i]
    p = path[i]
    img = img.reshape((3, 32, 32))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)
    im = Image.fromarray(img)
    im.save(root + p)
    row = [i, label[i], p]
    fo.writerow(row)

