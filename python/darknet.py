from ctypes import *
import math
import random
import argparse
import os
from os import listdir
from itertools import chain
import re
from os.path import isfile, join


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    return (ctype * len(values))(*values)

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

lib = CDLL("/content/gdrive/My Drive/Colab/yolo-9000/darknet/libdarknet.so", RTLD_GLOBAL)
#lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict_p
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

make_boxes = lib.make_boxes
make_boxes.argtypes = [c_void_p]
make_boxes.restype = POINTER(BOX)

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

num_boxes = lib.num_boxes
num_boxes.argtypes = [c_void_p]
num_boxes.restype = c_int

make_probs = lib.make_probs
make_probs.argtypes = [c_void_p]
make_probs.restype = POINTER(POINTER(c_float))

detect = lib.network_predict_p
detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network_p
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

network_detect = lib.network_detect
network_detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    boxes = make_boxes(net)
    probs = make_probs(net)
    num =   num_boxes(net)
    network_detect(net, im, thresh, hier_thresh, nms, boxes, probs)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if probs[j][i] > 0:
                res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_ptrs(cast(probs, POINTER(c_void_p)), num)
    return res
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--directory', help='treat all images in this directory (give the path)')
    parser.add_argument('-img','--image', help='treat the image (give the path)')
    parser.add_argument('-cfg',help = 'path to cfg file')
    parser.add_argument('-weights' ,help = 'path to weights')
    parser.add_argument('-data', help = 'path to data file')
    parser.add_argument('output', help= 'output file containing ')
    
    args = parser.parse_args()

    #print(args.directory)
    #print(args.output)
    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]
   
    net_file = args.cfg.encode('utf-8')
    weights_file =args.weights.encode('utf-8')
    meta_file = args.data.encode('utf-8')
    detects = []
    net = load_net(net_file,weights_file , 0)
    meta = load_meta(meta_file)
    output_file = args.output
    print('here output')
    print(output_file)
    if args.directory:
        dir_path = args.directory
        
        files = [ os.path.join(dir_path, f) for f in listdir(dir_path) if isfile(os.path.join(dir_path, f))]
        for f in files :
            file = f.encode('utf-8')
            r = detect(net,meta, file)
            detects.append((f, r))
    elif args.image:
        image_path = args.image.encode('utf-8')
        r = detect(net, meta, image_path)
        detects.append((image_path, r))
    
    Cnum = re.compile('b\'.+\'')
    path_reg = re.compile('(/.+)+')
 
    
    rows= []
    print(dir_path)
    i = 0
    for (image,boxes) in detects:
        print(str(len(boxes)) + ' detected')
        im = str(image[2:])
        im = im.split('/')
        im = im[-3] +'/' +im[-2]+ '/'+im[-1]
        for box in boxes:
            row = []
            
            row.append(im)
        
            for x in box : 
                if isinstance(x, tuple):
                    for y in x : 
                        row.append(str(y))
                else  :
                    row.append(str(x))
        

            row_str = ','.join(row)
            rows.append(row_str)

    with open(output_file, 'w') as f:
        for (i,row) in enumerate(rows):
            if i != 0 :
                f.write('\n')
            f.write(row)

    print('Done')
    
def exec_dir(dir_path, output_file, is_new_file):

    files = [ os.path.join(dir_path, f) for f in listdir(dir_path) if isfile(os.path.join(dir_path, f))]
    for f in files :
            file = f.encode('utf-8')
            r = detect(net,meta, file)
            detects.append((f, r))
    if is_new_file: 
        w = 'w' 
    else:
        w = 'a+'
    with open(output_file, 'w') as f:
        i = 0
        for (image,boxes) in detects:
            for box in boxes:
                row = []
                row.append(os.path.basename(str(image[2:])))
            
                for x in box : 
                    if isinstance(x, tuple):
                        for y in x : 
                            row.append(str(y))
                    else  :
                        row.append(str(x))
            
    
                row_str = ','.join(row)
                print(row_str)
                if i != 0 :
                    f.write('\n')
                f.write(row_str)
                i = i +1
            
    
    print('Done')
