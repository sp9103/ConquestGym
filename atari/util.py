import numpy as np
from scipy.misc import imresize

def rgb2gray(im_raw):
    # need to check the conversion performance
    im_np = np.asarray(im_raw)
    return np.dot(im_np[...,:3], [.299, .587, .114])

def phi(ob):
    ob_ = rgb2gray(ob)
    ob_ = imresize(ob_, (110, 84), mode='F')
    i0 = 18
    i1 = i0+84
    ob_ = ob_[i0:i1,:]
    return ob_
