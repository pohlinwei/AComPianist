"""
The preprocessing step uses VGG16 from the research paper 
'Very Deep Convolutional Networks for Large-Scale Image Recognition' by 
Karen Simonyan, Andrew Zisserman. Find the paper here: https://arxiv.org/abs/1409.1556

The calculation of pleasure, arousal and dominance values uses the formula from the
research paper 'Effects of Color on Emotions' by Patricia Valdez, Albert Mehrabian.
Find the paper here: https://pdfs.semanticscholar.org/4711/624c0f72d8c85ea6813b8ec5e8abeedfb616.pdf
"""

# for general purpose
import os, sys
import numpy as np
import cv2
from pathlib import Path as path

# for local binary pattern
from skimage.data import load
from skimage.feature import multiblock_lbp
from skimage.transform import resize

# for vgg16
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

# image height/width
IMAGE_SIZE = 256

# settings for multilevel lbp
HEIGHT = int(IMAGE_SIZE / 3)
WIDTH = HEIGHT

""" Extracts texture features using local binary pattern.
    Returns an integer value.
"""
def calculate_lbp(img_path):
    # load img in grayscale
    img = load(img_path, as_gray=True)
    return multiblock_lbp(img, 0, 0, WIDTH, HEIGHT)

""" Returns mean value of RGB """
def mean(img_path):
    img = load(img_path)
    # flatten image to be 2D and compute mean rgb
    mean_rgb_val = mean_helper(img)
    # convert image to hsv scale
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # calculate mean
    mean_hsv_val = mean_helper(hsv)
    return mean_rgb_val, mean_hsv_val

""" Calculates mean value of a plane given a 3D matrix """
def mean_helper(org_mat):
    # "flatten" matrix to a 2D matrix
    temp = org_mat.T.reshape(3, IMAGE_SIZE * IMAGE_SIZE)
    mean_val = temp.mean(axis=1)
    return mean_val

""" Returns a 1000 * 1 matrix with probabilities of possible objects in image """
def predict_object(img_path):
    # load vgg16 model
    vgg_model = VGG16()
    img = load(img_path)
    # reshape to size 224 to fit model
    img = resize(img, (224, 224)) * 255
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_input(img)
    probabilities = vgg_model.predict(img)
    return probabilities

""" Calculates pleasure, arousal and dominance values. 
    Note 'value' in hsv is used as a measure of brightness in this case
"""
def calculate_pad(hsv):
    saturation = hsv[1]
    brightness = hsv[2] # or 'value' in hsv
    pleasure = 0.69 * brightness + 0.22 * saturation
    arousal = -0.31 * brightness + 0.6 * saturation
    dominance = -0.76 * brightness + 0.32 * saturation
    pad = np.array([pleasure, arousal, dominance])
    return pad


""" Returns all necessary features and average hue """
def preprocess(img_path):
    absolute_path = os.getcwd() + "/" + img_path
    # obtain neceassy features as a row matrix
    lbp = np.array(calculate_lbp(absolute_path))
    obj = predict_object(absolute_path).flatten()
    rgb, hsv = mean(absolute_path)
    rgb = rgb.flatten()
    # calculate pleasure, arousal and dominance values
    pad = calculate_pad(hsv).flatten()
    hsv = hsv.flatten()
    # concatenate all matrices
    features = np.concatenate((lbp, rgb, pad, hsv, obj), axis=None)
    return features, hsv[0]
