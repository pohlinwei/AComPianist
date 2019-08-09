""" 
This code is adapted from that provided by Afshine Amidi and Shervine Amidi
Find the original code here: 
https://github.com/afshinea/keras-data-generator/
"""

import numpy as np
import keras

# emotion categories
EMOTIONS = {'positive' : 0, 'negative' : 1}
NUM_CATEGORIES = len(EMOTIONS)

class DataGenerator(keras.utils.Sequence):
    def __init__(self, filenames, batchsize, dim, shuffle=True):
        self.dim = dim
        self.batchsize = batchsize
        self.filenames = filenames
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        # no. of batches per epoch
        return int(np.floor(len(self.filenames) / self.batchsize))
    
    def __getitem__(self, index):
        indices = self.indices[index * self.batchsize : (index + 1) * self.batchsize]
        # files at these indices
        files_temp = [self.filenames[i] for i in indices]
        X, Y = self.data_generation(files_temp)
        return X, Y
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.filenames))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def extract_label(self, file_path):
        # assume that file is stored in dataset/*/emotion/*.jpg
        emotion = file_path.split("/")[2]
        # obtain integer representation of label
        label = EMOTIONS[emotion]
        return label

    def data_generation(self, files_temp):
        X = np.empty((self.batchsize, self.dim))
        Y = np.empty((self.batchsize), dtype=int)

        # generate data
        for i, f in enumerate(files_temp):
            # load npy file
            X[i] = np.load(f)
            # label
            Y[i] = self.extract_label(f)
        return X, Y
