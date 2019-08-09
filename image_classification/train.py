""" Neural network architecture is adapted from that found in the research paper 
'Building Emotional Machines: Recognizing Image Emotions through Deep Neural Networks' by
Hye-Rin Kim, Yeong-Seok Kim, Seon Joo Kim, In-Kwon Lee (https://arxiv.org/pdf/1705.07543.pdf)
and source code provided by Intel AI Developer Program for its 
'Hands-On AI Part 18: Emotion Recognition from Images Model Tuning and Hyperparameters'
(https://software.intel.com/en-us/articles/hands-on-ai-part-18-emotion-recognition-from-images-model-tuning-and-hyperparameters)
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.optimizers import SGD, Adam
from data_gen import DataGenerator
import os, multiprocessing
from pathlib import Path

# default feature size
FEATURES_NUM = 1010
# default batchsize
BATCHSIZE = 10
# number of epochs
NUM_EPOCHS = 100
# by default, use maximum number of cpus available
NUM_WORKERS = multiprocessing.cpu_count()
# number of emotional categories
NUM_EMOTIONS = 2
# directory which stores both training and testing data
MAIN_DIRECTORY = "dataset/"

params = {'dim' : (FEATURES_NUM), 'batchsize' : BATCHSIZE}

""" Neural network used for training """
def nn_model():
    model = Sequential()
    model.add(Dense(1000, input_shape=(FEATURES_NUM,), 
            activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', 
                optimizer=Adam(lr=0.0001),
                #optimizer=SGD(lr=0.00001, momentum=0.5), 
                metrics=['accuracy'])
    return model

""" Fetch files for train(ing)/test(ing). """
def fetch_files(purpose):
    subdirectory = MAIN_DIRECTORY + purpose 
    all_folders = [folder for folder in Path(subdirectory).iterdir() if folder.is_dir()]
    all_files = []
    for folder in all_folders:
        all_files.extend([str(x) for x in Path(folder).iterdir() if str(x).endswith('.npy')])
    return all_files

if __name__ == "__main__":
    # obtain relevant files
    training_files = fetch_files('train')
    validation_files = fetch_files('test')
    # get data generators
    training_gen = DataGenerator(training_files, **params)
    validation_gen = DataGenerator(validation_files, **params, shuffle=False)
    model = nn_model()
    model.fit_generator(generator=training_gen,
                        validation_data=validation_gen,
                        use_multiprocessing=True,
                        workers=NUM_WORKERS,
                        epochs=NUM_EPOCHS)
    model.save('../predict_emo.h5')
