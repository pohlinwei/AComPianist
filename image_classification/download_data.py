""" Script uses images specified in 
    "Building a Large Scale Dataset for Image Emotion Recognition: The Fine Print and The Benchmark", 
    the Thirtieth AAAI Conference on Artificial Intelligence (AAAI), 2016, 
    by Quanzeng You, Jiebo Luo, Hailin Jin and Jianchao Yang.
    The dataset is found in 'agg'.
    Note: This script only downloads the required image temporarily for preprocessing.
    It saves the preprocess data instead. 
"""

""" To separate test and training set, pass trainRatio and testRatio as arguments in the 
    command line.
    This segments training and test into #trainingCSVFile : #testCSVFile approximately.
"""
import os, sys, zipfile, time, random, shutil, multiprocessing
from pathlib import Path as path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from skimage import io, transform
from preprocess import preprocess
import keras

# use max no. of workers
MAX_WORKERS = multiprocessing.cpu_count()

# image height/width
IMAGE_SIZE = 256

# used for parallel processing of data
CHUNKSIZE = 3
# emotion categories
EMOTIONS = ['amusement', 'awe', 'anger', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']
# emotion categories
POSITIVE_EMOTIONS = {'amusement', 'awe', 'contentment', 'excitement'}
NEGATIVE_EMOTIONS = {'fear', 'sadness', 'anger', 'disgust'}
# categories required for binary neural network 
CAT = ['positive', 'negative']
# number of emotions in each CAT
NUM_PER_CAT = 4
# number of csv files for each emotion (as found in 'agg')
FILES_PER_EMO = 7

# required weights for neural net used in preprocessing step
WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5')

""" Resize image, label it and save it """
def create_img_file(directory, line_from_csv):
    # note that line_from_csv is formatted as emotion, link, # of disagree, # of agree
    # remove line break
    line_from_csv = line_from_csv.split()
    # separate information
    emo, link, disgree_num, agree_num = line_from_csv[0].split(',')
    # convert string to integer
    disagree_num = int(disgree_num)
    agree_num = int(agree_num)

    # determine whether to keep img (i.e. whether respondents think that img evokes emo)
    if agree_num < disagree_num:
        print("Image ignored")
        return # ignore image since result is not conclusive
    print("Processing image")
    # obtain id of image
    temp_link = link.split("/")
    id = temp_link[len(temp_link) - 1]

    # determine category
    cat = CAT[0] if {emo}.issubset(POSITIVE_EMOTIONS) else CAT[1]
    # create img file
    file_name = directory + cat + "/" + id
    img_arr = io.imread(link)
    # resize img before saving 
    img_arr = transform.resize(img_arr, (IMAGE_SIZE, IMAGE_SIZE)) * 255
    io.imsave(file_name, img_arr.astype("uint8"))

    # preprocess image to obtain features
    features = preprocess(file_name)
    # delete image file
    os.remove(file_name)

    # remove .jpg extension
    file_name = file_name.split(".")[0]
    # save feautures as .npy file
    np.save(file_name, features)
    print("Done processing")
    return

""" Wrapper function for above function. 
    Used when creating images for training
"""
def create_img_train(line_from_csv):
    create_img_file("dataset/train/", line_from_csv)
""" Wrapper function for above function. 
    Used when creating images for testing
"""
def create_img_test(line_from_csv):
    create_img_file("dataset/test/", line_from_csv)

""" Process and download relevant data 
    purpose specifies whether data is used for "train" or "test"
    The default value is "train"
    directory specifies where data is stored after downloading
    The default directory is "dataset"
    Note that data can be found in the path directory/purpose
"""
def process_all(directory="dataset/", purpose="train"):
    # create required sub-directories, if they don't already exist
    for category in CAT:
        subdirectory = directory + purpose + "/" + category 
        if not path(subdirectory).is_dir():
            os.makedirs(subdirectory)
    
    # directory containing all csv files for purpose
    parent_dir = path('agg/' + purpose)
    # all csv files which contain labelled data
    all_csv_files = [x for x in parent_dir.iterdir() if str(x).endswith('.csv')]

    for file in all_csv_files:
        print("Processing for " + str(file) + " file starts: ")
        with open(file) as f:
            all_lines = f.read().split()
            start = time.time()
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:  
                # process each line in parallel
                if (purpose == "train"): 
                    executor.map(create_img_train, all_lines, chunksize=CHUNKSIZE) 
                else:
                    executor.map(create_img_test, all_lines, chunksize=CHUNKSIZE)       
            end = time.time()
            print("Time taken for " + str(file) + " file: " + str(end - start))
    return

""" Segment training and test set such that #trainingCSVFile : #testCSVFile is approximately
    trainRatio : testRatio
    If ratio is not specified, all dataset will be used for training.
    There will, by default, be at least one trainingCSVFile.
    Note that trainRatio : testRatio != #trainImg : #testImg because not all images in the dataset
    evoke the labelled emotion, i.e. when #agree < #disagree 
    (Refer to http://www.cs.rochester.edu/u/qyou/deepemotion/ for more info)
"""
def segment_data(trainRatio, testRatio):
    trainFrac = trainRatio / (trainRatio + testRatio)
    trainFileNum = trainFrac * FILES_PER_EMO if trainFrac * FILES_PER_EMO > 1 else 1
    all_csv_files = [x for x in path('agg').iterdir() if str(x).endswith('.csv')] 
    # shuffle files
    random.shuffle(all_csv_files)

    # create subdirectory to categorise training and testing datasets
    if not path('agg/train').is_dir():
        os.makedirs('agg/train')
    if not path('agg/test').is_dir():
        os.makedirs('agg/test')
    # dictionary to track remaining number of training files required 
    remainingTrainingFiles = {emotion : trainFileNum for emotion in EMOTIONS}
    # separate test files from training files
    for file in all_csv_files:
        filePath = str(file)
        # determine the predominant emotion images from this file evoke
        emotionGroup = filePath.split("/")[1].split("_")[0]
        if remainingTrainingFiles[emotionGroup] > 0:
            # use file for testing set
            shutil.copy(filePath, 'agg/train')
            # update remaining num of training files required for emotionGroup
            remainingTrainingFiles[emotionGroup] -= 1
        else:
            # use file for testing set
            shutil.copy(filePath, 'agg/test')

    # create directory to store all images
    if not path('dataset').is_dir():
        os.mkdir('dataset')
    # create all training data
    print("Creating data set for training")
    process_all(purpose="train")
    print("Creation of training set is done")
    # create all testing data
    """
    print("Creating data set for testing")
    process_all(purpose="test")
    print("Creation of testing set is done")"""
    return

class InvalidArgumentLenException(Exception):
    pass


""" Exception raised when 'agg' directory is not found """
class NoDataSetFoundException(Exception):
    pass

def main(trainRatio=1, testRatio=0):
    # check if agg folder exists in current directory
    try:
        if path('agg').is_dir():
            # download weights for VGG16, if it hasn't been done
            keras.utils.get_file(
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='64373286793e3c8b2b4e3219cbf3544b')
            # segment data set
            segment_data(trainRatio, testRatio)
        elif path('agg.zip').is_file():
            # download weights for VGG16, if it hasn't been done
            keras.utils.get_file(
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='64373286793e3c8b2b4e3219cbf3544b')
            # unzip file if it has yet to be done
            agg_file = zipfile.ZipFile(path('agg.zip'), 'r')
            # create subdirectory to store all files within zip
            os.mkdir('agg')
            agg_file.extractall(path('agg'))
            agg_file.close()
            # segment data set
            segment_data(trainRatio, testRatio)
        else:
            raise NoDataSetFoundException     
    except NoDataSetFoundException:
        print("Required data set cannot be found.")
        print("Download 'agg.zip' here: http://www.cs.rochester.edu/u/qyou/deepemotion/")

if __name__ == "__main__":
    try: 
        if len(sys.argv) == 1:
            main()
        elif len(sys.argv) == 3:
            trainRatio = int(sys.argv[1])
            testRatio = int(sys.argv[2])
            main(trainRatio, testRatio)
        else:
            raise InvalidArgumentLenException
    except InvalidArgumentLenException:
        print("Either input trainRatio and testRatio in command line ")
        print("or input no extra arguments in command line")
