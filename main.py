# check for errors and exceptions associated with import
try:
    import sys, os
    import urllib.request
    from keras.models import load_model
    from preprocess_image import preprocess
except ImportError as import_err:
    print('Import error: ' + str(import_err))
except Exception as err:
    print('Exception detected: ' + str(err))

# trained model
MODEL_PATH = 'predict_emo.h5'
# store user uploaded image here temporarily 
TEMP_FILE_PATH = 'uploaded_image.jpeg'
# input dimension for neural net
DIM = 1010

def data_url_to_image(data_url):
    data = urllib.request.urlopen(data_url)
    with open(TEMP_FILE_PATH, 'wb') as f:
        f.write(data.file.read()) 

if __name__ == "__main__": 
    img_data_url = sys.argv[1]
    try:
        data_url_to_image(img_data_url)
        x, avg_hue = preprocess(TEMP_FILE_PATH)
        model = load_model(MODEL_PATH)
        y = model.predict(x.reshape(1, DIM))
        positive = (y[0] < 0.5)[0]
        print([positive,avg_hue])
    except Exception as ex:
        print(ex) 
