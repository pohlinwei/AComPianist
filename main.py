# check for errors and exceptions associated with import
try:
    import sys, os, time
    import urllib.request
    from preprocess_image import preprocess
except ImportError as import_err:
    print('Import error: ' + str(import_err))
except Exception as err:
    print('Exception detected: ' + str(err))

# store user uploaded image here temporarily 
TEMP_FILE_PATH = 'uploaded_image.jpeg'
def data_url_to_image(data_url):
    data = urllib.request.urlopen(data_url)
    with open(TEMP_FILE_PATH, 'wb') as f:
        f.write(data.file.read()) 


if __name__ == "__main__": 
    img_data_url = sys.argv[1]
    start = time.time()
    try:
        data_url_to_image(img_data_url)
        x = preprocess(TEMP_FILE_PATH)
        # !!! to-do: add in prediction
    except Exception as ex:
        print(ex) 
    end = time.time()
    print('time taken: ' + str(end - start))











