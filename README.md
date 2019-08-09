# AComPianist
AComPianist is a computer pianist that plays an accompanying piece when an image is uploaded.

How Does It Work? 
-----------------
![Infographic](/images/infographic.png)

Instructions
------------
### Using Pretrained Weights for Web App
Download the above files. Then, install [Node.js and npm](https://nodejs.org/en/download/) and [pip](https://pip.pypa.io/en/stable/installing/) if they have yet to be installed. 

If `virutalenv` has not been installed,
```bash
pip install virtualenv
```

Go to the directory where the files are installed.

Run the following commands to download the required modules and packages, and to view the web app.

```bash
npm install && npm run first-build
```
In the future, to view the web app activate the virtual environment if it hasn't been done using

```bash
. venv/bin/activate
```

Then, to view the web app run the command

```bash
npm run view
```

For those who are interested to modify any `.ts` files, run the following command to view the changes made.

```bash
npm run build-and-view
```
### Re-training Neural Network for Emotion Classification of Images (Optional) 

If the virtual environment has not already been activated,

```bash 
. venv/bin/activate
```

Then, 

```bash
cd image_classification
```

Download the required [dataset](http://www.cs.rochester.edu/u/qyou/deepemotion/) and place it in `image_classification`.

To download and preprocess the data,

```bash
python3 download_data.py
```

Finally, to train the neural network, run the following command

```bash
python3 train.py
```

The newly-trained model is saved at the location `../predict_emo.h5`.

Note: If the downloading of data is done on Mac with multi-core, you might need to do the following to allow files to be downloaded in parallel.

Go to your bash profile and add the following line to the end

```bash
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
```

Close the terminal and re-open the terminal.


The Details
-----------
### Emotion Classification
The uploaded image is pre-processed and its features are extracted. Thereafter, a prediction on whether the image evokes positive or negative feelings is made using a neural network. The architecture of the neural network is largely adapted from that proposed in the research paper ['Building Emotional Machines: Recognising Image Emotions through Deep Neural Networks'](https://arxiv.org/pdf/1705.07543.pdf) by Hye-Rin Kim, Yeong-Seok Kim, Seon Joo Kim, In-Kwon Lee and that suggested in ['Hands-On AI Part 18: Emotion Recognition from Images Model Tuning and Hyperparameters'](https://software.intel.com/en-us/articles/hands-on-ai-part-18-emotion-recognition-from-images-model-tuning-and-hyperparameters) from Intel AI Developer Program. The neural network is trained using the dataset published in the research paper titled ['Building a Large Scale Dataset for Image Emotion Recognition: The Fine Print and The Benchmark'](https://arxiv.org/pdf/1605.02677.pdf) by Quanzeng You, Jiebo Luo, Hailin Jin and Jianchao Yang. 

### Music Generation
Music generation is made possible by using [Performance RNN](https://magenta.tensorflow.org/performance-rnn). In fact, the crucial code for AComPianist's music generation ability is a modification of that found in the [demo code of Performance RNN](https://github.com/tensorflow/magenta-demos/tree/master/performance_rnn). 

### The Link
The prediction made by the emotion classification neural network determines the tonality of the music. In particular, if the neural network predicts that the image evokes positive feelings, the tonality will be major (which often induces positive feelings in the listener); if otherwise, it will be minor (which often induces negative feelings). In addition, the image's average hue is calculated. As one moves the mouse over the uploaded image, the difference between the average hue and the hue of the pixel which the cursor is currently pointing to will be determined. By leveraging on the similarity between the colour wheel and the circle of fifths, as shown below, we can relate a difference in visual experience to an auditory one. 

![Colour wheel and circle of fifths](/images/circles.png)

Image source: [Colour wheel](https://pixabay.com/vectors/rainbow-colors-circle-color-spectrum-154569/) and [circle of fifths](https://www.flickr.com/photos/ethanhein/6190222353)

The brightness of the pixel, which the cursor is currently pointing to, determines the note density: the brighter the pixel, the lower the note density and vice versa. Since a lower note density makes the piece sound less complex and a lighter colour often represents pureness, this feature can strengthen the connection between the visual experience and the auditory one.  

Lastly, the rate at which the cursor moves influences the loudness of the generated music. The cursor movements can represent the hand movements of a videographer as he/she films a video (which consists of many images). Since rapid and abrupt movements while one films a video, can result in more noise, we would want the generated music to be louder, in hopes that it would drown out any noise captured due to the abrupt movements.


References
----------
1. Afshine Amidi and Shervine Amidi (2018) Keras Data Generator [Source Code]. https://github.com/afshinea/keras-data-generator/
2. Quanzeng You, Jiebo Luo, Hailin Jin and Jianchao Yang. (2016) Building a Large Scale Dataset for Image Emotion Recognition: The Fine Print and The Benchmark [Research Paper]. https://arxiv.org/pdf/1605.02677.pdf
3. Karen Simonyan, Andrew Zisserman (2015) Very Deep Convolutional Networks for Large-Scale Image Recognition [Research Paper]. https://arxiv.org/abs/1409.1556
4. Patricia Valdez and Albert Mehrabian (1994) Effects of Color on Emotions [Research Paper]. https://pdfs.semanticscholar.org/4711/624c0f72d8c85ea6813b8ec5e8abeedfb616.pdf
5. Intel AI Developer Program (2017) Hands-On AI Part 18: Emotion Recognition from Images Model Tuning and Hyperparameters [Source Code]. https://software.intel.com/en-us/articles/hands-on-ai-part-18-emotion-recognition-from-images-model-tuning-and-hyperparameters
6. Hye-Rin Kim, Yeong-Seok Kim, Seon Joo Kim, In-Kwon Lee (2017) Building Emotional Machines: Recognizing Image Emotions through Deep Neural Networks [Research Paper] https://arxiv.org/pdf/1705.07543.pdf
7. Ian Simon and Sageev Oore (2017) Performance RNN: Generating Music with Expressive Timing and Dynamics [Research Paper] https://magenta.tensorflow.org/performance-rnn
