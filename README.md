# AComPianist
AComPianist is a computer pianist that plays an accompanying piece when an image is uploaded.

How Does It Work? 
-----------------
![Infographic](/infographic.png)

Instructions
------------

The Details
-----------
### Emotion Classification
The uploaded image is pre-processed and its features are extracted. Thereafter, a prediction on whether the image evokes positive or negative feelings is made using a neural network. The architecture of the neural network is largely adapted from that proposed in the research paper ['Building Emotional Machines: Recognising Image Emotions through Deep Neural Networks'](https://arxiv.org/pdf/1705.07543.pdf) by Hye-Rin Kim, Yeong-Seok Kim, Seon Joo Kim, In-Kwon Lee and that suggested in ['Hands-On AI Part 18: Emotion Recognition from Images Model Tuning and Hyperparameters'](https://software.intel.com/en-us/articles/hands-on-ai-part-18-emotion-recognition-from-images-model-tuning-and-hyperparameters) from Intel AI Developer Program. The neural network is trained using the dataset published in the research paper titled 'Building a Large Scale Dataset for Image Emotion Recognition: The Fine Print and The Benchmark' (https://arxiv.org/pdf/1605.02677.pdf) by Quanzeng You, Jiebo Luo, Hailin Jin and Jianchao Yang. 

### Music Generation
Music generation is made possible by using [Performance RNN](https://magenta.tensorflow.org/performance-rnn). In fact, the crucial code for AComPianist's music generation ability is a modification of that found in the [demo code of Performance RNN](https://github.com/tensorflow/magenta-demos/tree/master/performance_rnn). 

### The Link
The prediction made by the emotion classification neural network determines the tonality of the music. In particular, if the neural network predicts that the image evokes positive feelings, the tonality will be major (which often induces positive feelings in the listener); if otherwise, it will be minor (which often induces negative feelings). In addition, the image's average hue is calculated. As one moves one's mouse over the image, the difference between the average hue and that of the pixel the cursor is currently pointing to will be determined. By leveraging on the similarity between the colour wheel and the circle of fifth, as shown below, we can relate a difference in visual experience to an auditory one. 

The brightness of the pixel, which the cursor is currently pointing to, determines the note density: the brighter the pixel, the lower the note density and vice versa. Since a lower note density makes the piece sound less complex and a lighter colour often represents pureness, this feature can strengthen the connection between the visual experience and the auditory one.  

Lastly, the rate at which the cursor moves influences the loudness of the generated music. The cursor movements can represent the hand movements of a videographer as he/she films a video (which consists of many images). Since rapid and abrupt movements while one films a video, can result in more noise, we would want the generated music to be louder, in hopes that it would drown out any noise captured due to the abrupt movements.


References
----------
