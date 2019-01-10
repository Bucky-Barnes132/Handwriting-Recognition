# Handwriting-Recognition

As people have different handwriting it is difficult for a computer or any device to understand those handwritings of different people. The handwriting recognition is the ability of a computer or a device to take input handwriting in the form of an image such as picture of handwritten text which is fed to the pattern recognition algorithm or a model. And this model recognizes the handwritten digits.

Here I have used Convolutional Neural Network (CNN) and a Support Vector Machine (SVM) classifier to recognize the handwritten digits. I have used MNIST (Modified National Institute of Standards and Technology) database of handwritten digits as an input to these models to recognize the digits. The CNN model performs well which has accuracy of 99%.

# Project Overview

As we know that all of us have different handwriting styles such as different strokes to represent a same letter, the amount of pressure given on paper to write something. All these things differs the handwriting of an individual.

For the humans to recognize the digits such as 0, 1, 2… it is easy for them, even if they have written differently. It is because they have millions of neurons present in the brain, which helps humans to easily recognize the data. But for a computer to understand these different handwriting it is difficult for them. So handwriting recognition systems were developed.

I have researched on Neural Network model, a model which was inspired from the neurons present in our brain. And a SVM classifier a supervised machine learning model to recognize the digits. I have used Python as a language to implement these machine learning models.

# Handwriting Recognition Using SVM Model

----  What is SVM?

A Support Vector Machine (SVM) is a supervised machine learning algorithm which can be used for both classification and regression problem. But it is mostly used for classification purpose. Let’s say we have given 2 or more labeled classes of data, a SVM classifies those different classes by drawing an optimal hyperplane that separates all the classes.

img
img

A hyperplane is drawn in such a way that it is at a midpoint between the classes. A hyperplane is a linear decision surface that splits the space into two parts.

----  Getting the Data

I have used a MNIST dataset of handwritten digits which has a training set of 60,000 examples and a test set of 10,000 examples. The digits have been size-normalized and centered in a fixed size 28x28 images. The MNIST dataset is in a CSV (Comma Separated Value) format.

img

----  Understanding the Code

Here I have imported all the necessary packages which are required such as pandas, matlplotlib and scikit-learn. Using the pandas read_csv () function I have used MNIST dataset. I have selected the data up to 21,000 rows using iloc () function. And then split the training and testing data using the train_test_split function. After splitting the datasets I convert an image into matrix of 28x28 pixels and pass it through the SVM classifier to classify the digits. The accuracy of the model was 92.57%.   

Refer (Handwriting Recognition using SVM.ipynb) file for the code

img 
img

# Handwriting Recognition Using CNN Model

----  What is CNN?

A Convolutional Neural Network is inspired by the neurons present in the visual cortex of our brain. Whenever we see something a millions of neurons gets activated. The CNN model is divided into two parts one is feature learning and other is classification. The feature learning consists of Convolutional layer, ReLu layer and a Pooling layer. The classification consists of Fully Connected layer and a Softmax layer.

img

Convolution
  In convolution we take a feature matrix (filter) and slide that matrix over the input images by performing the dot product between the input images and the filters. A feature matrix is a smaller size matrix in comparison to the input dimensions of the image that consists of real valued entries

img

Pooling
  The pooling reduces the size of the image by picking the highest number generated from the convolution of filter and the input image (convoluted feature). We take a window size of 2x2 and slide it over the convoluted feature.
  
  img
  
 ReLu Layer/Activation Layer
   The ReLu layer (Rectified Linear Unit) which is an activation function. The ReLu layer simply makes a negative number 0.
    
 Dropout
   The dropout forces an artificial neural network to learn multiple independent representations of the same data by alternately randomly disabling neurons in the learning phase. To perform a dropout on a layer, we randomly set some of the layers value to 0 during forward propagation. The dropout is used to prevent the problem of overfitting.
    
    img
    
Softmax
 The Softmax function is used to convert the outputs to probability values for each class. Well after going through the fully connected layer there will be a stack of images. And this function is used to make some probability out of it to identify the image.

----  Understanding the Code

I have import all the necessary packages using tensorflow as a backend. I have used random.seed() function so that it does not give random output every time. I have load the datasets using mnist.load_data() function. Before sending the input to the model it is required to normalize the image. I have created a CNN model and pass that data over the model. I have used batch size of 200 images, over 10 epochs to train the model. The accuracy of the model was 99%.

img



