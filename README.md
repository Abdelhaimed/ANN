# ANN
Artificial Neural Network for Hadwritten digit recognition
Yahia Asma,Bouaicha Mohammed Abd Elhai,Houari Houcine Abd Allatif
Universite Kasdi Merbah Ouargla ´
Faculte des Nouvelles Technologies de l’Information et de la Communication ´
Abstract
Classification is an important data mining technique with a wide range of applications to classify the various
types of data existing in almost all areas of our lives. In this study we are going to implement a handwritten
digit recognition application using the MNIST dataset. To solve this problem, we will be using a special type
of deep neural networks that is Artificial Neural Network. The key component to most Artificial Neural
Network (ANN) projects is understanding how the training actually happen, and being able to test the results
correctly in order to get the most out of the given dataset, to tune training parameters, and at the end,
achieving an efficient ANN model.
An existing article was utilized as the ground example to the structure used in the project [1]. First step was
to identify the main parts, starting with data normalization, defining the structure Artificial NN model, and
then training and testing it. Then by understanding every piece of code and highlighting tasks that are
handled by high-level functions to be replaced and reprogrammed (in particular functions and methods that
are defined by the keras module) and finally apply it to solve our problem. During the process, existing tools
were used, denoting numpy, and matplotlib (for plotting and testing).
Key words: Artificial Neural Network, Handwritten digits, Artificial intelligence, Neural networks, MNIST,
numpy, Big data, Machine learning.
1 Introduction
Machine Learning is an artificial intelligence technology that enables computers to learn and imitate human
skills. Just like humans, computers need experience in
order to learn and grow, they need data to analyze and
train on. In fact, Big Data is the essence of Machine
Learning.
In the current age of digitization, handwriting recognition plays an important role in information processing. A lot of information is available on paper, and
processing of digital files is cheaper than processing traditional paper files. The aim of a handwriting recognition system is to convert handwritten characters into
machine readable formats. Since the end of the sixties, intensive work has been carried out in the field
of handwriting recognition. Indeed, the applications
of digit recognition now is available and various, it includes in reading the numbers of checks, postal mail
sorting, bank check processing, form data entry, etc.
The heart of the problem lies within the ability to apply an efficient algorithm that can recognize hand written digits and which is submitted by users by the way
of a scanner, tablet, and other digital devices. Thanks
to recent advances in computing power, many recognition techniques writing have been developed already,
and the one we are using in this study is the Artificial
Neural Network.
2 What is the handwritten digit
recognition
The handwritten digit recognition is the ability of machines to recognize human handwritten digits. The difficulty of this task for the machine lies in the diversity
of hand writings and the variety of ways of writing
numbers made with many different flavors. We present
in this study the solution to this problem by training the machine to recognizes the handwritten digit
present in any image.
3 Artificial neural network
3.1 What is Artificial neural network
Artificial Neural network is a machine learning algorithm and a programming paradigm ,it inspired by
the biological neural networks that constitute animal
brains, and it’s simply about telling the computer what
to do by making it learn from data .[2] [3]
1
3.2 The architecture of an ANN
Artificial Neural Networks are complex structures
made of connected artificial neurons organized in layers
:
• An input layer
• One or multiple hidden layers
• An output layer
Each node is connected with another node from the
next layer, and each such connection has a particular
weight. Weights are assigned to a neuron based on its
relative importance against other inputs.
Figure 1: Artificial Neural Network architecture
3.3 What is it used for
Generally ANNs are invented to make computer acts
like human and learns how to solve problems from a
previous experience, and they are used for modelling
non-linear problems like classification, clustering, prediction. . . [4]
3.4 How does it work
ANNs are based on predication the set of connection
between neurons values, when we say predication it
means that there is a probability in background. So
the task is to get an output value for given input parameters(data) after being trained on similar data. [4].
3.5 How do we train it
To train the ANN model we need to follow this process:
1. Feed forward propagation
2. Back propagation
3. Updating the weights and baises
4 Conception
4.1 Data set
The MNIST dataset is one of the most popular datasets
available on the internet. It contains in total 70,000 labeled images of handwritten digits from zero to nine,
60000 of them are specified for training and 10,000
for testing. Each example included in the MNIST
database is a 28x28 grayscale image of handwritten
digit and its corresponding label (0-9).
This Python script enables us to load the MNIST
database and save into numpy arrays that we will use
in our training, noting that the input images (of 28*28
dimensions) are reshaped into 784-unit vectors. This
is necessary because the first layer of the network is
expecting a vector input.
Figure 2: Dataset loading
Currently the pixels in the input vectors range from
[0,255], but we need to normalize them to [0,1], so we
devise each pixel by 255.
Figure 3: Normalization of the input vectors
Now that our dataset is loaded and our training labels are present in a vector of length 60,000 containing
the 0-9 labels for each training image (there is a single
digit for each image), we call this function that transforms each one of our labels into a binary vector with
2
one column for each class value to match the output
of the network.The element with the index of the expected class take 1 and all the other elements of the
vector take the 0 . This is required to calculate the
error for the output layer later.
Figure 4: to categorical function
4.2 Initialize Network
Our network is organized into the following layers:
• The input layer: it is a row from our training
dataset, consisting of 28x28 images. We flatten
these images into one array of 784 , which is the
number of weights in the hidden layer, each node
represents a pixel of the input image. Nodes are
neurons that actually do nothing. They just take
their input value and send it to the neurons of the
next layer.
• The hidden layers with a selectable number of neurons for each. This will allow us to change the
number of neurons for a better learning.
• The output layer with 10 neurons corresponding
to our 10 classes of digits, from 0 to 9.
This is a dense neural network, which means that
each node in each layer is connected to all nodes in the
previous and next layers.
first we will create our model, initialize the weights
and bias of the network. This function creates the
model , it takes as parameter a list of 4 integers that
represent the number of neurons of the layers (input
layer , 1st hidden layer, 2nd hidden layer , output layer)
. It creates a dictionary where :
• ”Wi” : represents a matrice of m*n dimension
where m is the number of neurons of the layer(i)
and n is the number of neuron of its previous layer.
The function creates a list of random weights for
each neuron of the layer.
• ”bi” : represents a list of n elements where n is
the number of neurons of the layer(i), and each
element is initialized randomly and it represents
the bias of a neuron in the layer.
Figure 5: Initialize network
4.3 Activation functions
The role of activation functions is to introduce nonlinearity, non-linear means that the output cannot be
reproduced from a linear combination of the inputs.
There are too many activation functions and because
we are dealing with a classification problem, we use
the sigmoid activation function in the hidden layers
because it keeps values in the range between 0 and 1 .
For the output layer we use the softmax activation
function because it is used for multi-class classification,
the later amplifies the big values in the set of output
values and lowers the smaller ones, which helps increase
the difference between the dominating class then the
others and thus reduces the cost.
Here we represent the activation functions used and
their derivatives that will be needed later in the backpropagation .
Figure 6: Activation functions
3
4.4 Forward Propagation
After creating our model (network), the first step in our
training is the forward propagation, we can calculate an
output from a neural network by propagating an input
signal through each layer until the output layer outputs
its values. We call this forward-propagation. It is the
technique we will need to generate predictions during
training and it will be corrected within the training
process and used again to make predictions on new
data.
Our forward propagation function takes as parameters the model that we have created and the input
layer.This function takes all the layers, it gets the dot
product of each layer’s weights with the values of the
previous layer, it brings them together with the bias
and then feed this sum to an activation function .
Figure 7: Forward propagation
4.5 Backward Propagation
The back-propagation algorithm is named for the way
in which weights are trained. Error is calculated between the expected outputs and the outputs forward
propagated from the network. These errors are then
propagated backward through the network from the
output layer to the hidden layers, saving the errors of
each layer as a vector that will be used later to tune
the weights and biases.
The first step is to calculate the error for each output
neuron, this will give us our error signal (input) to
propagate backwards through the network. The error
in the hidden layers is calculated in function of the error
of it’s next layer’s error , and it repeats this backward
process until it arrives to the first hidden layer .
Figure 8: Back propagation
4.6 Updating network
Once errors are calculated for each neuron in the network via the back propagation method above, they can
be used to update weights. The new weights will be
calculated by multiplying the learning rate by the error,which has been calculated in the back-propagation
procedure,multiplied by input and finally added to the
old weight. The same procedure can be used for updating the bias except for the input which takes the
fixed value 1.0.
Figure 9: Updating network
The learning rate is a parameter used to adjust the
speed of learning, a value of 0.1 will update the weight
10% of the amount that it possibly could be updated.
Small learning rates are preferred that cause slower
learning over a large number of training iterations.
This increases the likelihood of the network finding a
good set of weights across all layers rather than the
fastest set of weights that minimize error (called premature convergence).After many tests, we have found
that in our case of study, 0.01 is a reasonable value to
avoid overshooting and yet reach convergence in less
iterations as possible.
4.7 Training
The network is updated using stochastic gradient descent(Batch Size = 1). This involves first looping for a
fixed number of epochs and within each epoch updating the network for each row in the training dataset.
Because updates are made for each training pattern,
this type of learning is called online learning. If errors
were accumulated across an epoch before updating the
weights, this is called batch learning or batch gradient
descent (Batch Size = Size of Training Set). [5]
After preparing our training dataset and initializing
our network we can call the training function, that, by
4
default, has a learning rate equal to 0.01, and a fixed
number of epochs equal to 10. For each input, which
is an image presented as a vector of 784 elements, we
apply a forward propagation. The output of the forward propagation and the expected label pass in the
backward-propagation function. This gives us a dictionary of updates to the weights, we apply these updates
to the neural network.
Figure 10: training
After each epoch iteration, we can call an evaluation
function that measure the accuracy and the loss (error
between the expected output and the network output).
This is helpful to create a trace of how much the network is learning and improving after each iteration over
the whole dataset.
Figure 11: Evaluation function
In the end of these iterations the network will be well
trained to recognize handwritten digits.
5 Implementation
5.1 Evaluate the model
Now that we have all the pieces we can call our functions to solve our problem.
First we will load the data of training and testing
then we initialize the network with 128 neurons in the
first hidden layer and 64 neurons on the second hidden
layer. After that we start the training.
Within each epoch iteration we call the evaluation
function and we print the accuracy and the loss in order to make the comparison . As you can notice from
the results the cross entropy loss has dramatically decreased from the first to the last iteration while the accuracy has increased which shows how good our neural
network is improving within each epoch.
Figure 12: Accuracy graph
Figure 13: Loss graph
5
5.2 prediction
Now our network is ready to predict any handwritten
digit by simply passing it through the forward propagation function . It will be more useful to turn this
output, which is a vector of probabilities for each class,
back into the the class value (digit) with the higher
probability. This is done using the arg-max function.
We have a dataset of 10000 images to test, so we select
any index and get it’s picture shown with the expected
label and the label predicted by our network.
Figure 14: A prediction test on the network
5.3 Discussion
The performance of classification algorithm is generally
examined by measuring the accuracy of the classification. In this study, we have shown the effectiveness
of the ANN classification algorithm and the good results obtained classifying the handwritten digits using
this algorithm. After training the network with ANN
over the MNIST dataset, we have got an accuracy of
90.68%, as a conclusion that this algorithm is effective
for the handwriting digit classification and it can be
usefully applied to other classification problems.
6 Conclusion
In this study, we suggest a model for the detection of
handwritten digits using ANN.This domain of neural
networks is increasingly becoming a major player due
to its ability to adapt to a wide range of problems in
other domains and disciplines. Although it is not comparable with the power of the human brain, still it is
the basic building block of the artificial intelligence especially with the evolution of the Big Data.
References
[1] Casper Hansen
Neural Network From Scratch with NumPy and
MNIST, url=”https://mlfromscratch.com/neuralnetwork-tutorial/”
[2] Nielsen, Michael Neural Networks and Deep Learning 2015.
[3] Zerium, Aegeus Good
Audience,url=https://blog.goodaudience.com/artificialneural-networks-explained-436fcf36e75 24-07-
2014.
[4] Stevin Walczak, narciso Cerpa ScienceDirect,Artificial Neural Network,
url=https://www.sciencedirect.com/topics/engineering/artificialneural-network
[5] Jason Brownlee Difference Between a Batch and
an Epoch in a Neural Network, Deep Learning,
url=https://machinelearningmastery.com/differencebetween-a-batch-and-an-epoch/
6
