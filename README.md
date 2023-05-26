# Scene Recognition and Segmentation with Deep Learning
## Project Description
In this project, we design and train deep convolutional networks for scene recognition. In Part 1, we train a simple network from scratch. In Part 2, we implement a few modifications on top of the
base architecture from Part 1 to increase recognition accuracy to ∼55%. In Part 3, we instead fine-tune
a pre-trained deep network to achieve more than 80% accuracy on the task. We will use the pre-trained
ResNet architecture which was not trained to recognize scenes at all.
These different approaches (starting the training from scratch or fine-tuning) represent the most common
approach to recognition problems in computer vision today–train a deep network from scratch if you have
enough data, and if you cannot then fine-tune a pre-trained
network instead.

**See Report for more project related details**

## The basic learning objectives of this project are:
* Construct the fundamental pipeline for performing deep learning using PyTorch
* Experiment with different models and observe the performance

**In order to train a model in Pytorch, following four components:**
* Dataset: an object which can load the data and labels given an index 
* Model - an object that contains the network architecture definition 
* Loss function - a function that measures how far the network output is from the ground truth label
* Optimizer - an object that optimizes the network parameters to reduce the loss value


