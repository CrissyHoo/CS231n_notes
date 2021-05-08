# Lecture 1 | Introduction to Convolutional Neural Networks for Visual Recognition

## the history of computer vision
the history of the life development

the history of vision

the history of camera(similar to the eyes structure)

the research about the cat's visual cortex

input->primal sketch->2D sketch-> 3D model

image segmentation, face detection, object recognition

## an overview
image classification/ object detection/ image caption

CNN——an important tool

AlexNet(2012)->GoogleNet->VGG(2014)

how to make the machine understand the content of one image, it's really amazing

# Lecture 2 | Image Classification
 a basic framework of training, testing a model.

#### how to compare images:

L1 distance: directly minus 

k-nearst neighbors

#### how to use data

split data into train,val and test; choose hyperparameters on val and evaluate on test

### Linear classification
linear classifier: f(x,W)=Wx+b. It can't deal with a complex classification problem.

# Lecture 3 | Loss Functions and Optimization

the definition of loss function and optimazation

optimization: walk in a valley. There are some strategies:
<ol>
<li>Random search. 
```python
bestloss=float("inf")
	for num in xrange(1000):
		W=np.random.randn(10,3073)*0.0001
		loss=L(X_train,Y_train,W)
		if loss<bestloss:
			bestloss=loss
			bestW=W
```

</li>

</ol>

 










