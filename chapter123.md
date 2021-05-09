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

the definition of loss function and optimization

optimization: walk in a valley. There are some strategies:

1. Random search. (really bad)
```python
bestloss=float("inf")
for num in xrange(1000):

	W=np.random.randn(10,3073)*0.0001 #update w randomly 
	
	loss=L(X_train,Y_train,W)
		
	if loss<bestloss:
		
	bestloss=loss
		
	bestW=W
```


2. Find a slope

   the slope is the derivative of a function.

   In 1-dimension, it's a single value. In multiple dimension, the gradient is a vector. 

   The process: We get a current weight. We add an "h" to the weight. Then we calculate the loss respectively. The gradient is 
   $$
   \frac{df(x)}{dx}=\lim\limits_{h\to0}{\frac{f(x+h)-f(x)}{h}}
   $$
    The f(x) is just the loss function. After getting the slope, we can use slope to adjust the directions we update the weights.

   which is super slow and super bad.

   or using calculus. We directly find the dW value.

3. Gradient Descent

   ```python
   while True:
       weights_grad=evaluate_gradient(loss_func,data,weight)
       weights+=-step_size*weight_grad
   ```

   **stochastic gradient descent**

   When we calculate loss, we have to get the average value, which means we have to div N. We use minibatch of examples to calculate, whose size is just the value of N.
   
   **the extraction of image features**
   
   extract and then concatenate
   
   use histogram of oriented gradient to express image feature
   
   use bag of words.  some method referred nlp.
   
   use covnet directly
   
   









