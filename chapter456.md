# Lecture 4 | Introduction to Neural Networks

Review

numerical gradient is slow but easy to write. analytic gradient is fast, exact but is error-prone. In practice, we derive analytic gradient, check the implementation with numerical gradient.

### how to calculate analytic gradient

#### computational graph

#### backpropagation

![](https://i.loli.net/2021/05/09/Gz4rRA8SUIjkBD1.png)

we calculate the derivation of each node in the graph.(the red word) 

Then we get a more complex example. The process of calculating the local gradient from the end of the graph to the start is just the process of backpropagation.

Transfer to a computational graph & use chain rule

##### a vectorized example

you need to know something about Jacobian. 

Using the basic rules of derivation.

### Neural Network

multiple layers

nothing special to say

## About assignment1

1. **KNN exercise**

   

# Lecture 5 | Convolutional Neural Networks

##### some history 

##### some applications of CNN

##### Fully connected layer

some concept: filter, receptive field, 

#### ![](https://i.loli.net/2021/05/10/wqrMvzRm8GO6ZxX.png)

> the filters at the earlier layers usually represent low-level features, like edges.
>
> At the middle level, we get more complex kind of features, like corners and blobs.

some questions about padding

![](https://i.loli.net/2021/05/10/dsvHBhqM1mJanGO.png)

spatial extent F is just the size of the filter.

##### 1*1 convolution layers make perfect sense

after the operation of 32 filters with a size of 1* 1 * 64 on a 56 * 56 * 64 image, the image size become 56 * 56 * 32

about pooling

# Lecture 6 | Training Neural Networks I

activation functions

**sigmoid**

when it's very negative, the gradient is nearly zero.

the outputs are not zero-centered( [some explanation about aero-centered](https://blog.csdn.net/weixin_43835911/article/details/89294613) )

exp() is a bit compute expensive

**tanh**

zero-centered but still kill gradients

**relu**

does not saturate in +region

computationally efficient

easy to converge

is more biologically plausible

non-zero centered

**leaky relu**

doesn't saturate(saturate means, like sigmoid, when the input value is pretty high or low, the gradient is nearly zero)

computational efficient

converges much faster

**exponential linear units**

in negative field, it's more robust.

In summary,

![](https://i.loli.net/2021/05/10/qmDdLXeBVZ8jhix.png) 

## Data preprocessing

make the data normalized

In practice, we substract the mean image(AlexNet) or substract per-channel mean(VGGNet).

## Weight initialization

you'd better not initialize every weight equally cause every neuron will active in a same way.

#### some ideas

1. small random numbers

   work for small networks but problems with deeper networks.

   ![](https://i.loli.net/2021/05/10/nq13NuxQ76v9mfs.png)

   an experiment showing why there are problems with deeper networks.

   we are given a 10-layer network and then in each layer there are about 500 neurons. After we give the weight using the above method, we found that the mean value of  layer input is all around zero. 

   *What the gradient look like on the weights?*

   the input value of each layer is very small as we have mentioned above, whose mean value is around zero.

   As a result, as the gradient is propagated upstream, the gradient need to multiply by the weights. The gradient will become smaller and smaller and then we may find the gradient is collapsing to zero.

   **As we have choose 0.01 in the weight assignment equation,** we got a very small weight. What if we use 1 instead of 0.01?
   It's gonna to be saturated.

2. Xavier Initialization![](https://i.loli.net/2021/05/10/8iKt7kpRXWJ1jsA.png)

   when using the Relu nonlinearity it breaks.

   [a more specific explanation](https://blog.csdn.net/weixin_36670529/article/details/104336598)

## Batch normalization

We have N training examples in one batch.

![](https://i.loli.net/2021/05/10/JdFKt3zQBw29C1v.png)

forcing our data to become unit gaussian

## Monitor the training process

how to choose the learning rate

## Hyperparameters optimization

##### Cross-validation strategy

