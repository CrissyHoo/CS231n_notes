# Lecture 11 | Detection and Segmentation

this time is the application of neural network on computer vision task.

talk something about transpose convolution. It's a learnable upsampling way. It's just another kind of convolution. 

RNN is really slow. So there is fast rnn. 

introduce some thing about object detection, not pretty deep into it, just introduce.

# Lecture 12 | Visualizing and Understanding

input is an image and then the computer should give some descriptions about this image.

What happened in side the convnetwork.

the 1st layer: 

usually the conv layer, but we can't get some useful message from the visualization result.

the last layer:

  tell us the result, usually the fully connected layer. 

occlusion decision: replace some part of the image with mean pixel values, and see the prediction result. if the replacement make the result change drastically, then we can know that the replaced part is very important for the decision.

an example: two image: the one is an elephant standing on the left side and the other is on the right side. From image level, these 2 images are totally not the same but from feature maps that extracted, they are much more similar.

Saliency maps: masking, is also can be used in semantic segmentation. We want to know which pixels in the image are important for the classification. One way is to compute the gradient of the predicted class score with respect to the pixels of the input image. 

Gradient ascent: fix the weights of the training network. 

gram matrix matching: do the synthesis.

some description about style transfer

# Lecture 13 | Generative Models

### unsupervised learning

in supervised learning, we have data and label. like classification, regression, object detection.

un: clustering, dimensionally reduction, feature learning, density estimation.

**Generative Model**

VAE

another kind of generative model. encoder and decoder. 

GANs

the main idea:

![](https://i.loli.net/2021/05/18/MSqpTOKtwuoBXgI.png)

![](https://i.loli.net/2021/05/18/XwyxBGimYtDUvQL.png)

about the problems of inference queries, see the example of VAE

# Lecture 14 | Deep Reinforcement Learning

Q-Network

state and action.

experience replay

