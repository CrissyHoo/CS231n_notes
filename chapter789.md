#   Lecture 7 | Training Neural Networks II                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     

review: activation functions; weight initialization; Data preprocessing; batch normalization; hyperparameter search; 

## Fancier optimization

**SGD**

what if loss changes quickly in one direction and slowly in another?

then seeing the path during the optimization, you may find it is like a zigzag.  

local minimal: the gradient is zero and we will get stuck.

usually get the information from a mini-batch, which is usually inaccurate.

**in order to solve the above problem, we add momentum**

SGD+momentum

```python
vx=0
while True:
    dx=compute_gradient(x)
    vx=rho*vx+dx  #rho gives friction,is usually 0.9 or 0.99
    x+=learning_rate*vx
```

Why does this work?

the setting of rho helps. The idea is that if go back and look at this velocity estimate and look at the velocity computation, we're adding in the gradient at every time step. It kind of depends on your setting of rho, and you can image if the gradient is relatively small and if rho is well behaved in this situation  then our velocity could actually monotonical increase up to a point where the velocity could now be larger than the actual gradient. So that we might actually make faster progress along the poorly conditioned dimension.

![a picture here](https://i.loli.net/2021/05/13/7fsZ3WJwD4KUncX.png)

**AdaGrad**

```python
grad_squared=0
while True:
    dx=compute_gradient(x)
    grad_squared+=dx*dx
    x-=learning_rate*dx/(np.sqrt(grad_squared)+1e-7)
```

X has divided a square to do the modification to avoid zigzag.

**RMSProp**

```python
grad_squared=0
while True:
    dx=compute_gradient(x)
    grad_squared=decay_rate*grad_squared+(1-decay_rate)*dx*dx
    x-=learning_rate*dx/(np.sqrt(grad_squared)+1e-7)
```

RMSProp makes the square estimate gradually decay.

**Adam**

using snapshot during the training

some tricks(maybe useful)

![a picture here](https://i.loli.net/2021/05/13/kNp2n9rFCgZRi31.png)

Dropout

for a fully connected network, we make some of the neurons eliminate.

Dropout makes our output random.

**Transfer learning**

how to do better with less data

# Lecture 8 | Deep Learning Software

**CPU vs GPU**

NVIDA vs AMD

Having introduced some popular framework. 

The basic code during the training period with pytorch:

```python
import torch
from torch.autograd import Variable
from torch,utils,data import TensorDataset, DataLoader
N,D_in,H,D_out=64,1000,100,10

x=torch.randn(N,D_in)
y=torch.randn(N,D_out)
Loader=DataLoader(TensorDataset(x,y),batch_size=8)

model=TwoLayerNet(D_in,H,D_out)

criterion=torch.nn.MSELoss(size_average=False)
optimizer=torch.optim.SGD(model.parameters(),lr=1e-4)
for epoch in range(10):
    for x_batch,y_batch in loader:
        x_var,y_var=Variable(x),Variable(y)
        y_pred=model(x_var)
        loss=criterion(y_pred,y_var)
        
        optimizer.zero_grad()
        loss.backward()#backpropagation
        optimizer.step()#update the weights and the value of epoch plus 1
```

Visdom(similar to tensorboard. add your code and then visualize in a browser)

**recurrent network/recursive network/modular network**

need a dynamic graph cause the network sequence can be very different depending on the input.

# Lecture 9 | CNN Architectures

give an introduction of some typical network.

**AlexNet**

give us the architecture. 

![a picture here](https://i.loli.net/2021/05/14/JZNKkr9VoqmfvlB.png)

**VGG**

![a picture here](https://i.loli.net/2021/05/14/smA6TZzSaNyf8uM.png)

*If there is any loss during the 1 by 1 conv?*

maybe there will be some information loss but at the same time you do the concatenation and make the linear combination of these input feature maps. and after that you have introduced non-linear relationship into the network. these work can be helpful and we also find that using this can make the net work better.

**ResNet**

We found that as the layer becomes larger, the training error and the testing error can become larger.

It is mainly the optimization problem.

*Why resNet work?*

![a picture here](https://i.loli.net/2021/05/14/NBqPWK342etlJRw.png)

![a picture here](https://i.loli.net/2021/05/14/oE8WGqZNxm4BDQS.png)

???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????resnet????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

introduce some network: NiN???identity mapping???ResNeXt

Densenet: dense block, squeezenet(have a squeeze layer)

