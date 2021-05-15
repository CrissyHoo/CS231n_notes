# Lecture 10 | Recurrent Neural Networks

Some Review

Vanilla Neural Network(one to one, lack of flexibility)

there are many other kind of input and output type.

**Recurrent Network**

having a hidden state that feeds back at itself. 

![The basic structure of RNN](https://i.loli.net/2021/05/15/8U5DSAkuoImweYK.png)

In this network, there can be many middle result, and this can be used to deal with different problems, such as many to one and one to many problem.

![](https://i.loli.net/2021/05/15/QCfbHxZNJm7DG9j.png)

backpropagation in recurrent network

![](https://i.loli.net/2021/05/15/xTJnp8FEdqXLQaK.png)

quote detection, generate paper, code, proof and so on.

 LSTM（long short term memory）

![](https://i.loli.net/2021/05/15/32lebzBhjNHsd6p.png)

helps with the problem of vanishing and exploding gradients.

ht: hidden state      ct: cell state; is a kind of internal, kept inside of the LSTM

It's really amazing how people create this structure.

![the computation graph](https://i.loli.net/2021/05/15/YaHFvbsOrcljXVh.png)

uninterrupted gradient flow.

感觉讲的我理解得不是很好，所以用中文再写一遍。思路来自李宏毅。

#### RNN

这个的特点就是可以很好地处理序列数据。在nlp中有很多的应用，每个句子可以视为一个序列，每个单词会因为上下文的不同而表达不同的意思。

**一般的RNN**

![](https://i.loli.net/2021/05/15/M3UfrEmdaGwogsj.jpg)

x, y为当前节点的输入输出，h为接收到的上一个节点的输入，h撇为给下一个节点的输入。

![](https://i.loli.net/2021/05/15/cPB5mRwX4fkdaol.jpg)

对于每一个节点来说，y和h撇是关于h和x的函数。

#### LSTM

它其实是RNN的变体，为了解决长序列训练过程中梯度消失和梯度爆炸的问题。与RNN相比，LSTM能够在较长的序列上表现得更好。

![](https://i.loli.net/2021/05/15/CpkUMotBb1z8ewE.jpg)

这个是LSTM和简单的RNN的比较。

LSTM在状态传输方面，包含有h_t(hidden state)，还有一个隐藏的c_t(cell state)【RNN中ht对应LSTM的ct（为啥）】

这就可以结合cs231的PPT看，c_t通常是在c_t-1的基础上加上一些数值。h_t的逻辑要更复杂。

**关于LSTM更详细的说明**

对于LSTM的任一状态，构造一定的关系，这个关系的输入是x_t和上一个状态传递下来的h_t-1,我们可以得到四个状态。

![](https://i.loli.net/2021/05/15/xSnEZiFmyI3efGk.jpg)

![](https://i.loli.net/2021/05/15/zTDIlQum4Gi8MtV.jpg)

我们可以看到后三个状态最后都经过了sigmoid函数的处理，这个函数会把值转换到0到1之间，所以我们会在LSTM中听说“门”这个概念。而里面的输入就是concatenation后乘以权重矩阵。

z有略微的不同，z是通过tanh函数来进行激活的，处理后的值域为-1到1。

然后就回到LSTM中有四个状态，分别就是图中的几个z，然后贴一张和cs231其实有重复部分的图。这里描绘的是每一个单元的操作。

![](https://i.loli.net/2021/05/15/lmouxt58brMaFsj.jpg)

⊙是Hadamard Product，将矩阵中对应的元素相乘，相乘的两矩阵size需要一致。⊕代表矩阵加法。

LSTM的三个阶段

1. 忘记阶段。这个阶段主要是对上一个节点传进来的输入进行**选择性**忘记。简单来说就是会 “忘记不重要的，记住重要的”。z_forget就是一个忘记门控，来控制是否要抛弃c_t-1

2. 选择记忆阶段

   将输入有选择地记忆下来，也就是x_t，即上图中的z_information

   从图中我们也可以看到，将当前输入内容计算一波后加上z_information就是我们要传给下一个状态的c_t。

3. 输出阶段

   这个阶段将决定哪些将会被当成当前状态的输出。主要是通过 z_o。

   输出y_t最终也是通过h_t得到。

   以上参考

   []: https://zhuanlan.zhihu.com/p/32085405	"人人都能看懂的LSTM"

   

