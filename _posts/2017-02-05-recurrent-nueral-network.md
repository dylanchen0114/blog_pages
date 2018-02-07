---
title: "Recurrent-Neural-Networks"
comments: true
mathjax: true
share: true
toc: true
---

### what are RNNS?

全连接的一般神经网络如下图，其隐藏层的计算只取决于当前样本的输入，与其他样本无关，认为所有的输入或输出都是相互独立的。

![foo](https://upload-images.jianshu.io/upload_images/1667471-7d73a2ab30e3353a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/478)

而RNN的主要思想是利用顺序上的信息。例如在预测一句话中下一个词会出现什么，就必须把句子中先前的词语一并考虑进去；或是在时间序列问题中，先前的情况同样需求考虑在内。

下图展示了隐藏层间的连接方式，从左向右都是相互关系有时序的

![foo](http://img.blog.csdn.net/20161112232627589?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

RNN被称为recurrent主要由于其会对每一个序列做相同的操作，且每个序列的output都与之前一系列的计算相关，可以被理解为是一种'memory'。RNNs可以记忆非常长的一个序列，但在实际应用中，通常只会回溯过去一小段的情况。

![foo](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/09/rnn.jpg)

上图展示了单个RNN完全展开下的情况，举例来说如果想计算一个词语数为5的句子，那么该序列会被展开成一个5层的神经网络，以下详细说明图中各符号的意义：

*  $$x_t$$表示在第t步时的输入值。例如$x_1$表示的是句子第二个词的one-hot向量
*  $$s_t$$表示在第t步时的hidden state。即上述所提到的'memory'，其由先前步骤的hidden state与当前t步的输入计算而得：$$s_t = f(U*x_t+W*s_{t-1}))$$，这里的函数f为激活函数，通常为tanh或Relu。$$s_{-1}$$通常会是0初始值
*  $$o_t$$是在第t步时的输出值。

下面的展开说明了为何RNNs可以记忆先前样本的信息，并结合当前输入来得到输出：

![foo](https://upload-images.jianshu.io/upload_images/1667471-a3efd4e7588c38fe.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/553)

### Initialization

对于参数U、V、W的初始化，不能简单地将其都设为0，会造成所有层的参数计算都对称。通常不同的激活函数会对应不同的初始化方法。如果这里用tanh来作为初始化函数的话，比较好的方式是将其控制在$$\left[-\frac{1}{\sqrt{n}}, \frac{1}{\sqrt{n}}\right]$$间，其中n是先前层的输出维度。

如果这里选取$$x_t$$的维度为8000，hidden layer size为100，则有：

$$\begin{aligned}s_t &= \tanh(Ux_t + Ws_{t-1}) \\o_t &= \mathrm{softmax}(Vs_t)\end{aligned}$$

$$\begin{aligned}x_t & \in \mathbb{R}^{8000} \\o_t & \in \mathbb{R}^{8000} \\s_t & \in \mathbb{R}^{100} \\U & \in \mathbb{R}^{100 \times 8000} \\V & \in \mathbb{R}^{8000 \times 100} \\W & \in \mathbb{R}^{100 \times 100} \end{aligned}$$



以下是实现各权重初始化的python实现：

```python
class RNNNumpy:
  
  def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
    # Assign instance variables
    self.word_dim = word_dim
    self.hidden_dim = hidden_dim
    self.bptt_truncate = bptt_truncate
    # Randomly initialize the network parameters
    self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
    self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
    self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
```


### Training RNN — Back Propagation Through Time

BPTT 的基本原理和 BP 算法是一样的，同样是三步：

1. 前向计算每个神经元的输出值；
2. 反向计算每个神经元的误差项值，它是误差函数E对神经元j的加权输入的偏导数；
3. 计算每个权重的梯度;
4. 用随机梯度下降算法更新权重

手动推导过程：[https://www.jianshu.com/p/39a99c88a565](https://www.jianshu.com/p/39a99c88a565)

python实现RNN：[https://github.com/dennybritz/rnn-tutorial-rnnlm](https://github.com/dennybritz/rnn-tutorial-rnnlm)



### Case Study

[https://weiminwang.blog/2017/09/29/multivariate-time-series-forecast-using-seq2seq-in-tensorflow/](https://weiminwang.blog/2017/09/29/multivariate-time-series-forecast-using-seq2seq-in-tensorflow/)