---
title: "Recurrent-Neural-Networks"
comments: true
mathjax: true
share: true
toc: true
---

### what are RNNS?

RNN的主要思想是利用顺序上的信息。在传统的神经网络中，通常会认为所有的输入或输出都是相互独立的，但有很多情况下并非如此：例如在预测一句话中下一个词会出现什么，就必须把句子中先前的词语一并考虑进去；或是在时间序列问题中，先前的情况同样需求考虑在内。

RNN被称为recurrent主要由于其会对每一个序列做相同的操作，且每个序列的output都与之前一系列的计算相关，可以被理解为是一种'memory'。RNNs可以记忆非常长的一个序列，但在实际应用中，通常只会回溯过去一小段的情况。

![foo](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/09/rnn.jpg)

上图展示了单个RNN完全展开下的情况，举例来说如果想计算一个词语数为5的句子，那么该序列会被展开成一个5层的神经网络，以下详细说明图中各符号的意义：

*  $$x_t$$表示在第t步时的输入值。例如$x_1$表示的是句子第二个词的one-hot向量
*  $s_t​$表示在第t步时的hidden state。即上述所提到的'memory'，其由先前步骤的hidden state与当前t步的输入计算而得：$s_t = f(U*x_t+W*s_{t-1}))​$，这里的函数f为激活函数，通常为tanh或Relu。$s_{-1}​$通常会是0初始值
*  $o_t$是在第t步时的输出值。