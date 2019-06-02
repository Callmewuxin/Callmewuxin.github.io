---
layout:     post
title:      "对于训练数据可验证的鲁棒性的训练方法"
subtitle:   "论文解读"
date:       2019-06-02
author:     "JohnReese"
header-img: "img/post-6-2.jpg"
mathjax: True
tags:
    - ML
---

## 引言
论文链接为：[https://arxiv.org/abs/1711.00851v3](https://arxiv.org/abs/1711.00851v3)  
论文主要提出了一种方法。我们还是从头开始梳理这篇论文吧。

## Introduction

作者简单地介绍了近几年的抵御对抗训练的方法被`纷纷打脸`。
>  Somewhat memorably, many of the adversarial defense papers at the most recent ICLR conference were broken prior to the review period completing (Athalye et al., 2018)

这句话的翻译如下：
> 令人印象深刻的是，最近ICLR会议上的许多对抗防御论文在审查期结束之前就被破坏了。

那么作者就引出了自己的贡献，我也觉得很厉害，所以专门写个Blog详细记录下：
1. a method for training `provably` robust deep ReLU classifiers, classifiers that are guaranteed
to be robust against `any` norm-bounded adversarial perturbations on the `training set`.
2. a provable method for detecting `any` previously unseen adversarial example, with `zero false negatives`. 当然作者也说能达到这样的结果也会导致将一些非对抗样本列为对抗样本。


## Training Provably Robust Classifiers
1. A $k$ layer feedforward ReLU-based neural network, $f_{\theta}: \mathbb{R} ^ {|x|} \rightarrow \mathbb{R} ^ {|y|} $ given by the equations:
    $$
    \hat{z}_{i+1} = W_{i}z_{i} + b_{i}, for\ i = 1,...,k-1 \\
    z_{i} = max(\hat{z}_{i}, 0) \\
    其中z_{1} \equiv x\  且\ f_{\theta}(x) \equiv \hat{z}_{k} \\
    使用\theta = \{W_{i}, b_{i}\}_{i=1,...,k}来表示网络中的所有参数
    $$

    这一段是基本的关于DNN的描述。
2. 我们使用集合
    $$ Z_{\epsilon}(x) = \{f_{\theta}(x + \Delta):{||\Delta||}_{\infty} \leq \epsilon \} $$

来表示对抗多边形(`adversarial polytope`)的点的集合,即图1中间那个坐标轴的图形，是非凸的。而他们方法的基础是使用外凸边界(`convex outer bound`)来包裹对抗多边形。那么只要能保证外凸边界中的点都改变不了最终DNN的输出就能够保证原来的对抗多边形的点也是如此，那么鲁棒性就得到了满足。这是比较好理解的。

    ![img](/img/2019-6-2/image1.JPG)

3. 构造外凸边界的第一步是对于ReLU函数的线性表示。
    ![img](/img/2019-6-2/image2.JPG)
