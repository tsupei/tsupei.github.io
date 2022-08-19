---
layout: article_post
title:  "交叉熵(cross entropy)"
categories: cs nlp
tags: entropy cross-entropy loss
excerpt_separator: <!--more-->
highlight: true
---

{% include widget/excerpt.html text="In information theory, the cross entropy between two probability distributions 
$p$ and 
$q$
over the same underlying set of events measures the average number of bits needed to identify an event drawn from the set if a coding scheme used for the set is optimized for an estimated probability distribution 
$q$
, rather than the true distribution 
$p$
." %}

<!--more-->

### 抽色球例子 
---

[「怎樣理解Cross-Entropy」](http://shuokay.com/2017/06/23/cross-entropy/)這篇文章舉的例子我認為非常適合來幫助我們理解`Cross Entropy`，假設我們有從袋子裡面抽出色球，裡面分別有一顆`藍球`、`紅球`、`綠球`、`橘球`，抽到的機率都為$\frac{1}{4}$，我們每次抽完一顆球後，可以問一個是二分的是非題，例如：是不是藍球或紅球？而我們想計算平均要問幾次，才知道抽的球的顏色

所以我們最好的策略是

```bash
藍球或紅球？
├── Yes
│   └── 藍球？
│       ├── Yes
│       │   └── 藍球
│       └── No
│           └── 紅球
└── No
    └── 綠球？
        ├── Yes
        │   └── 綠球
        └── No
            └── 橘球
```

1. 問是不是藍球或紅球？
2. 是的話，問是不是藍球
3. 否的話，問是不是綠球

上述事件的熵(Entropy)可以透過以下式子計算

$$ \textit{Entropy} = \sum_{i} p_{i} log_{2}(\frac{1}{p_{i}}) = \sum_{i} -p_{i} log_{2}(p_{i}) $$

$$ = \frac{1}{4} \times 2 + \frac{1}{4} \times 2 + \frac{1}{4} \times 2 + \frac{1}{4} \times 2 $$

$$ = 2 $$

不過假設我們今天不用上述的策略來問，改成以下策略

```bash
藍球？
├── Yes
│   └── 藍球
└── No
    └── 紅球？
        ├── Yes
        │   └── 紅球
        └── No
            └── 綠球？
                ├── Yes
                │   └── 綠球
                └── No
                    └── 橘球
```

這時候我們就必須借助`Cross Entropy`來衡量哪樣的策略比較適合，從中文維基百科的定義來看：

在資訊理論中，基於相同事件測度的兩個概率分布$p$和$q$的交叉熵是指，當基於一個「非自然」（相對於「真實」分布$p$而言）的概率分布$q$進行編碼時，在事件集合中唯一標識一個事件所需要的Average Number of bits

所以我們的策略可以看做是$q$，而真實分布也就是抽色球的機率為$p$，當我們使用$q$來選擇是，平均需要多少次問題才能得到正確答案

$$ \textit{Cross Entropy} = \sum_{i} p_{i} log_{2}(\frac{1}{q_{i}}) = \sum_{i} -p_{i} log_{2}(q_{i})$$

$$ = \frac{1}{4} \times 1 + \frac{1}{4} \times 2 + \frac{1}{4} \times 3 + \frac{1}{4} \times 3 $$

$$ = 2.25 $$


而如果$p$分布跟$q$分布相同的話，即為$p$的Entropy。

我們可以進一步證明，兩分布不同時所得到的$ \textit{Cross Entropy}$一定大於真實分布的熵$ \textit{Cross Entropy}(p)$

我們表示$p$和$q$的`Cross Entropy`為$H(p, q)$，$p$和$q$個別的熵為$H(p)$和$H(q)$，$ D_{KL}(p \parallel q)$表示我們基於$q$分布來表示$p$分布所需要的額外編碼，我們在之後的篇幅會更詳細介紹

$$ H(p, q) =  D_{KL}(p \parallel q) + H(p) $$

我們可以利用$x-1 \ge log(x)$來證明$ D_{KL}(p \parallel q) \gt 0 $

$$ D_{KL}(p \parallel q) = -\sum_{i} p_{i} log_{2}(\frac{q_{i}}{p_{i}}) \ge -\sum_{i} p_{i} (1-\frac{q_{i}}{p_{i}}) = \sum_{i} p_{i} (1 - \frac{p_{i}}{q_{i}}) = -\sum_{i}p_{i} + \sum_{i}q_{i} = 0
$$

### 分類問題

在分類問題的模型，常常在最後一層會加一層`softmax`來模擬機率分布，而在損失函數的選擇上通常會選用`Cross Entropy`，讓我們來看一下如果是單類別的分類模型，會是怎麼進行

$$ \textit{Cross Entropy} = \sum_{i} -p_{i} log_{2}(q_{i}) = -p_{k} log_{2}(q_{k}) = -log_{2}(q_{k}) $$ 

因為我們的正確答案只會是某一個類別，假設是類別$k$，所以$p_{k} = 1 $，而其他為0


### Entropy
---

$$ H(X) = \sum_{i} -p_{i} log_{2}(p_{i}) $$

{% include widget/photo.html url="https://photos1.blogger.com/blogger/5682/4111/1600/EntropyVersusProbability.0.png"
description="From [http://www.chinakdd.com/article-124761.html]<br>Entropy is max on pi=0.5"
%}

### Reference
---
1. [Entropy from Wiki](https://en.wikipedia.org/wiki/Entropy_(information_theory))
2. [A Gentle Introduction to Cross-Entropy for Machine Learning by Jason Brownlee](https://machinelearningmastery.com/cross-entropy-for-machine-learning/)
3. [機器/深度學習: 基礎介紹-損失函數(loss function) by Tommy Huang](https://medium.com/@chih.sheng.huang821/機器-深度學習-基礎介紹-損失函數-loss-function-2dcac5ebb6cb)
4. [怎樣理解Cross-Entropy](http://shuokay.com/2017/06/23/cross-entropy/)