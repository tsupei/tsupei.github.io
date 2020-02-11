---
layout: article_post
title:  "詞向量(Word Vector)"
categories: cs nlp
tags: word-vector skip-gram softmax CBOW hierarchical-softmax negative-sampling NCE
excerpt_separator: <!--more-->
highlight: true
---

`Stanford CS 224N`課程第一週內容

在2014年，Mikolov Tomàs等人提出的這篇 `Distributed Representations of Words and Phrases and their Compositionality` 為接下來許多以詞向量為基礎的模型定下根基，以往使用神經網路(Neural Network)訓練詞向量耗時長，這篇論文提出了一些方法進行改善，不僅提升效能也大幅降低訓練成本。

使用詞向量(word vector)讓我們能夠將詞彙轉換成能夠被機器理解的向量，並且這個向量能夠有效反應出詞義，近義詞之間有著相近的詞向量，例如：紅茶、奶茶、綠茶它們都是飲料類，所以會具備類似的詞向量，且在訓練的過程中我們只需提供足夠大的語料，不需要透過人工標記就能夠訓練這樣的模型(Skip-Gram, CBOW, ...)，甚至也能直接使用網路上其他人預訓練的模型(Google, Facbook, ...)


<hr>
*You shall know a word by the company it keeps - J. R. Firth 1957*

<hr>

一個字的意思是由周遭的字決定，這樣的想法基本上就是`skip-gram`所做的事情，我們藉由一個訓練目標函式，讓一個`中間字(center word)`能夠對預測周遭的字給出較高的機率，經過這樣的訓練過程，擁有相近的前後文的詞，就會得到更相近的詞向量

<!--more-->

### word2vec
<hr>

`word2vec`通常特指`Mikolov`在2013提出的詞向量及其訓練方法，而這樣使用`Distributed Representation`的向量，有一些很有用的特性，相似的字會具有相似的向量，因此在向量空間中，可以看出類似的概念會叢集在一起，此外，也具有詞間推理(Analogical Inference)的關係，像是`France - Paris + Japan = Tokyo`這樣的現象

### Review
<hr>

$$ \textit{Likelihood} = L(x) = L(\theta \mid x) = P( X = x \mid \theta) $$ 

$$ \textit{sigmoid function} = f(x) = \frac{1}{1+e^{-x}} $$ 

$$ \textit{logistic regression} = f(x) = \sigma(w^{T}x + b) $$ 

### Skip-gram
<hr>

$$ \textit{Likelihood} = L(\theta)= \prod_{t=1}^{T} \prod_{-m \leq j \leq m} P(w_{t+j} \mid w+{t} \:; \theta) $$

$$ J(\theta) = - \frac{1}{T} \log L( \theta ) = - \frac{1}{T} \sum_{t=1}^{T} \sum_{-m \leq j \leq m} \log P(w_{t+j} \mid w_{t} \: ; \theta ) $$

$$ \textit{for center word c and context word o,} \: P(o \mid c) = \frac{  e^{u_{o}^{T} v_{c}} }{ \sum_{w \in V} e^{u_{w}^{T} v_{c}}} $$

### Full Softmax
<hr>

$$ \textit{softmax} (x_{i}) = \frac{  e^{x_{i}} }{ \sum_{j=1}^{n} e^{x_{j}}} = p_{i} $$

### HS(Hierarchical Softmax)
<hr>

$$ n(w, \: j) $$ be the $$ j $$-th node from the root to $$ w $$

$$ L(w) $$ be the length of this path

$$ ch(n) $$ be the arbitrary fixed child of $$ n $$, always the left node

$$ \left [ x \right ] $$ be 1 if $$ x $$ is true else 0

$$ p(w \mid w_{I}) =  \prod_{j=1}^{L(w)-1} \sigma \left ( \left [ n(w, \:  j+1) = ch(n \: (w, \: j)) \right ] \cdot {v}'_{n(w,j)} \:^{T} v_{w_{I}} \right )  $$

`Hierarchical Softmax`採取二元樹的結構來計算 $$ P(w \mid w_{I}) $$，可以將時間複雜度壓縮到
 $$ O( \log ( \left | V \right | )) $$，`HS`不對每個字訓練向量，而是訓練從根節點到底部中節點的向量(The node in the path from the root to the leaf node)，舉例來說

$$ P(w_{2} \mid w_{i}) = p(n(w_{2}, \: 1), \: \textit{left}) \cdot p(n(w_{2}, \: 2), \: \textit{left}) \cdot  p(n(w_{2}, \: 3), \: \textit{right}) $$

$$ = \sigma(v_{n(w_{2}, 1)}^{T}v_{w_{i}}) \cdot  \sigma(v_{n(w_{2}, 2)}^{T}v_{w_{i}}) \cdot \sigma(-v_{n(w_{2}, 3)}^{T}v_{w_{i}})$$

可以看到，走左邊的話，維持正號，走右邊的好則加上負號，而整棵樹每個`leaf node`的機率維持總和為1，因為左右節點相加為1，$$ \sigma(v_{n}^{T}v_{w_{i}}) + \sigma(-v_{n}^{T}v_{w_{i}}) = 1 $$

### Negative Sampling
<hr>

$$ \textit{Objective} = \log \sigma({v}'_{w_{O}} \:^{T} v_{w_{I}}) + \sum_{i=1}^{k} \mathbb{E}_{w_{i} \sim P_{n}(w)} \left [ \log \sigma (-{v}'_{w_{i}} \:^{T} v_{w_{I}}) \right ] $$

對於`Skip-gram`來說，給定一個$ \textit{center word} = c$ 我們要預測$ \textit{context word} = o$，而使用`Full Softmax`的情況，分母需計算所有 $$ V $$ 需耗費大量時間，`Negative Sampling`改變了目標函數(Objective Function)，改成給定$\left ( c = \textit{orange}, o = \textit{juice} \right )$，若為正樣本則等於1，反之$\left ( c = \textit{orange}, o = \textit{bank} \right )$則等於0，變成一個`Binary Classification`問題，因此對於每次計算，原本 $ \left \| V \right \| $ 的計算量減少為 $ k+1 $，$ k $ 為負採樣的數量，通常小資料集設定為$5 \sim 20$，大資料集設定為$2 \sim 5$

*上式以`Skip-gram`為例，而`CBOW`同樣也能使用`Negative Sampling`來訓練，只不過是改成負採樣`center word`*

Best Choice of $$ P_{n}(w) $$ seems to `the Unigram Model` raised to the power of 3/4

### Subsampling Frequent Words
<hr>

$$ P(w_{i}) = 1 - \sqrt{\frac{t}{f(w_{i})}} $$

文章中常常有一些`停用字(stopwords)`，在skip-gram的訓練過程中，並不能賦予center word太多意思，舉例來說`Paris, France`跟`Paris, the`，反之，`the`也很難被周圍的字所表示，所以論文中透過上面的式子，設定一個機率來做`二次取樣(subsampling)`，把太高頻的字丟掉，經過實驗後，門檻值$$ t = 10^{-5} $$ 通常得到不錯的效果，透過這樣的步驟可以提升速度，並且增加罕見字(rare word)的準確率

### Reference

1. [第一週投影片](http://web.stanford.edu/class/cs224n/slides/cs224n-2020-lecture01-wordvecs1.pdf)、[筆記](http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes01-wordvecs1.pdf)
2. [Word2Vec Tutorial](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
3. [Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
4. [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)
5. [Negative Sampling by Andrew Ng](https://www.youtube.com/watch?v=TaZz_K2xJy8)
