---
layout: article_post
title:  "Dirichlet Distribution"
categories: math
tags: Probability Dirichlet-Distribution LDA 
excerpt_separator: <!--more-->
---


LDA Topic Model中，用到了一些機率分布，像是`Multinomial Distribution`, `Beta Distribution`, `Dirichlet Distribution`，不過對於這些分布並沒有一個完整的了解，我們參考`PRML`這本書並在這篇文章整理複習一下吧！

<!--more-->

### Distribution
---

在進入到Dirichlet Distribution之前，我們先複習一下一些常見的Distribution

1. Bernoulli Distribution
2. Binomial Distribution

#### Bernoulli Distribution

假設我們擲一個硬幣一次，在擲出正面的機率為$$\mu$$的前提下，得到正面$$x=1$$的機率為

$$Bern(x\mid \mu)=\mu^{x}(1-\mu)^{1-x}$$

當我們給定$$\mu$$時

$$Bern(x=1\mid \mu) = \mu$$

$$Bern(x=0\mid \mu) = 1-\mu$$

*注意: Bernoulli只看一次的機率，而$$x\in{0,1}$$*

#### Binomial Distribution

如果我們進行$$N$$次的Bernoulli Distribution，就會得到Binomial Distribution

$$Bin(m\mid N,\mu)=\binom{N}{m}\mu^{m}(1-\mu)^{N-m}$$

### Conjugacy
---

關於共軛先驗、共軛分布的概念，這篇[博客](https://blog.csdn.net/baimafujinji/article/details/51374202)寫的相當清楚，建議先進去看看，然後接著我會稍微整理我的理解

讓我們先看一下`Bayes' Theorem`

$$P(A\mid B) = \frac{P(B\mid A)P(A)}{P(B)}$$

$$P(A\mid B)$$我們稱為後驗機率(posterior)，$$P(A)$$是先驗機率(prior)，$$P(B\mid A)$$是似然機率(likelihood)

先驗機率是我們基於對現實世界的認知，得到的機率

例如在一個裝有100顆相同大小的球的箱子，其中有60顆紅色球、40顆黃色球，球上面編號1到100，1到60為紅色，61到100為黃色

抽到紅色球的機率我們會認為是$$0.6$$

而後驗機率則是，我們基於某證據B，得到的機率，假設$事件R表示抽到紅色球的事件，事件T代表抽到球編號大於50的事件$

今天我們抽到了一個編號大於50的事件，求是紅色球的機率為何

$$P(R\mid T)=\frac{P(T\mid R)P(R)}{P(T)} = \frac{\frac{1}{6}\times 0.6}{0.5} = \frac{1}{5}$$

我們可以看到這個例子中，先驗機率為0.6而後驗機率為0.2，在我們得知某個條件或者證據之下，求出的機率即為後驗機率

在看一個例子我們會更了解，我們拿上面博客中拿來說明共軛分佈的硬幣例子

假設我們今天擲了一個硬幣五次，得到三次正面，兩次反面，稱這個結果為事件$T$，而我們已知直到硬幣正面的機率$\theta$只可能是`0.5`或者`0.8`

而且$P(\theta = 0.5)=0.5, P(\theta = 0.8) = 0.5$

根據Bayes' Theorem，我們寫出以下式子

$$P(\theta=0.5 \mid T) = \frac{P(T \mid \theta=0.5) \times P(\theta =0.5)}{P(T)} = 0.30204$$

根據以上式子，本來硬幣擲到正面的機率是0.5的機率(也就是$P(\theta = 0.5)$)是0.5，但在有觀測結果$T$的前提下，$P(\theta = 0.5)$降低到0.30204
	

---

而如果今天，`prior`不是一個機率，而是一個分布的話，就稱為`先驗分布`

那這個`prior`要怎麼取呢？我們來看一下課本的解釋

{% include widget/excerpt.html text="If we choose a prior to be proportional to powers of μ and (1 − μ), then the posterior distribution, which is proportional to the product of the prior and the likelihood function, will have the same functional form as the prior. This property is called conjugacy" %}

我們取跟`likelihood function`指數成正比的`prior`，今天要討論的`Beta分布`，其實就是`Binominal分布`的`共軛分布`

以剛剛的擲硬幣例子來說，如果我們改成用Beta分布來表示硬幣擲到正面的機率，而後驗分布也會是一個Beta分布(證明請看上面參考博客)

$$P(\theta \mid X) = Beta(\theta \mid a+3, b+2)$$

我理解共軛性(conjugacy)的重要性是，我們可以經由觀測數據來修正原本的`先驗分布`

假設一開始Beta分布的高峰值在0.8，但經過我們的觀測後，修正了原本的Beta分布高峰值在0.7

回到課本中，我們現在知道了，我可以經由`Binominal distribution`跟先驗分布的`Beta Distribution`相乘而得到後驗分布

$$P(μ\mid m, l, a, b) \propto μ^{m+a−1}(1 − μ)^{l+b−1}, m為正面次數, l為反面次數, a和b為prior的係數$$

### Multinomial distribution and Dirichlet distribution
---

我們直接來看Multinomial distribution的公式

$$Mult(m_{1},m_{2},...,m_{K} \mid \mu ,N) = \binom{N}{m_{1} m_{2} ... m_{k}} \prod_{k=1}^{K} u_{k}^{m_{k}}$$

其中$K$表示所有可能的$state$，在`Binomial Distribution`只有兩種可能，但在這邊有$ K $種$ state $

我們用$ m_{k} $表示該$ state $下的出現次數，$ \mu_{k} $則是該$state$的機率，另外所有$m_{k} $的總和即為$ N $

$$\sum_{k=1}^{K} m_{k} = N$$

而`Dirichlet Distribution`就只是`Multinomial Distribution`的先驗分布`prior`

所以我們要取跟上面的`Multinomial Distribution`相同形式的函式，並且用$\alpha $當作參數控制

$$p(\mu \mid \alpha) \propto \prod_{k=1}^{K} \mu_{k}^{\alpha_{k-1}}$$

經過normalize之後，我們得到

$$Dir(\mu \mid \alpha) = \frac{\Gamma(\alpha_{0})}{\Gamma(\alpha_{1})...\Gamma(\alpha_{K})} \prod_{k=1}^{K} \mu_{k}^{\alpha_{k}-1}$$

$$while \space \alpha_{0} = \sum_{k=1}^{K} \alpha^{k}$$

再來，我們只要把`Likelihood Function`跟`Prior Function`相乘後，就能得到我們的`posterior`，而posterior也會是`Dirichlet Function`

只是參數有變化

$$p(\mu \mid D, \alpha) = Dir(\mu \mid \alpha + m) = \frac{\Gamma(\alpha_{0}+N)}{\Gamma(\alpha_{1}+m_{0})...\Gamma(\alpha_{K}+m_{k})} \prod_{k=1}^{K} \mu_{k}^{\alpha_{k}+m_{k}-1}$$

所以，`Dirichlet Distribution`的參數從原來的$\alpha$變成了$\alpha + m$，$m$是觀察到各個$state$的次數

### Reference
---

1. PRML
2. [博客](https://blog.csdn.net/baimafujinji/article/details/51374202)











