---
layout: article_post
title:  "DistilBERT"
categories: nlp 
excerpt_separator: <!--more-->
tags: BERT DistilBERT Distillation 
---

今天要介紹的是 Distilled BERT，是由hugging face在今年(2019)所提出的一篇論文，相信用pytorch來做BERT的朋友一定對 hugging face 不陌生，他們在 Github上開源的[transformers專案](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwiz-bCDg7_nAhVhJaYKHT7sCxEQFjAAegQICBAC&url=https%3A%2F%2Fgithub.com%2Fhuggingface%2Ftransformers&usg=AOvVaw2tFvUqz2_mjqk4l73AGqMY)提供許多Pytorch用戶可以方便的使用BERT相關模型

回到今天的主題，Distill的意思是蒸餾，我們可以從字面上猜測，我們要從一個很大的模型，蒸餾成比較小的模型，也可以用一種角度想，我們讓大的模型當作小的模型的老師，而小模型這個學生，只會盡可能的學老師的每個動作

大致上Distilled BERT的思想就是這樣簡單，根據作者的實驗數據，DistilBERT的參數大約只有BERT的40%，而速度快了60%，並保有一定一精準度

<!--more-->

### 方法
---

作者提出的DistilBERT的架構，層數為BERT的一半，這邊的重點其實是Loss Function，訓練採用的Loss Function為三種的結合(`Triple Loss`)

$$ L_{ce} = \sum_{i}t_{i} \times log(s_{i}) $$

$$ p_{i} = \frac{exp(z_{i} / T)}{ \sum_{j} exp(z_{j} / T)}$$

1. `Distillation Loss`: 從圖中可以看出，當BERT預測($t_{i}$)越高，而DistilBERT預測($s_{i}$)越低，產生的Loss就會越高，而機率是通過 softmax-temperature (Hinton et al., 2015) 計算
2. `Masked Language Modeling Loss`: 參考BERT
3. `Cosine Embedding Loss`: 用於讓DistilBERT能趨向產生跟BERT更像的hidden vector

#### Student Architecture

1. `Linear layer and Layer Normalization`: 已被高度優化且證明有效(Highly optimized)，因此不更動
2. `The size of hidden vector of last layer`: 根據實驗發現減少 hidden vector size 並不太影響效能
3. `Number of layers`: 影響效能、推理速度較多，因此作者注重在此參數

#### Student Initialization

因爲層數為BERT的一半，直接使用BERT的參數當作初始值

#### Compute Power

8 16GB V100 GPUs for approximately 90hrs on Toronto Book Corpus(原先BERT使用的資料集)

### Evaluation
---

- GLUE: DistilBERT retrain 97% of BERT performance，有些甚至超越BERT Base
- IMDb accuracy: BERT(93.46) DistilBERT(92.82)
- Speed(A full pass of GLUE task): BERT(668s) ELMo(895s) DistilBERT(410s)

### On device computation
---

將容量縮小後，也可以直接將模型放在設備端(i.e. APP)，論文中的DistilBERT大約207MB大

1. Building a mobile application for QA
2. 207 MB, could be further reduced with quantization

### 運行在下游任務(Downstream tasks)
---

{% include widget/excerpt.html 
text="We also studied whether we could add another step of distillation during the adaptation phase by fine-tuning DistilBERT on SQuAD using a BERT model previously fine-tuned on SQuAD as a teacher for an additional term in the loss (knowledge distillation). In this setting, there are thus two successive steps of distillation, one during the pre-training phase and one during the adaptation phase." %}

---
從上面的段落看來，在問答任務(屬於一種下游任務)中，將DistilBERT再跟Fine-tuned BERT進行一次蒸餾，就可以達到不錯效果

### Conclussion
---

DistilBERT整體來說很適合使用在應用層面，我們只需完成蒸餾的動作，就可以大幅增進推理速度，以及減少模型大小，可以減低伺服器的成本

### Reference

1. [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
2. [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108)
