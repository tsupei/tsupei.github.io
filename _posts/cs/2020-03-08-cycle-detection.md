---
layout: article_post
title:  "Cycle Detection Algorithm"
categories: cs
tags: cs cycle-detection 
excerpt_separator: <!--more-->
---

在寫[LeetCode 287](https://leetcode.com/problems/find-the-duplicate-number/)時遇到了這個問題，看了解答後，發覺到有一個叫做`龜兔賽跑`的解法，看起來非常神奇，但其實背後的想法蠻簡單的，這篇文章稍微整理一下

<!--more-->

#### 龜兔賽跑(Floyd's Tortoise and Hare)

龜和兔同時從起點出發，每次龜走一步，而兔走兩步，如果圖是帶循環的話，那麼兔子一定會追上烏龜，而且具有以下的數學關係，我們將兔子的行進距離記做 $ D_{hare} $ ，烏龜的行近距離記做 $ D_{tortoise} $，循環長度為 $ L_{cycle} $

1. 兔子追上烏龜時，兔子的行進距離為烏龜的兩倍 $ D_{hare} = 2 \, D_{tortoise} $
2. 烏龜的行進距離為循環長度的倍數 $ D_{tortoise} = n \, L_{cycle} $

試著推導(2) 

$$ D_{hare} - D_{tortoise} = n \, L_{cycle} = D_{tortoise} $$

以上只是第一步驟，接下來我們將兔子留在原處，烏龜回到起點，然後烏龜跟兔子每次都各走一步，當龜兔相遇時，即是循環起點

為什麼成立呢，假設龜兔第一回合相遇在離循環起點 $x$處，而循環起點距離原點為 $L_{cycle} - x$，所以當龜兔都再各走$L_{cycle} - x$的距離，就會在循環起點相遇了

我們回到[LeetCode 287](https://leetcode.com/problems/find-the-duplicate-number/)，可以把輸入的序列視為一個有向圖的表示，index為節點id，value為指向的節點id，如此一來發生循環的地方表示有`duplicate number`的出現

```python
class Solution(object):
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        tortoise = 0
        hare = 0
        while True:
            tortoise = nums[tortoise]
            hare = nums[nums[hare]]
            if hare == tortoise:
                break
        tortoise = 0
        while tortoise != hare:
            tortoise = nums[tortoise]
            hare = nums[hare]
        return tortoise
```

#### Reference

1. [演算法筆記](http://www.csie.ntnu.edu.tw/~u91029/Function.html)
2. [LeetCode 287](https://leetcode.com/problems/find-the-duplicate-number/)

