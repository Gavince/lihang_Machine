## Em算法

### 0.参考

>[Em算法](https://blog.csdn.net/zouxy09/article/details/8537620)
>
>[EM算法与高斯混合模型](https://www.cnblogs.com/jiangxinyang/p/9278608.html)
>
>[高斯混合模型EM](https://mp.weixin.qq.com/s?src=11&timestamp=1566619458&ver=1809&signature=ijqrcwAh0WUXbXipOApdW4md-B-LY9*WqoNySbjh*gNSiDPp181oU0lJT0eJwm2RipMFDWGX9hiaRvQpsq8DuIcKOGZesWPoEr0ySOzTh4cmSdBvaD9REOwIDeaLrOGV&new=1)
>
>[聚类](https://www.jianshu.com/p/cd9bc01b694f)

### 1. 前言

>​	E算法是一种迭代算法,用于含有隐变量的概率模型参数的极大似然估计,或极大后验概率估计.EM算法的每次迭代由两部分组成:E步,求期望, M步:求极大.所以这一算法被称为期望极大算法,简称EM算法.

### 2.EM算法

#### 2.1EM算法的引入

>问题:三枚银币投掷问题, 求解每一枚银币的概率估计参数?

1．EM算法是含有隐变量的概率模型极大似然估计或极大后验概率估计的迭代算法。含有隐变量的概率模型的数据表示为$\theta$ )。这里，$Y$是观测变量的数据，$Z$是隐变量的数据，$\theta$ 是模型参数。EM算法通过迭代求解观测数据的对数似然函数${L}(\theta)=\log {P}(\mathrm{Y} | \theta)$的极大化，实现极大似然估计。每次迭代包括两步：

$E$步，求期望，即求$logP\left(Z | Y, \theta\right)$ )关于$ P\left(Z | Y, \theta^{(i)}\right)$)的期望：

$$Q\left(\theta, \theta^{(i)}\right)=\sum_{Z} \log P(Y, Z | \theta) P\left(Z | Y, \theta^{(i)}\right)$$
称为$Q$函数，这里$\theta^{(i)}$是参数的现估计值；

$M$步，求极大，即极大化$Q$函数得到参数的新估计值：

$$\theta^{(i+1)}=\arg \max _{\theta} Q\left(\theta, \theta^{(i)}\right)$$

在构建具体的EM算法时，重要的是定义$Q$函数。每次迭代中，EM算法通过极大化$Q$函数来增大对数似然函数${L}(\theta)$。

2．EM算法在每次迭代后均提高观测数据的似然函数值，即

$$P\left(Y | \theta^{(i+1)}\right) \geqslant P\left(Y | \theta^{(i)}\right)$$

在一般条件下EM算法是收敛的，但不能保证收敛到全局最优。

3．EM算法应用极其广泛，主要应用于含有隐变量的概率模型的学习。高斯混合模型的参数估计是EM算法的一个重要应用，下一章将要介绍的隐马尔可夫模型的非监督学习也是EM算法的一个重要应用。

4．EM算法还可以解释为$F$函数的极大-极大算法。EM算法有许多变形，如GEM算法。GEM算法的特点是每次迭代增加$F$函数值（并不一定是极大化$F$函数），从而增加似然函数值。

5. EM算是通过不断极大化下界,从而得到极大化后的参数,EM算法的直观解释如下图(P160):![](/home/gavin/Machine/机器学习理论/code /第九章/Em算法解释.png)

#### 2.2 代码解析

>**E step**:
>$$
>\mu^{i+1}=\frac{\pi (p^i)^{y_i}(1-(p^i))^{1-y_i}}{\pi (p^i)^{y_i}(1-(p^i))^{1-y_i}+(1-\pi) (q^i)^{y_i}(1-(q^i))^{1-y_i}}
>$$
>**M step:**
>$$
>\pi^{i+1}=\frac{1}{n}\sum_{j=1}^n\mu^{i+1}_j
>$$
>
>$$
>p^{i+1}=\frac{\sum_{j=1}^n\mu^{i+1}_jy_i}{\sum_{j=1}^n\mu^{i+1}_j}
>$$
>
>$$
>q^{i+1}=\frac{\sum_{j=1}^n(1-\mu^{i+1})_jy_i}{\sum_{j=1}^n(1-\mu^{i+1}_j)}
>$$

```python
class EM:
    """
    实现EM算法
    """
    def __init__(self, prob):
        self.pro_A, self.pro_B, self.pro_C = prob
        
    def pmf(self, i):
        """
        计算Estep
        """
        pro_1 = self.pro_A * math.pow(self.pro_B, data[i])*math.pow(
        (1 - self.pro_B), (1-data[i]))
        
        pro_2 = (1 - self.pro_A)*math.pow(self.pro_C, data[i])*math.pow(
        (1-self.pro_C), (1-data[i]))
        
        return pro_1/(pro_1+pro_2)
    
    def fit(self, data):
        """
        寻找参数最优解
        m-step
        """
        count = len(data)
        print("init value: pro_A = {}, pro_B = {}, pro_C = {}".format(
        self.pro_A, self.pro_B, self.pro_C
        ))
        #迭代d次
        for d in range(count):
            #1.计算所有的期望
            _ = yield
            _pmf = [self.pmf(k) for k in range(count)]
            
            pro_A = 1/count * sum(_pmf)
            pro_B = sum([_pmf[k] * data[k] for k in range(count)])/sum(
                [_pmf[k] for k in range(count)])
            pro_C = sum([((1 - _pmf[k])*data[k]) for k in range(count)])/sum(
                [(1-_pmf[k]) for k in range(count)])
            
            
            self.pro_A = pro_A
            self.pro_B = pro_B
            self.pro_C = pro_C
            print("Interiation:%d" %(d+1))
            print("Pro_A:%f, Pro_B:%f, Pro_C:%f" % (self.pro_A, self.pro_B, self.pro_C))
            
if __name__ == "__main__":
    data = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]
```

<font color= #0099ff>**参数一**</font>
![](/home/gavin/Machine/机器学习理论/code /第九章/参数1.png)

<font color= #0099ff>**参数二**</font>

![](/home/gavin/Machine/机器学习理论/code /第九章/参数二.png)

<font color = red>**结论:**</font>

​	EM算法与初值的选择有关.选择不同的 可能得到不同的参数估计.

### 3. EM算法在高斯混合模型学习中的应用

#### 3.1 高斯混合模型推导

![高斯1](/home/gavin/Machine/机器学习理论/code /第九章/高斯1.png)

![高斯2](/home/gavin/Machine/机器学习理论/code /第九章/高斯2.png)