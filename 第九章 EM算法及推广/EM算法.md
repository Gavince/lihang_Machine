## Em算法

### 1. 前言

​		EM算法是一种迭代算法,用于含有隐变量的概率模型参数的极大似然估计,或极大后验概率估计.EM算法的每次迭代由两部分组成:E步,求期望, M步:求极大.所以这一算法被称为期望极大算法,简称EM算法.

### 2. EM算法引入

**已知：**

​		三枚硬币（A,B,C）投掷正面的概率分别是（$\pi,p, q$），投掷原则如下图，最终投掷结果出现正面记作１，出现反面记作０；独立重复n次实验，假只能设观测到投掷硬币的结果，不能观测投掷硬币的过程．<font color = blue>问题：如何求解三硬币模型正面朝上的概率．</font>

![投硬币](/home/gavin/Machine/机器学习理论/code /第九章/投硬币.png)

**求解：**

三硬币模型的概率写作(参数：$\theta= (\pi, p, q)$：
$$
P(Y|\theta) = \sum_ZP(Y, Z|\theta) = \sum_ZP(Z|\theta)P(Y|Z,\theta)
$$
我们希望通过极大似然估计来求解最大化概率下的参数$\theta$, 但是这个问题没有解析解，如下：
$$
\hat \theta = \arg\max\limits_{\theta} logP(Y|\theta)
$$
所以，只有通过迭代的方法求解，EM算法就是可以用于求解这个问题的一种迭代算法．计算过程如下：

**E step:**计算模型参数$\pi^{(i)}, p^{(i)}, q^{(i)}$下观测数据$y_{i}$来自投掷硬币Ｂ的概率
$$
\mu^{i+1}=\frac{\pi (p^i)^{y_i}(1-(p^i))^{1-y_i}}{\pi (p^i)^{y_i}(1-(p^i))^{1-y_i}+(1-\pi) (q^i)^{y_i}(1-(q^i))^{1-y_i}}
$$
**M step:**计算模型参数新的估计值
$$
\pi^{i+1}=\frac{1}{n}\sum_{j=1}^n\mu^{i+1}_j
$$

$$
p^{i+1}=\frac{\sum_{j=1}^n\mu^{i+1}_jy_i}{\sum_{j=1}^n\mu^{i+1}_j}
$$

$$
q^{i+1}=\frac{\sum_{j=1}^n(1-\mu^{i+1})_jy_i}{\sum_{j=1}^n(1-\mu^{i+1}_j)}
$$

- **代码**

  ```python
  import numpy as np
  import math
  
  
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

- **代码结果**

  **初始化参数一:[0.5, 0.5, 0.5]**

  ```python
  em = EM(prob=[0.5, 0.5, 0.5])
  f = em.fit(data)
  next(f)
  ```

  ```
  init value: pro_A = 0.5, pro_B = 0.5, pro_C = 0.5
  ```

  ```python
  f.send(1)
  ```

  ```
  Interiation:1
  Pro_A:0.500000, Pro_B:0.600000, Pro_C:0.600000
  ```

  ```python
  f.send(2)
  ```

  ```
  Interiation:2
  Pro_A:0.500000, Pro_B:0.600000, Pro_C:0.600000
  ```

  ```python
  f.send(3)
  ```

  ```
  Interiation:3
  Pro_A:0.500000, Pro_B:0.600000, Pro_C:0.600000
  ```

  ```python
  next(f)
  ```
  
  ```python
  Interiation:4
  Pro_A:0.500000, Pro_B:0.600000, Pro_C:0.600000
              
              
  ```
  
- **初始化参数二:[0.4, 0.6, 0.7]**

  ```python
    #参数二
    em = EM(prob=[0.4, .6, .7])
    f2 = em.fit(data)
    next(f2)
  ```
  
  ```
    init value: pro_A = 0.4, pro_B = 0.6, pro_C = 0.7
  ```
  
  ```python
    next(f2)
  ```
  
  ```
    Interiation:1
    Pro_A:0.406417, Pro_B:0.536842, Pro_C:0.643243
  ```
  
  ```python
    f2.send(9)
  ```
  
  ```
    Interiation:8
    Pro_A:0.406417, Pro_B:0.536842, Pro_C:0.643243
  ```

- **结论：**

  ​		EM可以迭代的求解含有隐含条件下的参数，并且，从上面可以观察到，EM算法与初值的选择有关，选择不同的初值可能得到不同的参数估计，所以ＥＭ选择合适的初始化参数尤为重要．

### 3.EM算法

#### EM描述

​		***EM算法***是含有隐变量的概率模型极大似然估计或极大后验概率估计的迭代算法。含有隐变量的概率模型的数据表示为($\theta$ )。这里，$Y$是观测变量的数据，$Z$是隐变量的数据，$\theta$ 是模型参数。EM算法通过迭代求解观测数据的对数似然函数${L}(\theta)=\log {P}(\mathrm{Y} | \theta)$的极大化，实现极大似然估计。每次迭代包括两步：

- **$E-step$，**求期望，即求$logP\left(Z | Y, \theta\right)$ )关于$ P\left(Z | Y, \theta^{(i)}\right)$)的期望：
  $$
  Q\left(\theta, \theta^{(i)}\right)=\sum_{Z} \log P(Y, Z | \theta) P\left(Z | Y, \theta^{(i)}\right)
  $$

  称为$Q$函数，这里$\theta^{(i)}$是参数的现估计值；

- $M-step$，求极大，即极大化$Q$函数得到参数的新估计值：
  $$
  \theta^{(i+1)}=\arg \max _{\theta} Q\left(\theta, \theta^{(i)}\right)
  $$
  在构建具体的EM算法时，重要的是定义$Q$函数。每次迭代中，EM算法通过极大化$Q$函数来增大对数似然函数${L}(\theta)$。

- 重复以上步骤，直到收敛

#### EM算法理论

**问题：为什么EM算法能近似的实现对观测数据的极大似然估计？**

**解决问题：**

1). 极大化观测参数Ｙ关于参数$\theta$的对数似然函数
$$
\begin{aligned} L(\theta) &=\log P(Y | \theta)=\log \sum_{Z} P(Y, Z | \theta) \\ &=\log \left(\sum_{Z} P(Y | Z, \theta) P(Z | \theta)\right) \end{aligned}
$$
2). 使用EM算法迭代的思想，使得$L(\theta)$相较与上一次有所增加，即$L(\theta) > L(\theta^{i})$
$$
L(\theta)-L\left(\theta^{(i)}\right)=\log \left(\sum_{Z} P(Y | Z, \theta) P(Z | \theta)\right)-\log P\left(Y | \theta^{(i)}\right)
$$
3). 使用Jensen不等式得到下界
$$
\begin{aligned} L(\theta)-L\left(\theta^{(t)}\right) &=\log \left(\sum_{z} P\left(Y | Z, \theta^{(i)}\right) \frac{P(Y | Z, \theta) P(Z | \theta)}{P\left(Y | Z, \theta^{(i)}\right)}\right)-\log P\left(Y | \theta^{(i)}\right) \\ & \geqslant \sum_{Z} P\left(Z | Y, \theta^{(i)}\right) \log \frac{P(Y | Z, \theta) P(Z | \theta)}{P\left(Z | Y, \theta^{(i)}\right)}-\log P\left(Y | \theta^{(i)}\right) \\ &=\sum_{Z} P\left(Z | Y, \theta^{(i)}\right) \log \frac{P(Y | Z, \theta) P(Z | \theta)}{P\left(Z | Y, \theta^{(i)}\right) P\left(Y | \theta^{(i)}\right)} \end{aligned}
$$
令
$$
B\left(\theta, \theta^{(i)}\right) \triangleq L\left(\theta^{(i)}\right)+\sum_{Z} P\left(Z | Y, \theta^{(i)}\right) \log \frac{P(Y | Z, \theta) P(Z | \theta)}{P\left(Z | Y, \theta^{(i)}\right) P\left(Y | \theta^{(i)}\right)}
$$
则有
$$
L(\theta) \geqslant B(\theta, \theta^{i})
$$
这里B(θ, θ(i)) 是L(θ) 的一个下界，而且由的表达式可知
$$
L(\theta^{i}) = B(\theta^{i}, \theta^{i})
$$
所以任何可以使Ｂ(θ, θ(i)) 增大的参数都能使得$L(\theta)$增大

4). 极大化Ｂ得到新的参数 $\theta$
$$
\begin{aligned} \theta^{(i+1)} &=\arg \max _{\theta}\left(L\left(\theta^{(i)}\right)+\sum_{Z} P\left(Z | Y, \theta^{(i)}\right) \log \frac{P(Y | Z, \theta) P(Z | \theta)}{P\left(Z | Y, \theta^{(i)}\right) P\left(Y | \theta^{(i)}\right)}\right) \\ &=\arg \max _{\theta}\left(\sum_{Z} P\left(Z | Y, \theta^{(i)}\right) \log (P(Y | Z, \theta) P(Z | \theta))\right) \\ &=\arg \max _{\theta}\left(\sum_{Z} P\left(Z | Y, \theta^{(i)}\right) \log P(Y, Z | \theta)\right) \\ &=\arg \max _{\theta} Q\left(\theta, \theta^{(i)}\right) \end{aligned}
$$
所以ＥＭ算法是不断求解下界的极大化逼近求解对数似然函数极大化的算法．(<font color = red>本质</font>)

**图示：**

![](/home/gavin/Machine/机器学习理论/code /第九章/Em算法解释.png)

​		由图可以得知，EM算法在每次迭代中，通过不断增大下界的值来提升$L(\theta)$的值，并且，每一次迭代都需要重新计算Ｑ函数的值，进行下一次迭代，在这个过程中对数似然函数$L(\theta)$不断增大，除此之外，从图中可以推断<font color = red>出EM算法不能保证找到全局最优值</font>

### ４. EM算法在高斯混合模型学习中的应用

#### 高斯混合模型

**定义：**高斯混合模型是具有以下<font color = blue>概率分布</font>的模型：
$$
P(y | \theta)=\sum_{k=1}^{K} \alpha_{k} \phi\left(y | \theta_{k}\right)
$$
其中$a_k$是系数，$a_k>=0$，并$\sum_{k=1}^{K} a_k =1$, 其中$Φ(y|θ_k) $是高斯分布密度$(θ_k =(\mu_k，σ_k^2))$
$$
\phi\left(y | \theta_{k}\right)=\frac{1}{\sqrt{2 \pi} \sigma_{k}} \exp \left(-\frac{\left(y-\mu_{k}\right)^{2}}{2 \sigma_{k}^{2}}\right)
$$
称为第k个分模型．

**图释：**(图中每个点都由Ｋ个子模型中的某一个生成)

![preview](https://pic1.zhimg.com/v2-b1a0d985d1508814f45234bc98bf9120_r.jpg)

**求解参数：**

​		如果对于高斯混合模型直接使用极大似然估计来估计参数$\theta$，如下：
$$
\log L(\theta)=\sum_{j=1}^{N} \log P\left(y_{j} | \theta\right)=\sum_{j=1}^{N} \log \left(\sum_{k=1}^{K} \alpha_{k} \phi\left(y | \theta_{k}\right)\right)
$$
由上式可以得出结论，在高斯混合模型下的使用极大似然估计去是估计参数，是非常困难的，所以可以使用ＥＭ迭代的思想求解混合模型下的参数，此时要引入隐变量$\gamma_{jk}$,表示第j个观测数据来自第k个分模型：

$$
\gamma_{jk}=
  \begin{cases}
  1, &第j个观测来自第k个分模型\\
  0, &否则
  \end{cases}\\
  j=1,2,\dots,N; k=1,2,\dots,K; \gamma_{jk}\in\{0,1\}
$$
然后使用ＥＭ算法：

(1) 去参数的初始值开始迭代

(2) **E－step**:依据当前模型参数，计算模型k对观测数据$y_j$的响应度
$$
\hat{\gamma}_{j k}=\frac{\alpha_{k} \phi\left(y_{j} | \theta_{k}\right)}{\sum_{k=1}^{K} \alpha_{k} \phi\left(y_{j} | \theta_{k}\right)}
$$
(3) **M-step：**计算新一轮迭代的模型参数
$$
\hat{\mu}_{k}=\frac{\sum_{j=1}^{N} \hat{\gamma}_{k} y_{j}}{\sum_{j=1}^{N} \hat{\gamma}_{j k}}, \quad k=1,2, \cdots, K
$$

$$
\hat{\sigma}_{k}^{2}=\frac{\sum_{j=1}^{N} \hat{\gamma}_{k}\left(y_{j}-\mu_{k}\right)^{2}}{\sum_{j=1}^{N} \hat{\gamma}_{j k}}, \quad k=1,2, \cdots, K
$$

$$
\hat{\alpha}_{k}=\frac{\sum_{j=1}^{N} \hat{\gamma}_{j k}}{N}, \quad k=1,2, \cdots, K
$$

(4) 重复第(2)步和第(2)步，直达收敛．

- **代码**

  ```python
  import math
  import copy
  import numpy as np 
  import matplotlib.pyplot as plt
  
  #要给定一个初始值
  def init_data(Sigma, Mu1, Mu2, k, N):
      """
      Sigma:均方差
      mu1, mu2:各高斯模型的期望
      k: 几个高斯模型
      N: 数据量
      """
      global X
      global Mu
      global Expectations#初始化期望
      
      X = np.zeros((1, N))
      Mu = np.random.random(2)
      Expectations = np.zeros((N, k))
      
      #制造数据分布
      for i in range(0, N):
          if np.random.random(1) > 0.5:
              X[0, i] = np.random.normal()*Sigma + Mu1
          else:
              X[0, i] = np.random.normal()*Sigma + Mu2
   
      
  def e_step(Sigma, k, N):
      """
      计算期望
      """
      global Expectations
      global Mu
      global X
      
      for i in range(0, N):#遍历数据点
          Denom = 0
          #计算总和
          for j in range(0, k):
              Denom += math.exp((-1/(2 * float(Sigma**2)))*(float(X[0, i]-Mu[j])**2))
          #计算分子
          for j in range(0, k):
              
              Numer = math.exp((-1/(2 * float(Sigma**2)))*(float(X[0, i]-Mu[j])**2))
              
              Expectations[i, j] = Numer / Denom
              
  #     print("期望: {}".format(Expectations))
      
      
  def m_step(k, N):
      """
      计算期望最大化参数
      """
      global Expectations
      global X
      
      for j in range(0, k):
          Numer = 0
          Denom = 0
          for i in range(0, N):
              Numer += Expectations[i, j] * X[0, i]
              Denom += Expectations[i, j]
          Mu[j] = Numer / Denom
          
          
  def run(Sigma, Mu1, Mu2, k, N, inter_num, Epsilon):
      init_data(Sigma, Mu1, Mu2, k, N)
      Ex1 = []
      Ex2 = []
      
      for i in range(inter_num):
          Old_Mu = copy.deepcopy(Mu)
          e_step(Sigma, k, N)
          m_step(k,N)
          print("迭代 :{}, 期望 : {}".format(i, Mu))
          
          if sum(abs(Mu-Old_Mu)) < Epsilon:
              break
              
          Ex1.append(Mu[0])
          Ex2.append(Mu[1])
      return Ex1, Ex2, i
  
  if __name__ == "__main__":
      Ex1, Ex2, iter_num = run(6, 40, 20, 2, 1000, 1000, 0.0001)
      X_num = np.arange(0, iter_num, 1)
      plt.hist(X[0, :], 100)
      plt.savefig("高斯混合模型1")
      plt.show() 
  ```



- **输出结果:**

![输出结果1](/home/gavin/Machine/机器学习理论/code /第九章/输出结果1.png)

- **期望迭代图：**

  ```python
  plt.figure(figsize=(10, 6))
  plt.subplot(121)
  plt.plot(X_num, Ex1)
  plt.xlabel("迭代")
  plt.ylabel("期望")
  plt.title("期望迭代图")
  plt.savefig("Ex1")
  
  plt.subplot(122)
  plt.plot(X_num, Ex2)
  plt.xlabel("迭代")
  plt.ylabel("期望")
  plt.title("期望迭代图")
  plt.savefig("Ex2");
  ```

- **输出结果:**

  ![Ex2](/home/gavin/Machine/机器学习理论/code /第九章/Ex2.png)

### 参考链接
[Em算法](https://blog.csdn.net/zouxy09/article/details/8537620)
[EM算法与高斯混合模型](https://www.cnblogs.com/jiangxinyang/p/9278608.html)
[高斯混合模型EM](https://mp.weixin.qq.com/s?src=11&timestamp=1566619458&ver=1809&signature=ijqrcwAh0WUXbXipOApdW4md-B-LY9*WqoNySbjh*gNSiDPp181oU0lJT0eJwm2RipMFDWGX9hiaRvQpsq8DuIcKOGZesWPoEr0ySOzTh4cmSdBvaD9REOwIDeaLrOGV&new=1)
[聚类](https://www.jianshu.com/p/cd9bc01b694f)
[GMM](https://zhuanlan.zhihu.com/p/30483076)