### 前言

>   **问题的引入**
>
> **条件随机场**是给定一组输入随机变量条件下另一组输出随机变量的条件概率分布模型，其特点是假设暑促的随机变量构成马尔科夫随机场．

### 预备知识

#### 概率图模型

​		概率图模型是使用图来表示概率分布．如下图，其中图的结点表示随机变量，边表示相互连接的结点之间的依赖关系．

![概率图模型](/home/gavin/Machine/机器学习理论/code /第十一章/概率图模型.png)

#### 成对马尔科夫性

​		给定随机变量组$Y_O$的条件下随机变量$Y_u$和$Y_v$是条件独立的(下图中$o1,o2,o3$构成随机变量组$Ｏ$,$u,v$是<font color = red>**任意没有边连接的结点**</font>)
$$
P(Y_u,Y_v|Y_O)=P(Y_u|Y_O)P(Y_v|Y_O)​
$$
![成对马尔科夫性](/home/gavin/Machine/机器学习理论/code /第十一章/成对马尔科夫性.png)

#### 局部马尔科夫性

​		给定随机变量组$Y_W$的条件下随机变量$Y_v$与随机变量组$Y_O$是独立的（$w1, w2, w3$表示与结点$v$有边连接的结点，构成随机变量组$W$, O是$w, v$之外的<font color = red>**所有结点**</font>）
$$
P(Y_v,Y_O|Y_W)=P(Y_v|Y_W)P(Y_O|Y_W)
$$


![局部马尔科夫性](/home/gavin/Machine/机器学习理论/code /第十一章/局部马尔科夫性.png)

#### 全局马尔科夫性

​		给定随机变量组$Y_C$的条件下随机变量组$Y_A$和$Y_B$是条件独立的（A, B是无向图Ｇ中被集合C分开的<font color = red>**任意点的集合**</font>）
$$
P(Y_A,Y_B|Y_C)=P(Y_A|Y_C)P(Y_B|Y_C)
$$
![全局马尔科夫性](/home/gavin/Machine/机器学习理论/code /第十一章/全局马尔科夫性.png)

#### 概率无向图模型

​		设在一个概率图模型中，随机变量组成联合概率分布$P(Y)$，如果满足**成对,局部和全局马尔科夫性质**，则称此联合概率分布为概率无向图模型或马尔科夫随机场．

#### 团和最大团

​		无向图中<font color = red>**任何两个结点的均有边连接的结点子集**</font>称为团，若团中不能再加入任何一个结点构成一个更大团，则称该团为最大团．下图中$\{Y1, Y2\},\{Y1, Y3\},\{Y1, Y4\},\{Y4, Y5\},\{Y2, Y5\},\{Y3, Y4\}$称为团，$\{Y1, Y3,Y4\}$称为最大团．

![团和最大团](/home/gavin/Machine/机器学习理论/code /第十一章/团和最大团.png)

#### MRF因子分解

​	将概率无向图模型的联合概率分布表示为其**最大团**上的随机变量的函数的乘积形式的操作，称为概率无向图模型的因子分解(factorization)，概率无向图模型的最大特点就是**易于因子分解**．其中Z是规范化因子，使得最后的概率属于$[0, 1]$.
$$
P(Y)=\frac{1}{Z} \prod_{C} \Psi_{C}\left(Y_{C}\right)
$$

$$
Z=\sum_{\left(Y_{1}, Y_{}2, Y_{3}, \ldots, Y_{N}\right)}\left\{\prod_{C} \Psi_{C}(Y_{C})\right\}
$$

### 线性条件随机场

#### 定义

设$X=(X_1,X_2,\cdots,X_n)，Y=(Y_1,Y_2,\cdots,Y_n)$均为线性链表示的随机变量序列，若在给定随机变量序列$X$的条件下，随机变量序列$Y$的条件概率分布$P(Y|X)$构成条件随机场，即满足马尔可夫性
$$
P(Y_i|X,Y_1,\cdots,Y_{i-1},Y_{i+1},\cdots,Y_n)=P(Y_i|X,Y_{i-1},Y_{i+1})\\
i=1,2,\cdots,n (在i=1和n时只考虑单边)
$$
则称$P(Y|X)$为线性链条件随机场。**在标注问题中**，$X$表示输入观测序列， $Y$表示输出标记序列或状态序．如下图

![线性条件随机场](/home/gavin/Machine/机器学习理论/code /第十一章/线性条件随机场.png)

#### 特征函数

**<font color = blue>对于这一部分的理解，请先阅读</font>**[[条件随机场(CRF)的理解]](https://www.cnblogs.com/JohnRain/p/9250502.html)

线性链条件随机场的参数化形式
$$
P(y|x)=\frac{1}{Z(x)}\exp\left(\sum\limits_{i,k}\lambda_kt_k(y_{i-1},y_i,x,i)+\sum_{i,l}\mu_ls_l(y_i,x,i)\right)
$$
其中

$t_k$是定义在**边**上的特征函数，称为转移特征

$s_l$是定义在**结点**上的特征函数，称为状态特征

注意到这种表达就是不同特征的加权求和形式，$t_k,s_l$都依赖于位置，是局部特征函数，此外，<font color = red>k和l不是确定的值，是根据具体问题设置的值</font>。

**例题11.1 **（题目见课本P196）

- 代码

  ```python
  from numpy import *
  
  #Tx 表示时间节点，　
  
  #转移特征(举例说明：T1[1][1]表示在时刻１时，状态y1转向y1的概率)
  T1 = [[0.6, 1], [1, 0]]
  T2 = [[0, 1], [1, 0.2]]
  
  #状态特征S(举例说明：S[1] 表示在时刻S1[1]时，状态为y1时的概率)
  S0 = [1, 0.5]
  S1 = [0.8, 0.5]
  S2 = [0.8, 0.5]
  
  #此时输入观测序列Y
  Y = [1, 2, 2]
  
  #将索引值对应到从零开始
  Y = array(Y)-1
  p = exp(S0[Y[0]])
  for i in range(1, len(Y)):
      p *= exp((eval("S%d" % i)[y[i]] + eval("T%d" % i)[Y[i-1]][Y[i]]))
      
  print(p)
  print(exp(3.2))
  ```

- 输出结果

  ```python
  24.532530197109345
  24.532530197109352
  ```

#### 简化形式

**原式：**
$$
P(y|x)=\frac{1}{Z(x)}\exp\left(\sum\limits_{i,k}\lambda_kt_k(y_{i-1},y_i,x,i)+\sum_{i,l}\mu_ls_l(y_i,x,i)\right)
$$
**简化：**

**已知**：

​		输出序列：$y = \{y_{1}, y_{2}, y_{3}, y_{4}, \dots, y_{T}\}^T$

​		输入序列：$x = \{x_{1}, x_{2}, x_{3}, x_{4},\dots,x_{T}\}^T$

**简化过程**：

1. 向量化参数$\lambda和\mu$

   令：$\lambda = \{\lambda_1, \lambda_2,\dots, \lambda_k\}^T$

   ​	$\mu = \{\mu_1, \mu_2, \dots, \mu_T\}^T$

$$
P(y|x)=\frac{1}{Z(x,\lambda, \mu)}\exp\sum_{i = 1}^{T}\left(\lambda^Tt_k(y_{i-1},y_i,x,i)+\mu^Ts_l(y_i,x,i)\right)
$$

 2. 向量化转移特征和状态特征

    令：$t(y_{i-1}, y_{i}, x])=\{t_1, t_2, t_3, \dots, t_k\}^T$

    ​	$s(y_i, x) =\{s_1, s_2, s_3, \dots, s_l\}^T $
    $$
    P(y|x)=\frac{1}{Z(x,\lambda, \mu)}\exp\sum_{i = 1}^{T}(\lambda^T*t+\mu^T*s)
    $$

3. 向量化时间序列

    令：$\theta = \{\lambda, \mu\}^T$

    ​	$H = \{\sum_{i=1}^{T}*t, \sum_{i=1}^T * s\}^T$	

    向量化结果：
    $$
    P(Y=y|X=x) = \frac{1}{Z(x,\lambda, \mu)}\exp(\theta^T*H(y_i, y_{i-1}))
    $$

### 条件随机场的基本问题

#### 概率计算问题

- **求解问题:**$P(y_i = i|x)$？

- **解决方法:**前向后向算法

  **联合概率积分求边缘概率:**

  ​		对于$P(y_i|x)$的概率求解，首先可以考虑对联合概率积分求和，从而得到边缘概率分布，即对$P(y|x)$,从$<1,t-1>和<t+1, T>$时刻进行积分对相应的y积分，从而得到$P(y_i|x)$的概率密度函数．

  **变量消除法:**

  ​		下图表示了势函数表达式与所关联状态的关系．

  ![变量消除法](/home/gavin/Machine/机器学习理论/code /第十一章/变量消除法.png)

  **推导**：

  1)左半部分求解

  令在**t**时刻：
  $$
  \alpha_t(i) = y_0, y_1, y_2, \dots,y_{t-1},(y_t = i   左半部分势函数）
  $$
  则在**t-1**时刻：
  $$
  \alpha_{t-1}(j) = y_0, y_1, y_2, \dots, (y_{t-1} = j左半部分势函数)
  $$
  

  最终的$\alpha_t(i):$
  $$
  \alpha_t(i) = \sum_{j}\psi_t(y_{t-1}=j, y_t= i, x) * \alpha_{t-1}(j)
  $$
  2)右半部分求解

  令在时刻他**t**时刻：
  $$
  \beta_{t}(i) = (y_t(i)右半部分势函数),y_{t+1}, y_{t+2},\dots,y_{T}
  $$
  则在**t+1**时刻：
  $$
  \beta_{t+1}(j) =(y_{t+1}右半部分势函数), y_{t+2}, \dots, y_{T}
  $$
  最终的$\beta_{t}(i)$
  $$
  \beta_t(i) = \sum_{y_{t+1}}\psi_{t+1}(y_t=i, y_{t+1}=j, x),\beta_{t+1}(j)
  $$
  3）P(y_i|x)预测概率:
  $$
  P(y_i|x) = \frac{1}{Z}\beta_t(i)*\alpha_t(i)
  $$

#### 学习问题

#### 预测问题





## 参考链接

[如何轻松愉快地理解条件随机场（CRF）？](https://blog.csdn.net/dcx_abc/article/details/78319246)

[[条件随机场（CRF）的理解]](https://www.cnblogs.com/JohnRain/p/9250502.html)

[条件随机场PPT](https://wenku.baidu.com/view/69e8fc1afad6195f312ba620.htmlhttps://wenku.baidu.com/view/69e8fc1afad6195f312ba620.html)