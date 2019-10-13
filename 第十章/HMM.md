### 0.导读

**马尔科夫链**

>​		随机过程有两个维度的不确定性。马尔可夫为了简化问题，提出了一种简化的假设，即随机过程中各个状态$s_t$的概率分布，只与它的前一个状态$s_{t-1}$有关, 即$P(s_t|s_1, s_2, s_3, \dots,s_{t-1})=P(s_t|s_{t-1})$
>
>​		这个假设后来被称为**马尔可夫假设**，而符合这个假设的随机过程则称为**马尔可夫过程**，也称为**马尔可夫链**。
>
>数学之美，吴军

### 1. 隐马尔科夫模型

> ​		隐马尔可夫模型是关于时间序列的概率模型，描述由一个隐藏的马尔科夫链随机生成不可观测的状态的随机序列，再由各个状态生成一个观测而产生观测序列的过程．

#### 1.1 一个模型

![](/home/gavin/Machine/机器学习理论/code /第十章/HMM图释.png)

> ​		隐马尔科夫模型由初始概率分布,状态转移概率分布以及观测概率分布确定，参数定义如下：
>
> $I=i_{1} i_{2} \cdots i_{T}$   --->  状态序列　				$Q = \{q_{1}, q_{2}, q_{3}, \cdots,q_{N}\}$ ---> 状态值集合
> $O = O_{1},O_{2},\cdots, O_{T}$ --->观测序列　　　　$V = \{V_{1}, V_{2}, V_{3}, \cdots, V_{M}\}$ --->观测值集合
> (i)   $   \lambda=(\pi, A, B)$
> $ \pi :$初始化状态矩阵
> $A: [a_{ij}]$ -->状态转移矩阵，　$a_{i j}=P\left(i_{t+1}=q_{j} | i_{t}=q_{i}\right)$
> $B: [b_{j}\left(k\right)]$　--->发射矩阵， $b_{j}(k)=P\left(O_{t}=V_{k} | {i}_{t}=q_{j}\right)$
#### 1.2 两个假设

**齐次马尔科夫性假设:**
$$
P(i_t|i_{t-1},o_{t-1},\dots,i_1,o_1) = P(i_t|i_{t-1}), t=1,2,\dots,T
$$

   >假设隐藏的马尔可夫链在**任意时刻$t$的状态**$\rightarrow i_t$
   >
   >只依赖于其前一时刻的状态$\rightarrow i_{t-1}$
   >
   >与其他时刻的状态 $\rightarrow i_{t-1, \dots, i_1}$
   >
   >及观测无关 $\rightarrow o_{t-1},\dots,o_1$
   >
   >也与时刻$t$无关 $\rightarrow t=1,2,\dots,T$

**观测独立性假设：**
$$
P(o_t|i_T,o_T,i_{T-1},o_{T-1},\dots,i_{t+1},o_{t+1},i_t,i_{t-1},o_{t-1},\dots,i_1,o_1)=P(o_t|i_t)
$$

   > 假设**任意时刻$t$的观测**$\rightarrow o_t$
   >
   > 只依赖于该时刻的马尔可夫链的状态 $\rightarrow i_t$
   >
   > 与其他观测  $\rightarrow o_T,o_{T-1},\dots,o_{t+1},o_{t-1},\dots,o_1$
   >
   > 及状态无关 $\rightarrow i_T,i_{T-1},\dots,i_{t+1},i_{t-1},\dots,i_1$

#### 1.3 三个问题

1. 概率计算问题
   输入: 模型$\lambda=(A,B,\pi)$, 观测序列$O=(o_1,o_2,\dots,o_T)$
   输出: $P(O|\lambda)$

1. 学习问题
   输入: 观测序列 $O=(o_1,o_2,\dots,o_T)$
   输出: 输出$\lambda=(A,B,\pi)$

1. 预测问题, 也称为解码问题(Decoding)
   输入: 模型$\lambda=(A,B,\pi)$, 观测序列$O=(o_1,o_2,\dots,o_T)$ 
   输出: 状态序列 $I=(i_1,i_2,\dots,i_T)$

   因为状态序列是隐藏的，不可观测的，所以叫解码。
### 2．三个基本问题算法
>There are three fundamental problems for HMMs:
>
>- Given the model parameters and observed data, estimate the optimal sequence of hidden states.
>- Given the model parameters and observed data, calculate the likelihood of the data.
>- Given just the observed data, estimate the model parameters.
>
>The first and the second problem can be solved by the dynamic programming algorithms known as the Viterbi algorithm and the Forward-Backward algorithm, respectively. The last one can be solved by an iterative Expectation-Maximization (EM) algorithm, known as the Baum-Welch algorithm.
>
>---hhmlearn
#### 2.1 Evaluation: Given $\lambda$, 求$P(O|\lambda)$

**(1) Forward 算法:**

- 算法推导

![](/home/gavin/Machine/机器学习理论/code /第十章/forward算法.png)

- 算法具体流程

  (1)初值

  >    $$
  >    \alpha_1(i)=\pi_ib_i(o_1), i=1,2,\dots,N
  >    $$
  >    观测值$o_1$, $i$的含义是对应状态$q_i$
  >
  >    这里$\alpha$ 是$N$维向量, 和状态集合$Q$的大小$N$有关系. $\alpha$是个联合概率.

  (2)递推

  >    $$
  >    \alpha_{t+1}(i) = \left[\sum\limits_{j=1}^N\alpha_t(j)a_{ji}\right]b_i(o_{t+1})\color{black}, \   i=1,2,\dots,N, \ t = 1,2,\dots,T-1
  >    $$
  >    转移矩阵$A$维度$N\times  N$,  观测矩阵$B$维度$N\times M$, 具体的观测值$o$可以表示成one-hot形式, 维度$M\times1$, 所以$\alpha$的维度是$\alpha = \alpha ABo=1\times N\times N\times N \times N\times M \times M\times N=1\times N$

  (3)终止

  >    $$
  >    P(O|\lambda)=\sum\limits_{i=1}^N\alpha_T(i)
  >    $$
  >    注意, 这里$O\rightarrow (o_1, o_2, o_3,\dots, o_t)$, $\alpha_i$见前面前向概率的定义$P(o_1,o_2,\dots,o_t,i_t=q_i|\lambda)$, 所以, 对$i$求和能把联合概率中的$I$消掉.

- 代码演示

  ```python
  import numpy as np
  
  
  def forward(self, Q, V, A, B, O, PI):
          """
          计算:p(o|lambda)
          """
          N = len(Q)  # 计算状态总数
          M = len(O)  # 计算观测总数
          
          alphas = np.zeros((N, M))  # 初始化空的alpha矩阵,每一个时刻都会有一个具体的观测变量,也即Ｍ可以表示时间节点
          T = M
          
          for t in range(T):  # 遍历所有时间节点来进行计算
              #  0.取得对应时刻观测序列的index
              indexOfO = V.index(O[t])
              #  1.初始化t=0时刻的alpha
              for i in range(N):  # 遍历所有的状态序列 
                  if t == 0:
                      alphas[i][t] = PI[t][i] * B[i][indexOfO]  # i状态时刻所对应的观测量值
                      print("初始化状态的alpha: \n",alphas)
                      
                  else: 
                      alphas[i][t] = np.dot([alpha[t-1] for alpha in alphas],
                                           [a[i] for a in A] )* B[i][indexOfO]
                      
          
              
          p = np.sum([alpha[M-1] for alpha in alphas])  # 取得最后一个alpha序列
          print("Alphas矩阵:\n", alphas)
          print("A:\n", A)
          print("概率值为多少: ", p)
  
          
  if __name__ == "__main__":
       #0.初始化数据
      Q = [1, 2, 3]
      V = ['红', '白']
  
      A = [[0.5, 0.2, 0.3],
          [0.3, 0.5, 0.2],
          [0.2, 0.3, 0.5]]
  
      B = [[0.5, 0.5],
          [0.4, 0.6],
          [0.7, 0.3]]
      O = ['红', '白', '红' ]
      pi = [[0.2, 0.4, 0.4]]  
      # 1.前向传播算法
      forward(Q, V, A, B, O, pi)
  ```

  <font color = ##0099ff>输出结果展示：</font>

  ```python
  初始化状态的alpha: 
   [[0.1 0.  0. ]
   [0.  0.  0. ]
   [0.  0.  0. ]]
  初始化状态的alpha: 
   [[0.1  0.   0.  ]
   [0.16 0.   0.  ]
   [0.   0.   0.  ]]
  初始化状态的alpha: 
   [[0.1  0.   0.  ]
   [0.16 0.   0.  ]
   [0.28 0.   0.  ]]
  Alphas矩阵:
   [[0.1      0.077    0.04187 ]
   [0.16     0.1104   0.035512]
   [0.28     0.0606   0.052836]]
  A:
   [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
  概率值为多少:  0.130218
  ```

  

**(2) Backward算法:**

- 算法推导

![](/home/gavin/Machine/机器学习理论/code /第十章/backward算法.png)

- 算法具体流程

  (1) 初值

  > $$
  > \beta_T(i)=1, i=1,2,\dots,N
  > $$
  > 在$t=T$时刻, 观测序列已经确定.
  >

  (2) 递推

  > $$
  > \beta_t(i)=\sum\limits_{j=1}^Na_{ij}b_j(o_{t+1})\beta_{t+1}(j)\color{black}, i=1,2,\dots,N, t=T-1, T-2,\dots,1
  > $$
  > 从后往前推
  > $\beta = ABo\beta = N \times N \times N \times M \times M \times N \times N \times 1 = N \times 1$

  (3) 终止

  > $$
  > P(O|\lambda)=\sum\limits_{i=1}^N\pi_ib_i(o_1)\beta_1(i)=\sum\limits_{i=1}\alpha_1(i)\beta_1(i)
  > $$
  >

- 代码演示

  ```python 
  def backward(self, Q, V, A, B, O, PI):
          """
          后项算法
          """
          N = len(Q)
          M = len(O)
          time = []
  
          betas = np.ones((N, M))
  
          for t in range(M-2, -1, -1):
              time.append(t)#记录时间序列
              indexOfO = V.index(O[t+1])#记录上一时刻的观测值
              
              for i in range(N):
                  betas[i][t] = np.dot(
                  np.multiply(A[i],[b[indexOfO] for b in B]),
                  [beta[t+1] for beta in betas]
                  )
  
          indexOfO = V.index(O[0])#记录初始状态的beta１值
          P = np.dot(np.multiply(PI, [b[indexOfO] for b in B]),
          [beta[0] for beta in betas])#最后时刻的beta1
          print("观测序列值：",time)
          print("后项算法的概率值Ｐ：", P)
          print("Beta矩阵：\n", betas)
         
         
  if __name__ == "__main__":
           #0.初始化数据
          Q = [1, 2, 3]
          V = ['红', '白']
  
          A = [[0.5, 0.2, 0.3],
              [0.3, 0.5, 0.2],
              [0.2, 0.3, 0.5]]
  
          B = [[0.5, 0.5],
              [0.4, 0.6],
              [0.7, 0.3]]
          O = ['红', '白', '红' ]
          pi = [[0.2, 0.4, 0.4]]  
          # 1.后向传播算法
          backward(Q, V, A, B, O, pi)
  ```

  <font color =##0099ff>输出结果展示：</font>

  ```python
  观测序列值： [1, 0]
  后项算法的概率值Ｐ： [0.130218]
  Beta矩阵：
   [[0.2451 0.54   1.    ]
   [0.2622 0.49   1.    ]
   [0.2277 0.57   1.    ]]
  ```

#### 2.2 Learning问题： $\lambda_{MLE} = argmax_{\lambda}P(O|\lambda)$

>​		假设给定训练数只包含Ｓ个长度为Ｔ的观测序列$\{O_{1}, O_{2},\dots O_{s}\}$而没有对应的状态序列，目标是学习隐马尔可夫模型$\lambda = (A, B, \pi)$.我们将观测序列数据看作观测数据$O$，将状态序列数据看作不可观测的隐数据$I$，那么隐马尔可夫模型事实上是一个含有隐变量的概率模型，它的参数学习可以由ＥＭ算法实现．

- Baum-Welch算法推导

  ![Em](/home/gavin/Machine/机器学习理论/code /第十章/Em.png)

#### 2.3 Decoding:  $ I = argmax_{I}P(I|\lambda)$ 

- 维特比算法

> ​		维特比算法实际上是用动态规划解隐马尔可夫模型预测问题，即用动态规划求概率最大路径(z最优路径).这时一条路径对应着一个状态序列．算法描述如下：
>
> 输入: 模型$\lambda=(A, B, \pi)$和观测$O=(o_1, o_2,\dots,o_T)$
>
> 输出: 最优路径$I^*=(i_1^*, i_2^*,\dots,i_T^*)$
>
> 1. 初始化
> $\delta_1(i)=\pi_ib_i(o_1), i=1,2,\dots,N$
> $\psi_1(i)=0, i=1,2,\dots,N$
> 1. 递推
> $t=2,3,\dots,T$
> $\delta_t(i)=\max\limits_{1\leqslant j \leqslant N}\left[\delta_{t-1}(j)a_{ji}\right]b_i(o_t), i=1,2,\dots,N$
> $\psi_t(j)=\arg\max\limits_{1\leqslant j \leqslant N}\left[\delta_{t-1}(j)a_{ji}\right], i=1,2,\dots,N$
> 1. 终止
> $P^*=\max\limits_{1\leqslant i\leqslant N}\delta_T(i)$
> $i_T^*=\arg\max\limits_{1\leqslant i \leqslant N}\left[ \delta_T(i)\right]$
> 1. 最优路径回溯
> $t=T-1, T-2, \dots,1$
> $i_t^*=\psi_{t+1}(i_{i+1}^*)$

- 代码演示

  ```python
   def viterbi(self, Q, V, A, B, O, PI):
          """
          维特比算法：通过观测变量推测隐藏变量的值
          """
          #  0. 初始化数据
          N = len(Q)
          M = len(O)
          deltas = np.zeros((N, M))
          psis = np.zeros((N, M))
          I = np.zeros((1, M))
          
          for t in range(M):
              realT = t + 1
              indexOfO = V.index(O[t])  # 读取状态索引
              for i in range(N):
                  realI = i + 1
          #  1. 初始化
                  if t == 0:
                      deltas[i][t] = PI[0][i] * B[i][indexOfO]
                      psis[i][t] = 0
                  
                  else:
          #　２．递推
                      deltas[i][t] = np.max(
                      np.multiply([delta[t-1] for delta in deltas],
                                 [a[i] for a in A])) * B[i][indexOfO]
                      
                      psis[i][t] = np.argmax(np.multiply([
                          delta[t - 1] for delta in deltas
                      ], [a[i] for a in A])) + 1 #返回索引值从零开始
                      
                      
                      
          #  3.终止条件
          I[0][M-1] = np.argmax([delta[M-1] for delta in deltas]) + 1
          
          #  4.选择路径．．
          for t in range(M-2, -1, -1):
              I[0][t] = psis[int(I[0][t+1]) -1][t + 1]#行数为所取得值（上一个状态的值, 列数为时间序列，＂Ｉ[0][t+1] - 1＂ 下标从零开始）
              print("行数：", (I[0][t+1]-1))
              print("列数：", (t+1))
              
          print("psis:", psis)    
          print("状态序列Ｉ:",I)  
  
          
  if __name__ == "__main__":
           #0.初始化数据
          Q = [1, 2, 3]
          V = ['红', '白']
  
          A = [[0.5, 0.2, 0.3],
              [0.3, 0.5, 0.2],
              [0.2, 0.3, 0.5]]
  
          B = [[0.5, 0.5],
              [0.4, 0.6],
              [0.7, 0.3]]
          O = ['红', '白', '红' ]
          pi = [[0.2, 0.4, 0.4]]  
          # 1.维特比算法
          viterbi(Q, V, A, B, O, pi)
  ```

  <font color =##0099ff>输出结果展示：</font>

  ```python
  psis: [[0. 3. 2.]
   [0. 3. 2.]
   [0. 3. 3.]]
  状态序列Ｉ: [[3. 3. 3.]]
  ```

  ### 参考链接

  [大神手板书推导笔记](https://github.com/shuhuai007/Machine-Learning-Session)

  [隐马尔可夫模型（HMM）攻略](https://blog.csdn.net/likelet/article/details/7056068)

  [隐马尔可夫模型HMM](https://zhuanlan.zhihu.com/p/29938926)
  
  [HMM算法推导](https://www.bilibili.com/video/av32471608?from=search&seid=6619590868201023931)
  
  [统计学习方法代码实现](https://github.com/fengdu78/lihang-code)



