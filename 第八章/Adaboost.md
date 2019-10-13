## Boosting方法

### 前言

​		提升（Boosting）方法是一种常用的统计学习方法，应用广泛且有效．在分类问题中，它通过<font color = red>改变训练样本的权重，学习多个分类器</font>，并且将这些分类器进行<font color = red>线性</font>的组合．

​		提升方法本身也基于这样一种思想：对于一个复杂任务来说，将多个专家的判断进行适当的综合所得出的判断，要比任何一个庄家单独判断的好，实际上就是＂**三个臭皮匠顶个诸葛亮**＂的道理．

![诸葛亮](/home/gavin/Machine/机器学习理论/code /第八章/诸葛亮.jpeg)

**强可学习：**一个概念（类），如果一存在一个多项式的学习算法能够学习它，并且正确率很高，那么就称这个概念是强可学习的．

**弱可学习：**一个概念（类），如果一存在一个多项式的学习算法能够学习它，学习的正确率仅比随机猜测略好，那么称这个概念是弱可学习的．

### AdaBoost

​		**AdaBoost**的提出主要依赖与解决提升算法的两个问题：（１）在每一轮如何改变训练数据的权值或概率分布．（２）如何将弱分类器组合成一个强分类器．而AdaBoost的巧妙之处就是在它将这些想法自然且有效地实现在一种算法里．

#### 算法

**输入**：训练数据集$T=\{(x_1,y_1), (x_2,y_2),...,(x_N,y_N)\}, x\in  \cal X\sube \R^n$, 弱学习方法

**输出**：最终分类器$G(x)$

**步骤**

1. 初始化训练数据的权值分布 
    $$
    D_1=(w_{11},\cdots,w_{1i},\cdots,w_{1N}),  (w_{1i}=\frac{1}{N}  i = 1, ...,N) 
    $$

2. m = 1,2,.....,M

    a）使用初始化权值学习基本分类器
    $$
    G_m(x):X->{-1,+1}
    $$
    b) 求$G_m$在训练集上的分类误差率，在这里可以理解当前分类器的**分类误差就是在训练数据集上被误分类点的权值之和**
    $$
    e_m=\sum_{i=1}^{N}P(G_m(x_i)\ne y_i)=\sum_{i=1}^{N}w_{mi}I(G_m(x_i)\ne y_i)
    $$
    c) 计算$G_m(x)$的系数，这里的对数为自然对数
    $$
    \alpha_m=\frac{1}{2}log\frac{1-e_m}{e_m}
    $$
    ​	当我们绘制出此函数的的函数图像时，通过观察分类误差与分类器系数之间的关系，如下图(x轴为$e_m$, y轴为$a_{m}$)，由图中我们可以得出当我们的分类器的误分类率$e_m<=\frac{1}{2}$时（$a_m >= 0$），随着分类误差的减小，分类器相应的系数增大，所以，这使得**分类误差越小的基本分类器在最终分类器中的作用越大**．

    ![误分类率](/home/gavin/Machine/机器学习理论/code /第八章/误分类率.png)

    

     d) 更新训练数据集的权值分布
    $$
    w_{m+1,i}=\frac{w_{mi}}{Z_m}exp(-\alpha_my_iG_m(x_i))
    $$

    $$
    Z_m=\sum_{i=1}^Nw_{mi}exp(-\alpha_my_iG_m(x_i))(规范化因子)
    $$

    ​	在下图中，a图为我们初始训练数据的权值分布，此时我们使得每一个训练样本在基本分类器中的学习作用相同，然后，在此分类器上对训练数据进行分类，得到误分类点，重新更新训练样本的权值分布后，学习下一个弱分类，反复如此学习，直到数据被正确分类，所以，我们在在每一代的弱分类器中，**不改变所给的训练数据**，当弱分类器的中出现了错误分类的点，我们就会在加大误分类样本的权值，使得它在下一轮的学习中起到更大作用．

    ![adaboost](/home/gavin/Machine/机器学习理论/code /第八章/adaboost.jpg)$​

3. 构造基本分类器的线性组合
    $$
    f(x)=\sum_{m=1}^M\alpha_mG_m(x)
    $$

4. 最终分类器（**加权表决**）
    $$
    G(x)=sign(f(x))=sign(\sum_{m=1}^M\alpha_mG_m(x))
    $$

#### 代码解析

- **源代码**

  ```python
  import numpy as np
  import pandas as pd
  from sklearn.datasets import load_iris
  from sklearn.model_selection  import train_test_split
  import matplotlib.pyplot as plt
  %matplotlib inline
  
  class Adaboost:
      """
      构建AdaBoost
      """
      
      def __init__(self, n_estimators=50, learning_rate = 1.0):
          """初始化超参数"""
          
          self.clf_num = n_estimators
          self.learning_rate = learning_rate
      
      def init_args(self, datasets, labels):
          """初始化参数"""
          
          self.error_array = []
          self.iter = []
  
          self.Y = labels
          self.X = datasets
          self.M, self.N = datasets.shape
          
          self.clf_sets = []  # 记录分类器参数
          self.weights = [1.0/self.M] * self.M #初始化权重
          
          self.alpha = [] #每一个分类器的系数
          
      def _G(self, features, labels, weights):
          """基本分类器"""
          
          m = len(features)
          error = 100000.0
          best_v = 0.0 #　最优切分点
          features_min = min(features)
          features_max = max(features)
          
          # 遍历次数
          n_step = (features_max - features_min+self.learning_rate)//self.learning_rate
          
          direct, compare_array = None, None
          
          for i in range(1, int(n_step)):
              v = features_min + self.learning_rate*i
              
              if v not in features:
                  
                  # positive
                  # 寻找最优的切分点
                  compare_array_positive = np.array([
                      1 if features[k] > v else -1 for k in range(m)
                  ])
                  
                  # 计算误分类率
                  weights_error_positive = sum([
                      weights[k] for k in range(m) if compare_array_positive[k] != labels[k]
                  ])
                  
                  # negative
                  compare_array_negative = np.array([
                      -1 if features[k] > v else 1 for k in range(m)
                  ])
                  weights_error_negative = sum([
                      weights[k] for k in range(m) if compare_array_negative[k] != labels[k]
                  ])
                  
                  if weights_error_positive < weights_error_negative:
                      weight_error = weights_error_positive
                      _compare_array = compare_array_positive
                      direct = "positive"
                  else:
                      weight_error = weights_error_negative
                      _compare_array = compare_array_negative
                      direct = "negative"
                      
                  #记录一次遍历之后最小误差的数据
                  
                  if weight_error < error:
                      
                      error = weight_error
                      compare_array = _compare_array
                      best_v = v #最优切分点
                      
          return best_v, direct, error, compare_array
      
      def _alpha(self, error):
          """计算分类器系数"""
          
          return 0.5 * np.log((1 - error) / error)
      
      def _Z(self, weights, a, clf):
          """计算规范化因子"""
          
          return sum([
              weights[i] * np.exp(-1 * a * self.Y[i] * clf[i]) for i in range(self.M)
          ])
      
      def _w(self, a, clf, Z):
          """计算新一步的权值"""
          
          for i in range(self.M):
              self.weights[i] = self.weights[i] * np.exp(
              -1 * a * self.Y[i] *clf[i]) / Z
              
      def G(self, x, v , direct):
          """决策树桩"""
          
          if direct == "positive":
              return 1 if x>v else -1
          else:
              return -1 if x>v else 1
          
      def fit(self, X, y):
          """训练数据"""
          
          self.init_args(X, y)
          axis = 0 #记录最优切分变量
          best_clf_error, best_v, clf_result = 100000, None, None
          
          #开始迭代
          for epoch in range(self.clf_num):
              for j in range(self.N):  # 选取不同的特征维度进行考量
                  features = self.X[:, j] # 遍历第j个切分变量的所有的所有切分点
                      
                  v, direct, error, compare_array = self._G(
                      features, self.Y, self.weights)
  
                  if error < best_clf_error:
                      best_clf_error = error
                      self.error_array.append(best_clf_error)
                      self.iter.append(epoch+1)
                      best_v = v
                      final_direct = direct
                      clf_result = compare_array # 记录分类器输出的结果
                      axis = j # 记录最优切分变量
                  if best_clf_error == 0:
                      break
  
              # 计算分类器系数
              a = self._alpha(best_clf_error)
              self.alpha.append(a)
  
              # 记录分类器参数
              self.clf_sets.append((axis, best_v, final_direct))
  
              # 记录规范化因子
              Z = self._Z(self.weights, a, clf_result) 
  
              # 权重更新
              self._w(a, clf_result, Z)
              
              #计算训练误差
  #             self.train_error = self.score(X, y)
  
      def predict(self, feature):
          """预测"""
          
          result = 0.0
          
          for i in range(len(self.clf_sets)):  # 遍历所有的分类器
              axis, clf_v, direct = self.clf_sets[i]
              f_input = feature[axis]  # 找到所对应的最优切分变量
              result += self.alpha[i] * self.G(f_input, clf_v, direct)
          
          return 1 if result > 0 else -1  # 符号函数sign
      
      def score(self, X_test, Y_test):
          """评估"""
          
          right_count = 0
          
          for i in range(len(X_test)):
              feature = X_test[i]
              if self.predict(feature) == Y_test[i]:
                  right_count += 1
                  
          return right_count/len(X_test)       
  ```

- **例题8.1**

  ```python
  #　1.创建数据
  X = np.arange(10).reshape(10, 1)
  y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
  
  #　2.图形化
  plt.scatter(X, y)
  plt.title("原始数据分布")
  plt.xlabel("数据")
  plt.ylabel("标签")
  plt.show()
  ```

  ![数据一](/home/gavin/Machine/机器学习理论/code /第八章/数据一.png)

- **预测样本数据：x=4**

  ``` python
  #  3.预测数据
  clf = Adaboost(n_estimators=16, learning_rate=0.5)
  clf.fit(X, y)
  clf.predict([4])
  ```
  
- **预测结果**

  ```python
  -1
  ```

- **预测样本数据：x=6**

  ```python
  #  3.预测数据
  clf = Adaboost(n_estimators=16, learning_rate=0.5)
  clf.fit(X, y)
  clf.predict([6])
  ```

- **预测结果**

  ```python
  1
  ```

- **训练误差趋势图**

  | ![弱分类器5](/home/gavin/Machine/机器学习理论/code /第八章/弱分类器5.png) | ![弱分类器16](/home/gavin/Machine/机器学习理论/code /第八章/弱分类器16.png) |
  | :----------------------------------------------------------: | :----------------------------------------------------------: |
  |                         弱分类器 = 5                         |                        弱分类器 = 16                         |
  | ![弱分类器30](/home/gavin/Machine/机器学习理论/code /第八章/弱分类器30.png) | ![弱分类器100](/home/gavin/Machine/机器学习理论/code /第八章/弱分类器100.png) |
  |                        弱分类器 = 30                         |                        弱分类器 = 100                        |

  **结论**：从上图中可以观察到随着弱分类数目的增加，弱分类器的分类误差也在相应的减小，但是随着弱分类器的数目达到最一定数量之后，即使再叠加弱分类的数目，分类误差依旧平缓．

- **Sklearn数据**

  ```python
  def create_data():
      iris = load_iris()
      X = iris.data[:100,:2]
      y = iris.target[:100]
      return X, y
  
  if __name__ == "__main__":
      # 2.数据模型
      X , y = create_data()
      plt.scatter(X[y==0, 0], X[y==0, 1], label = "0")
      plt.scatter(X[y==1, 0], X[y==1, 1], label = "1")
      plt.legend()
      plt.savefig("原始数据")
      plt.show()
  ```

  ![原始数据](/home/gavin/Machine/机器学习理论/code /第八章/原始数据.png)

  ```python
  # 3.训练集数据分布
  plt.scatter(X_train[:,0], y_train)
  plt.title("训练集数据分布")
  plt.show()
  
  
  ```

  ![训练集数据分布](/home/gavin/Machine/机器学习理论/code /第八章/训练集数据分布.png)

  ```python
  # 4.测试集数据分布
  plt.scatter(X_test[:, 1], y_test)
  plt.title("测试集数据分布")
  plt.savefig("测试集数据分布")
  plt.show()
  ```

  ![测试集数据分布](/home/gavin/Machine/机器学习理论/code /第八章/测试集数据分布.png)

- **测试集测试**

  ```python
  # 4.训练和测试
  clf = Adaboost(n_estimators=10, learning_rate=0.2)
  clf.fit(X_train, y_train)
  clf.score(X_test, y_test)
  
  ```

- **测试结果**

  ```python
0.7272727272727273
  ```

### 提升树

### XgBoost

### 参考
