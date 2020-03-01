# 支持向量机

### 0. 前言

> ​	支持向量机是一种二分类的模型，它的基本模型是定义在特征空间上间隔最大的线性分类器，间隔最大使它有别与其他感知机模型。有线性支持向量机，线性可分支持向量机和非线性支持向量机。

### 1. 线性可分支持向量机(Hard Margin)

#### 1.1 函数间隔和几何间隔

**函数间隔**

>​		对于给定数据集$T$和超平面$(w,b)$，定义超平面$(w,b)$关于样本点$(x_i,y_i)$的函数间隔为
>$$
>\hat \gamma_i=y_i(w\cdot x_i+b)
>$$
>定义超平面$(w,b)$关于训练数据集$T$的函数间隔为超平面$(w,b)$关于$T$中所有样本点$(x_i,y_i)$的函数间隔之最小值，即
>$$
>\hat \gamma=\min_{i=1,\cdots,N}\hat\gamma_i
>$$
>函数间隔可以表示分类预测的**正确性**及**确信度**。
>
>

**几何间隔**

>$$
>r=\frac{label(w^Tx+b)}{||w||_2}
>$$
>
>​		当数据被正确分类时，几何间隔就是点到超平面的距离为了求几何间隔最大，SVM基本问题可以转化为求解:($\frac{r^*}{||w||}$为几何间隔，(${r^*}$为函数间隔)
>
>
>$$
>\max\ \frac{r^*}{||w||}
>$$

#### 1.2 线性可分优化

>​		支持向量机最简单的情况是线性可分支持向量机，或硬间隔支持向量机。构建它的条件是训练数据线性可分。其学习策略是最大间隔法。可以表示为凸二次规划问题，其原始最优化问题为
>
>$$
>\min _{w, b} \frac{1}{2}\|w\|^{2}
>$$
>
>$$
>s.t. \quad y_{i}\left(w \cdot x_{i}+b\right)-1 \geqslant 0, \quad i=1,2, \cdots, N
>$$
>
>求得最优化问题的解为$w^*$，$b^*$，得到线性可分支持向量机，分离超平面是
>
>$$w^{*} \cdot x+b^{*}=0$$
>
>分类决策函数是
>
>$$f(x)=\operatorname{sign}\left(w^{*} \cdot x+b^{*}\right)$$
>
>最大间隔法中，函数间隔与几何间隔是重要的概念。
>
>线性可分支持向量机的最优解存在且唯一。位于间隔边界上的实例点为支持向量。最优分离超平面由支持向量完全决定。
>二次规划问题的对偶问题是
>$$\min \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)-\sum_{i=1}^{N} \alpha_{i}$$
>
>$$s.t. \quad \sum_{i=1}^{N} \alpha_{i} y_{i}=0$$
>
>$$\alpha_{i} \geqslant 0, \quad i=1,2, \cdots, N$$
>
>通常，通过求解对偶问题学习线性可分支持向量机，即首先求解对偶问题的最优值
>
>$a^*$，然后求最优值$w^*$和$b^*$，得出分离超平面和分类决策函数。

**<font color = blue>代码：</font>**

- 数据预处理

  ```python
  import numpy as np
  import matplotlib.pyplot as plt
  from sklearn import datasets
  
  iris = datasets.load_iris()
  X = iris.data
  Y = []
  y = iris.target
  for i in range(100):
      Y.append(y[i])
  #获取前一百个数据点
  X = X[y<2,:2]
  #y = [0:100]
  #Y
  iris = datasets.load_iris()
  X = iris.data
  Y = []
  y = iris.target
  for i in range(100):
      Y.append(y[i])
  #获取前一百个数据点
  X = X[y<2,:2]
  #y = [0:100]
  #Y
  ```

  ![](/home/gavin/Machine/机器学习理论/code /第七章/原始数据.png)

- 线性支持向量

  ```python
  #1.数据归一化，防止数据维度衡量的维度不同
  from sklearn.preprocessing import StandardScaler
  standardScalar = StandardScaler()
  standardScalar.fit(X)
  X_standard = standardScalar.transform(X)
  
  #2.引入sklearn中的svm
  from sklearn.svm import LinearSVC
  svc = LinearSVC(C=1e9)
  svc.fit(X_standard, Y)#训练数据
  
  
  
  ```

- 绘画出分类间隔

  ```python
  def plot_decision_boundary(model, axis):
      x0, x1 = np.meshgrid(
      np.linspace(axis[0], axis[1], int((axis[1] - axis[0])*100)).reshape(-1, 1),
      np.linspace(axis[2], axis[3], int((axis[3] - axis[2])*100)).reshape(-1, 1)
      )
      
      X_new = np.c_[x0.ravel(), x1.ravel()]
      
      
      y_predict = model.predict(X_new)
      zz = y_predict.reshape(x0.shape)
      print("zz",zz.shape)
      from matplotlib.colors import ListedColormap
      custom_cmap = ListedColormap(['#FF6A9A', '#FFF59D', "#90CAF9"])
      print(x0.shape, x1.shape, zz.shape)
      plt.contourf(x0, x1,  zz,linewidth=5, map = custom_cmap)
      
  if __name__ == "__main__":
      plot_decision_boundary(svc, axis = [-3, 3, -3, 3])
      plt.scatter(X_standard[:50, 0], X_standard[:50, 1])
      plt.scatter(X_standard[50:100, 0], X_standard[50:100, 1])
      plt.title("SVM")
      plt.savefig("C为无穷大时")
      plt.show()
  ```

  ![](/home/gavin/Machine/机器学习理论/code /第七章/C为无穷大时.png)

### 2. 线性近似可分（Soft Margin）

>​		现实中训练数据是线性可分的情形较少，训练数据往往是近似线性可分的，这时使用线性支持向量机，或软间隔支持向量机。线性支持向量机是最基本的支持向量机。
>
>对于噪声或例外，通过引入松弛变量$\xi_{\mathrm{i}}$，使其“可分”，得到线性支持向量机学习的凸二次规划问题，其原始最优化问题是
>
>$$
>\min _{w, b, \xi} \frac{1}{2}\|w\|^{2}+C \sum_{i=1}^{N} \xi_{i}
>$$
>
>$$
>s.t. \quad y_{i}\left(w \cdot x_{i}+b\right) \geqslant 1-\xi_{i}, \quad i=1,2, \cdots, N
>$$
>
>$$
>\xi_{i} \geqslant 0, \quad i=1,2, \cdots, N
>$$
>
>求解原始最优化问题的解$w^*$和$b^*$，得到线性支持向量机，其分离超平面为
>
>$$w^{*} \cdot x+b^{*}=0$$
>
>分类决策函数为
>
>$$f(x)=\operatorname{sign}\left(w^{*} \cdot x+b^{*}\right)$$
>
>线性可分支持向量机的解$w^*$唯一但$b^*$不唯一。对偶问题是
>
>$$\min _{\alpha} \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)-\sum_{i=1}^{N} \alpha_{i}$$
>
>$$s.t. \quad \sum_{i=1}^{N} \alpha_{i} y_{i}=0$$
>
>$$0 \leqslant \alpha_{i} \leqslant C, \quad i=1,2, \cdots, N$$
>
>线性支持向量机的对偶学习算法，首先求解对偶问题得到最优解$\alpha^*$，然后求原始问题最优解$w^*$和$b^*$，得出分离超平面和分类决策函数。
>
>对偶问题的解$\alpha^*$中满$\alpha_{i}^{*}>0$的实例点$x_i$称为支持向量。支持向量可在间隔边界上，也可在间隔边界与分离超平面之间，或者在分离超平面误分一侧。最优分离超平面由支持向量完全决定。
>
>线性支持向量机学习等价于最小化二阶范数正则化的合页函数
>
>$$\sum_{i=1}^{N}\left[1-y_{i}\left(w \cdot x_{i}+b\right)\right]_{+}+\lambda\|w\|^{2}$$,.

<font color =#0099ff>代码</font>

```python
svc2 = LinearSVC(C=0.01)
svc2.fit(X_standard, Y)

plot_decision_boundary(svc2, axis = [-3, 3, -3, 3])
plt.scatter(X_standard[:50, 0], X_standard[:50, 1])
plt.scatter(X_standard[50:100, 0], X_standard[50:100, 1])
plt.title("Soft margin")
plt.title("C001")
plt.show()
```

![](/home/gavin/Machine/机器学习理论/code /第七章/Soft margin.png)

**<font color = red> 不同C值下分类情况</font>**

| ![C001penalty](/home/gavin/Machine/机器学习理论/code /第七章/C001penalty.png) | ![](/home/gavin/Machine/机器学习理论/code /第七章/C1e9penalty.png) |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                            C=0.01                            |                            C=1e9                             |

**总结：**在软间隔的的优化过程中增加了类似与L1正则化项，并且增加了一项C值来调节惩罚值和最大间隔的比重问题，有上述图像可以观察得到，当Ｃ赋予比较的大值时，此时的间隔边界允许犯错的点比较少，并且Ｃ处于一个合理值时，会近似为求取硬间隔问题，而但Ｃ比较小时，此时间隔边界允许放错误的点变得更加多，即相当于增加了松弛变量的值。除此之外，我们在求取最优超平面时，所使用的知识样本上极少数的点，即在间隔边界上的点，这些点就叫做支持向量，其他点　对于求解超平面的解没有影响。

### 3.非线性支持向量机

#### 3.1 问题引入

```python
import numpy as np
import matplotlib.pyplot as plt

#制造数据
x = np.arange(-4, 5, 1)

#标签
y = np.array((x>=-2)&(x<=2), dtype="int")

plt.scatter(x[y==0], [0]*len(x[y==0]), label = "0")
plt.scatter(x[y==1], [0]*len(x[y==1]), label = '1')
plt.savefig("高斯数据")
plt.legend()
plt.show()
```

![高斯数据](/home/gavin/Machine/机器学习理论/code /第七章/高斯数据.png)

​		以上数据点在一维情况不能找到一个超平面对数据进行分类，所以需要寻找数据点在另一个特征空间内使得数据能够线性可分。所以使用高斯转换过，使得数据从以为装换到更高维度。

```python
def gaussian(x, l):
    gamma = 1.0
    return np.exp(-gamma * (x-l)**2)
    
    
#landmark
l1 , l2 = -1, 1
#获取新的维度数据
X_new = np.empty((len(x), 2))
for i,data in enumerate(x):
    X_new[i,0] = gaussian(data, l1)
    X_new[i,1] = gaussian(data, l2)  

＃绘画结果
plt.scatter(X_new[y==0,0], X_new[y==0,1])
plt.scatter(X_new[y==1,0], X_new[y==1,1])
plt.savefig("gaussian classification")
plt.show()
```

![gaussian classification](/home/gavin/Machine/机器学习理论/code /第七章/gaussian classification.png)

#### 3.2　核函数

>​		对于输入空间中的非线性分类问题，可以通过非线性变换将它转化为某个高维特征空间中的线性分类问题，在高维特征空间中学习线性支持向量机。由于在线性支持向量机学习的对偶问题里，目标函数和分类决策函数都只涉及实例与实例之间的内积，所以不需要显式地指定非线性变换，而是用核函数来替换当中的内积。核函数表示，通过一个非线性转换后的两个实例间的内积。具体地，$K(x,z)$是一个核函数，或正定核，意味着存在一个从输入空间x到特征空间的映射$\mathcal{X} \rightarrow \mathcal{H}$，对任意$\mathcal{X}$，有
>$$
>K(x, z)=\phi(x) \cdot \phi(z)
>$$
>对称函数$K(x,z)$为正定核的充要条件如下：对任意$$\mathrm{x}_{\mathrm{i}} \in \mathcal{X}, \quad \mathrm{i}=1,2, \ldots, \mathrm{m}$$，任意正整数$m$，对称函数$K(x,z)$对应的Gram矩阵是半正定的。
>
>所以，在线性支持向量机学习的对偶问题中，用核函数$K(x,z)$替代内积，求解得到的就是非线性支持向量机
>$$
>f(x)=\operatorname{sign}\left(\sum_{i=1}^{N} \alpha_{i}^{*} y_{i} K\left(x, x_{i}\right)+b^{*}\right)
>$$
>使用核函数的优点;
>
>-  不需要每一次都具体计算出原始样本点映射的新的无穷维度的样本点，直接使用映射后的新的样本点的点乘计算公式即可
>- 减少计算量  
>- 较少存储空间

##### 3.2.1 高斯核$\gamma$参数的理解

**高斯核：**

![](/home/gavin/Machine/机器学习理论/code /第七章/高斯核函数.png)

**<font color = blue>RBF核参数分析代码</font> **

1. 数据预准备

   ```python
   #RBF核
   import numpy as np
   import matplotlib.pyplot as plt
   from sklearn import datasets
   
   #加载数据
   X, y = datasets.make_moons(noise=0.15, random_state=666)
   print(len(X), y)
   plt.scatter(X[y==0, 0], X[y==0, 1])
   plt.scatter(X[y==1, 0], X[y==1, 1])
   plt.show()
   ```

   ![](/home/gavin/Machine/机器学习理论/code /第七章/原始数据的采集.png)

2. 数据分析

   ```python
   from sklearn.svm import SVC
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   
   
   def RBFKernelSVC(gamma):
       return Pipeline([
           ("std_scaler", StandardScaler()),
           ("svc", SVC(kernel="rbf", gamma=gamma))
       ])
       
       
   svc = RBFKernelSVC(gamma=1)    
   svc.fit(X, y)
   ```

   **输出结果**

   ```txt
   Pipeline(memory=None,
            steps=[('std_scaler',
                    StandardScaler(copy=True, with_mean=True, with_std=True)),
   
   ('svc',
                    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                        decision_function_shape='ovr', degree=3, gamma=1,
                        kernel='rbf', max_iter=-1, probability=False,
                        random_state=None, shrinking=True, tol=0.001,
                        verbose=False))],
            verbose=False)
   ```

   ```python
   def plot_decision_boundary(model, axis):
       
       x0, x1 = np.meshgrid(
           np.linspace(axis[0], axis[1], int((axis[1]-axis[0])*100)).reshape(-1, 1),
           np.linspace(axis[2], axis[3], int((axis[3]-axis[2])*100)).reshape(-1, 1),
       )
       X_new = np.c_[x0.ravel(), x1.ravel()]
   
       y_predict = model.predict(X_new)
       zz = y_predict.reshape(x0.shape)
   
       from matplotlib.colors import ListedColormap
       custom_cmap = ListedColormap(['#EF9A9A','#FFF59D','#90CAF9'])
       
       plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)
       
       
   plot_decision_boundary(svc, axis=[-1.5, 2.5, -1.0, 1.5])
   plt.scatter(X[y==0,0], X[y==0,1])
   plt.scatter(X[y==1,0], X[y==1,1])
   plt.title("gammea = 1")
   plt.savefig("gammea(1)")
   plt.show()
   ```

   | ![gammea(01)](/home/gavin/Machine/机器学习理论/code /第七章/gammea(01).png) | ![gammea(05)](/home/gavin/Machine/机器学习理论/code /第七章/gammea(05).png) |
   | :----------------------------------------------------------: | :----------------------------------------------------------: |
   |                          过度欠拟合                          |                            欠拟合                            |
   | ![gammea(3)](/home/gavin/Machine/机器学习理论/code /第七章/gammea(3).png) | ![gammea(100)](/home/gavin/Machine/机器学习理论/code /第七章/gammea(100).png) |
   |                          just right                          |                            过拟合                            |

   **结论：**在高斯核中$\gamma$参数表示着数据的拟合情况，参数越大表示数据拟合的情况越好，但参数设置过大时会造成过拟合现象的产生。

### 4. 参考

[支持向量机通俗导论](https://blog.csdn.net/v_JULY_v/article/details/7624837)

[机器学习：svm](https://www.cnblogs.com/volcao/p/9465214.html)

