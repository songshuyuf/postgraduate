参考资料:https://blog.csdn.net/qq_36816848/article/details/122286610?ops_request_misc=%257B%2522request%255Fid%2522%253A%25225996BF8D-DA62-4E0C-981A-6DAE43AE9238%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=5996BF8D-DA62-4E0C-981A-6DAE43AE9238&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-12-122286610-null-null.142^v100^pc_search_result_base7&utm_term=%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%85%B7%E4%BD%93%E6%B5%81%E7%A8%8B&spm=1018.2226.3001.4187





一、反向传播算法定义:反向传播(Backpropagation,缩写BP)，误差反向传播的简称一种与最优化方法（如梯度下降法）结合使用的，用来训练人工神经网络的常见方法。该方法对网络中的所有权重计算损失函数的梯度，这个梯度会反馈给最优化算法，用来更新权值以最小化损失函数。



二、激活函数:对于人工神经网络模型去学习、理解非常复杂和非线性的函数有十分重要的作用，目的：为了增加神经网络模型的非线性。

![image-20241112161656076](C:\Users\10648\AppData\Roaming\Typora\typora-user-images\image-20241112161656076.png)

1.Sigmoid:也叫Logistic函数:![image-20241112162156830](C:\Users\10648\AppData\Roaming\Typora\typora-user-images\image-20241112162156830.png)

![image-20241112162248818](C:\Users\10648\AppData\Roaming\Typora\typora-user-images\image-20241112162248818.png)

将输入映射到0~1，因此可以通过sigmoid函数将输出转译为概率输出，常用于表示分类问题的事件概率。

优点:平滑、易于求导。

缺点:指数级计算，计算量大；容易出现梯度弥散。



2.Tanh函数，即双曲正切函数，其定义为:![image-20241112162529816](C:\Users\10648\AppData\Roaming\Typora\typora-user-images\image-20241112162529816.png)

函数曲下如下:

![image-20241112162607563](C:\Users\10648\AppData\Roaming\Typora\typora-user-images\image-20241112162607563.png)

Tanh函数能将输入x映射到[-1,1]区间，是Sigmoid函数的改进版，收敛速度快，不容易出现loss值震荡，但无法解决梯度弥散问题，同时计算量很大。



3.ReLU：提出之前，sigmoid都是激活函数首选。RELU使网络层数达到了8层，定义如下:
ReLU(x)=max(0,x)

ReLu对小于0的值全部抑制为0，对于正数则直接输出(启发于生物学)，曲线如下:

![image-20241112162957466](C:\Users\10648\AppData\Roaming\Typora\typora-user-images\image-20241112162957466.png)

优点:训练快速收敛，解决了梯度弥散问题，在信息传递的过程中，大于0的部分梯度总是为1。

缺点:输入小于0，很大的梯度也会阻止。

**ReLU 函数的设计源自神经科学，计算十分简单，同时有着优良的梯度特性，在大量的深度学习应用中被验证非常有效，是应用最广泛的激活函数之一。**



4.Leaky ReLU：为了克服ReLU可能造成的梯度弥散现象所提出的，定义如下:![image-20241112163248687](C:\Users\10648\AppData\Roaming\Typora\typora-user-images\image-20241112163248687.png)

其中，p为超参数，p为0时退化为了ReLU，当P≠0时，x＜0时能够获得较小的梯度值，从而避免梯度弥散。函数曲下如下:

![image-20241112163433053](C:\Users\10648\AppData\Roaming\Typora\typora-user-images\image-20241112163433053.png)



5.softmax函数:定义如下:![image-20241112163503880](C:\Users\10648\AppData\Roaming\Typora\typora-user-images\image-20241112163503880.png)

不仅可以将输出值映射到[0,1]区间，还满足所有的输出值之和为1的特性，在多分类任务中使用的非常频繁。

另外，在**softmax函数多分类**问题中，若损失函数选用**交叉熵**，则下降梯度计算起来将会非常方便，使得网络训练过程中的迭代**计算复杂度**大大降低。

https://ai-wx.blog.csdn.net/article/details/104729911

https://blog.csdn.net/shyjhyp11/article/details/109279411



三、随机梯度下降:

非常有用的视频：

https://www.bilibili.com/video/BV18P4y1j7uH/?spm_id_from=333.999.0.0

学习网络训练:前向计算过程与反向传播过程，前向传播就是预定好的卷积、层化层等，按照网络结构一层层前向计算，得到预测的结果。反向传播过程，是为了将设定的网络中的众多参数一步一步调整，使得预测结果更加贴近真实值。

那么参数该如何更新呢？显而易见，朝着目标函数下降最快的方向更新，更确切地说，朝着梯度方向更新。

三种最基本的梯度下降方法:

1.SGD随机梯度下降方法，每次迭代(更新参数)只使用单个样本，其中x是输入数据,y是标签，参数更新表达式如下:
![image-20241112170629255](C:\Users\10648\AppData\Roaming\Typora\typora-user-images\image-20241112170629255.png)

优点:一次迭代只需对一个样本进行计算，速度很快。

缺点:1.目标损失函数值会剧烈波动，可能跳到局部最小值，永远不会收敛。

 2.一次迭代只用一个样本，没有发挥GPU并行的优势。



2.批量梯度下降:BGD，每次迭代更新中使用所有训练样本，更新表达式如下:

![image-20241112171004728](C:\Users\10648\AppData\Roaming\Typora\typora-user-images\image-20241112171004728.png)

优点:能保证收敛到全局最小值。

缺点：数据量很大时，迭代速度很慢。



3.小批量梯度下降:MBGD折中BGD与SGD方法，每次迭代使用batch_size个训练样本:

![image-20241112171303299](C:\Users\10648\AppData\Roaming\Typora\typora-user-images\image-20241112171303299.png)

优点:收敛比SGD更快，也能避免BGD在数据集大时，迭代速度慢的问题。

缺点：也可能收敛到全局最小值。





四、损失函数:用来评价网络模型的输出的预测值 与真实值之间的差异,数值尽可能的小。

https://www.bilibili.com/video/BV1RL411T7mT/?spm_id_from=333.999.0.0

在实际中，常使用交叉熵作为损失函数，

回归损失函数:

(1)均方误差损失函数:

![image-20241112184643940](C:\Users\10648\AppData\Roaming\Typora\typora-user-images\image-20241112184643940.png)

```python
def mean_squared_error(y_true,y_pred):
    return np.mean(np.squared(y_pred-y_true),axis=-1)
#axis=-1 代表着沿着最后一个轴计算均值。如果希望返回所有样本的MSE，可以去掉axis
```



(2)平均绝误差损失函数:

![image-20241112185608803](C:\Users\10648\AppData\Roaming\Typora\typora-user-images\image-20241112185608803.png)

```python
def mean_absolute_error(y_true,y_pred):
	return np.mean(np.abs(y_pred - y_true),axis = -1)
```

（3）均方误差对数损失函数:

![image-20241112185906446](C:\Users\10648\AppData\Roaming\Typora\typora-user-images\image-20241112185906446.png)

```python
def mean_squared_logarithmic_error(y_true,y_pred):
    first_log = np.log(np.clip(y_pred,10e - 6,None) + 1)
    second_log = np.log(np.clip(y_true,10e - 6,None) + 1)
    return np.mean(np.square(first_log - second_log),axis= -1)
#np.clip在于将值限制在10 e-6以上，避免出现0值的问题 +1(同理)
```

(4)平均绝对百分比误差损失函数：

![image-20241112202554884](C:\Users\10648\AppData\Roaming\Typora\typora-user-images\image-20241112202554884.png)

```python
def mean_absolute_percentage_error(y_true,y_pred):
	diff=np.abs((y_pred - y_true)/np.clip(np.abs(y_true),10e-6,None))
	return 100* np.mean(diff,axis=-1)
```

 

分类损失函数:

(1)Logistic损失函数:常用于二分类任务，常用于逻辑回归模型的最大似然估计。

![image-20241113101958137](C:\Users\10648\AppData\Roaming\Typora\typora-user-images\image-20241113101958137.png)

```python
def likelihood(y_true,y_pred):
    #确保y_pred在(0,1)之间
    y_pred = np.clip(y_pred,1e-15,1e-15)
    #计算似然函数
    likehood = np.prod(y_pred**Y_true * (1-y_pred)**(1-y_true))
    #prod是累乘
    return likehood
```

（2）负对数似然损失函数:逻辑回归损失函数

​	![image-20241113102715710](C:\Users\10648\AppData\Roaming\Typora\typora-user-images\image-20241113102715710.png)

```python
def log_loss(y_true,y_pred):
    y_pred = np.clip(y_pred,1e-15,1e-15) #保持预测值在(0,1)之间
    #计算损失函数
    loss = -np.mean(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))
    #np.mean会自动累和并求平均值
    #loss = -np.sum(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))
    return loss
```

(3)交叉熵损失:可以处理多个分类问题

![image-20241113103901519](C:\Users\10648\AppData\Roaming\Typora\typora-user-images\image-20241113103901519.png)

```python
def cross_entropy(y_true,y_pred):
    return -np.mean(y_true*np.log(y_pred+10e-6)) #10e-6是很小的数，防止出现0值
```

(4)Hinge损失函数:典型分类器是SVM算法，因为Hinge损失可以用来解决间隔最大化问题，当分类模型需要硬分类结果的，其是最优选择:

![image-20241113104303926](C:\Users\10648\AppData\Roaming\Typora\typora-user-images\image-20241113104303926.png)

```python
def hinge(y_true,y_pred):
    return np.mean(np.maximum(0.,1.-y_pred*y_true),axis=-1) #沿最后一个维度求平均
```

(5)指数损失函数:典型分类器是AdaBoost算法，定义如下:

![image-20241113105550005](C:\Users\10648\AppData\Roaming\Typora\typora-user-images\image-20241113105550005.png)

```python
def exponential(y_true,y_pred):
	return np.sum(np.exp(-y_pred*y_true))
```



四、常用损失函数:

损失函数可以自定义，前提要考虑数据本身和用于求解的优化方案，常用组合有以下三种:

(1)ReLU+MSE

均方误差损失函数无法处理梯度消失问题，而使用leak ReLU激活函数能够减少梯度消失问题，因此如果要使用均方误差损失函数，一般采用Leak RelU等减少梯度消失的激活函数。

(2)Sigmoid + Logistic

Sigmoid函数会引起梯度消失问题：根据链式求导法，Sigmoid函数求导后由多个[0,1]范围的数进行累乘，而类Logistic损失函数求导时，加上对数后连乘操作转换维求和操作，在一定程度上避免了梯度消失，所以经常看到sigmoid+Logistic的组合

(3)softmax + Logistic

数学上，softmax激活函数会返回类的互斥概率分布，也就是把离散的输出转换为一个同分布互斥概率,如(0.2,0.8)。Logistic基于概率最大似然估计函数，因此输出概率化能够更加方便优化算法进行求导和计算。
