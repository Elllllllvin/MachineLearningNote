# Exercise1 Linear Regression

```Python []
import pandas
#1.
path = 'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
# 读取数据集，names是属性名字
data.head()
# 显示数据前几个值

#2. pandas基本绘图函数
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
plot.show()

#显示图片

#3.在第0列添加name为‘Ones’的属性，值全部赋值为1
data.insert(0, 'Ones', 1)
#
#4.二维数组
#4.1 shape
​data.shape     二维数组的行，列数
data.shape[0]  行
data.shape[1]  列
​
#4.2取所有行和0-col-1列交叉的所有的数据
X = data.iloc[:, 0:cols-1]

#5.把dataframe格式转换为矩阵
X = np.matrix(X.values)
Y = np.matrix(Y.values)

#6.矩阵转置
theta.T #theta是经过np.matrix()函数处理过的

#7.初始化函数
iters = 1000
cost = np.zeros(iters)#建立一个含有1000个元素的数组，全部赋值为0

#8.常用作确定图的横坐标
np.linspace(start,stop,50) #构建等差数列
        #   开始点，结束点，样本数据量
np.arange() #函数返回一个有终点和起点的固定步长的排列,

#9.画一个散点图+直线
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + g[0, 1] * x

plt.figure(figsize=(12, 8)) #图的长宽
plt.xlabel('Population')    #横坐标表示
plt.ylabel('Profit')        #纵坐标表示
l1 = plt.plot(x, f, label='Prediction', color='red') #直线图（一次函数）
l2 = plt.scatter(data.Population, data.Profit, label='Traing Data', ) #散点图
plt.legend(loc='best') #在左上角显示label的内容
plt.title('Predicted Profit vs Population Size') #图的标题
plt.show()

```

⭐plt.plot()函数详解
plt.plot(x, y, format_string, \*\*kwargs)

其中 x，y 分别为 x，y 轴数据，可为列表或数组；format_string 控制曲线的格式字符串（颜色、风格、标记等），label 可以为该曲线，添加标签配合 legend 可以显示在左上角

## 线性回归模型

```Python []
from sklearn import linear_model
model = linear_model.LinearRegression() #线性回归模型
model.fit(X, Y)
#X,Y均为处理过的矩阵，X为n*m矩阵，n个样本*m条特征；Y为结果n*1矩阵
x = np.array(X[:, 1].A1)        #x横坐标的值（X的第1列元素化为一个一元矩阵
        #       .A1的作用是把矩阵化为扁平的一元矩阵
y = model.predict(X).flatten()  #求得结果y（纵坐标的值
                #  .flatten()是把二维数组压成一维数组（必须是numpy数组）
                #因为model.predict(X)仍是n*1的二维矩阵
```

## 3D 拟合平面

```Python []
# 画出拟合平面
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
X_ = np.arange(mins[0], maxs[0]+1, 1)
Y_ = np.arange(mins[1], maxs[1]+1, 1)
X_, Y_ = np.meshgrid(X_, Y_)
Z_ = transform_g[0,0] + transform_g[0,1] * X_ + transform_g[0,2] * Y_

# 手动设置角度
ax.view_init(elev=25, azim=125)

ax.set_xlabel('Size')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price')

ax.plot_surface(X_, Y_, Z_, rstride=1, cstride=1, color='red')

ax.scatter(data_[:, 0], data_[:, 1], data_[:, 2])
plt.show()

```

# Exercise2 Logical Regression

[scipy.optimize 优化器的各种使用](https://blog.csdn.net/jiang425776024/article/details/87885969)

```Python []
#使用两种函数来拟合优化θ，结果非常近似
#1.
import scipy.optimize as opt
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, Y))
                    #定义好代价函数，theta，梯度下降函数，数据集参数后，拟合出最优的θ
result

#2.
res = opt.minimize(fun=cost, x0=np.array(theta), args=(X, np.array(Y)), method='Newton-CG', jac=gradient)
#opt.minimize根据method参数的不同有多种拟合方法，这里使用牛顿共轭梯度法拟合函数
res
```
