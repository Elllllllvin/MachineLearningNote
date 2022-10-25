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
