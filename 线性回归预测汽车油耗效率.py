import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读入数据
path = R'C:\Users\cjy\PycharmProjects\PythonProject\learn_machine_learning\线性回归预测汽车油耗效率\dataset\auto-mpg.csv'
columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car name"]
# mpg - > 燃油效率
# cylinders -> 气缸
# displacement - > 排量
# horsepower - > 马力
# weight - > 重量
# acceleration - > 加速度
# model year - > 型号年份
# origin = > 编号
# car name - > 原产地
cars = pd.read_csv(path, delim_whitespace=True, names=columns)
cars.info()

# 探究数据模型
# 在horsepower中，有一些值是'?'
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
cars = cars[cars.horsepower != '?']
#用散点图分别展示气缸、排量、重量、加速度与燃油效率的关系
fig = plt.figure(figsize=(13,20))
ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(323)
ax4 = fig.add_subplot(324)
ax5 = fig.add_subplot(325)
ax1.scatter(cars['cylinders'],cars['mpg'],alpha=0.5)
ax1.set_title('cylinders')
ax2.scatter(cars['displacement'],cars['mpg'],alpha=0.5)
ax2.set_title('displacement')
ax3.scatter(cars['weight'],cars['mpg'],alpha=0.5)
ax3.set_title('weight')
ax4.scatter(cars['acceleration'],cars['mpg'],alpha=0.5)
ax4.set_title('acceleration')
ax5.scatter([float(x) for x in cars['horsepower'].tolist()],cars['mpg'],alpha=0.5)
ax5.set_title('horsepower')
# 从图我们可以看出，汽车的燃油效率mpg与排量displacement、重量weight、马力horsepower三者都存在一定的线性关系，
# 其中汽车重量weight与燃油效率线性关系最为明显，首先我们就利用weight一个单变量去构建线性回归模型，看看是否能预测出来

# 拆分训练集和测试集
# 取数据中的20%作为测试集，其他均为训练集
Y = cars['mpg']
X = cars[['weight']]
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

# 单变量线性回归
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr = lr.fit(X_train,Y_train)

# 可视化结果
# 利用我们训练完的模型去测试一下我们的训练集和测试集
plt.scatter(X_train, Y_train, color = 'red', alpha=0.3)
plt.scatter(X_train, lr.predict(X_train),color = 'green',alpha=0.3)
plt.xlabel('weight')
plt.ylabel('mpg')
plt.title('train data')

# 测试集
plt.scatter(X_test,Y_test,color = 'blue',alpha=0.3)
plt.scatter(X_train,lr.predict(X_train),color='green',alpha=0.3)
plt.xlabel('weight')
plt.ylabel('mpg')
plt.title('test data')

# 模型评价
print(lr.coef_)
print(lr.intercept_)
print('score = {}'.format(lr.score(X,Y)))
'''
[-0.00772198]
46.43412847740396
score = 0.6925641006507041
'''

# 多变量线性回归模型
cars = cars[cars.horsepower != '?']
mul = ['weight','horsepower','displacement'] # 选择三个变量进行建立模型
mul_lr = LinearRegression()
mul_lr.fit(cars[mul],cars['mpg']) # 训练模型
cars['mpg_prediction'] = mul_lr.predict(cars[mul])
print(cars.head())

# 模型得分
mul_score = mul_lr.score(cars[mul],cars['mpg'])
mul_score
# 0.7069554693444708
from sklearn.metrics import mean_squared_error as mse
mse = mse(cars['mpg'],cars['mpg_prediction'])
print('mse = %f'%mse)
print('rmse = %f'%np.sqrt(mse))
'''
mse = 17.806188
rmse = 4.219738
'''

# 可视化
fig = plt.figure(figsize = (8,8))
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)
ax1.scatter(cars['weight'], cars['mpg'], c='blue', alpha=0.3)
ax1.scatter(cars['weight'], cars['mpg_prediction'], c='red', alpha=0.3)
ax1.set_title('weight')
ax2.scatter([ float(x) for x in cars['horsepower'].tolist()], cars['mpg'], c='blue', alpha=0.3)
ax2.scatter([ float(x) for x in cars['horsepower'].tolist()], cars['mpg_prediction'], c='red', alpha=0.3)
ax2.set_title('horsepower')
ax3.scatter(cars['displacement'], cars['mpg'], c='blue', alpha=0.3)
ax3.scatter(cars['displacement'], cars['mpg_prediction'], c='red', alpha=0.3)
ax3.set_title('displacement')
plt.show()