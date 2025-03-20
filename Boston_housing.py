import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error

plt.style.use('ggplot')

# 读入数据
data = pd.read_csv(
    R'C:\Users\cjy\PycharmProjects\PythonProject\learn_machine_learning\Boston_housing\dataset\train_dataset.csv')
data.info()
# No	属性	数据类型	字段描述x
# 1	CRIM	Float	城镇人均犯罪率
# 2	ZN	Float	占地面积超过2.5万平方英尺的住宅用地比例
# 3	INDUS	Float	城镇非零售业务地区的比例
# 4	CHAS	Integer	查尔斯河虚拟变量 (= 1 如果土地在河边；否则是0)
# 5	NOX	Float	一氧化氮浓度（每1000万份）
# 6	RM	Float	平均每居民房数
# 7	AGE	Float	在1940年之前建成的所有者占用单位的比例
# 8	DIS	Float	与五个波士顿就业中心的加权距离
# 9	RAD	Integer	辐射状公路的可达性指数
# 10	TAX	Float	每10,000美元的全额物业税率
# 11	PTRATIO	Float	城镇师生比例
# 12	B	Float	1000（Bk - 0.63）^ 2其中Bk是城镇黑人的比例
# 13	LSTAT	Float	人口中地位较低人群的百分数
# 14	PRICE	Float	（目标变量/类别属性）以1000美元计算的自有住房的中位数

# 相关性检验
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='PuBu')
# 数据不存在相关性较小的属性，也不用担心共线性，所以我们可以用线性回归模型去预测
data.corr()['PRICE'].sort_values()
# LSTAT、RM、PTRATIO这3个字段是最具有相关性的

# 多变量研究
# 尝试了解因变量和自变量、自变量和自变量之间的关系
sns.pairplot(data[["LSTAT", "RM", "PIRATIO", "PRICE"]])

# 划分训练集和测试集
X, y = data[data.columns.delete(-1)], data['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=888)

# 建立线性回归模型
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
coef = linear_model.coef_  # 回归系数
line_pre = linear_model.predict(X_test)
print('SCORE:{:.4f}'.format(linear_model.score(X_test, y_test)))
print('RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_test, line_pre))))
coef
# SCORE:0.6384
# RMSE:6.0852
df_coef = pd.DataFrame()
df_coef['Title'] = data.columns.delete(-1)
df_coef['Coef'] = coef
df_coef

hos_pre = pd.DataFrame()
hos_pre['Predict'] = line_pre
hos_pre['Truth'] = y_test.reset_index(drop=True)  # 确保索引对齐
hos_pre.plot(figsize=(18, 8))

# 预测的房价偏高，预测值大于实际值占比51.64%
len(hos_pre.query('Predict > Truth')) / len(hos_pre)

# 模型评价
plt.scatter(y_test, line_pre,label='y')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4,label='predicted')

# 在整个数据集上评价模型
line_pre_all = linear_model.predict(X)  #预测值
print('SCORE:{:.4f}'.format(linear_model.score(X,y)))
print('RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y, line_pre_all))))
# SCORE:0.7271
# RMSE:4.8223
hos_pre_all = pd.DataFrame()
hos_pre_all['Predict'] = line_pre_all
hos_pre_all['Truth'] = y
hos_pre_all.plot(figsize=(18,8))

plt.scatter(y, line_pre_all,label='y')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4,label='predicted')

# 优化模型
data.corr()['PRICE'].abs().sort_values(ascending=False).head(4)
# 由此我们得出了三个相关性最高的特征，我们将其作为自变量去建立模型
X2 = np.array(data[['LSTAT','RM','PIRATIO']])
X2_train, X2_test, y_train, y_test = train_test_split(X2, y, random_state=1,test_size=0.2)

linear_model2 = LinearRegression()
linear_model2.fit(X2_train,y_train)
print(linear_model2.intercept_)
print(linear_model2.coef_)
line2_pre = linear_model2.predict(X2_test)  #预测值
print('SCORE:{:.4f}'.format(linear_model2.score(X2_test, y_test)))#模型评分
print('RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_test, line2_pre))))#RMSE(标准误差)
# SCORE:0.5296
# RMSE:6.3355

# 对于预测测试集的数据的得分score明显是没有开始的线性回归模型1高的，然后我们再看看，在整个数据集中它的表现
line2_pre_all = linear_model2.predict(X2)  #预测值
print('SCORE:{:.4f}'.format(linear_model2.score(X2, y)))#模型评分
print('RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y, line2_pre_all))))#RMSE(标准误差)
# 第一个模型还是比第二个模型优秀
# SCORE:0.6644
# RMSE:5.3481

# 数据标准化
from sklearn.preprocessing import StandardScaler
ss_x = StandardScaler()
X_train = ss_x.fit_transform(X_train)
X_test = ss_x.transform(X_test)
ss_y = StandardScaler()
y_train = ss_y.fit_transform(y_train.values.reshape(-1, 1))
y_test = ss_y.transform(y_test.values.reshape(-1, 1))

X ,y = data[data.columns.delete(-1)], data['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

# 梯度提升,最优
from sklearn import ensemble
#params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,'learning_rate': 0.01, 'loss': 'ls'}
#clf = ensemble.GradientBoostingRegressor(**params)
clf = ensemble.GradientBoostingRegressor()
clf.fit(X_train, y_train)
clf_pre=clf.predict(X_test) #预测值
print('SCORE:{:.4f}'.format(clf.score(X_test, y_test)))#模型评分
print('RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_test, clf_pre))))#RMSE(标准误差)
# SCORE:0.9166
# RMSE:2.5896
test_data = pd.read_csv(
    R'C:\Users\cjy\PycharmProjects\PythonProject\learn_machine_learning\Boston_housing\dataset\test_dataset.csv')
test_data.info()
x_test = test_data.iloc[:, 1:]
y_submission = pd.read_csv(R'C:\Users\cjy\PycharmProjects\PythonProject\learn_machine_learning\Boston_housing\dataset\SampleSubmission.csv')#测试集
y_submission=np.squeeze(y_submission)
prediction_GB = clf.predict(x_test)
result = pd.DataFrame({'ID':y_submission['ID'].values, 'value':prediction_GB.astype(np.int32)})
result.to_csv(R"C:\Users\cjy\PycharmProjects\PythonProject\learn_machine_learning\Boston_housing\result/predictions_GB.csv", index=False)

# Lasso 回归
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score  # 确保导入了 r2_score

# 假设 X_train, X_test, y_train, y_test 已定义
lasso = Lasso()
lasso.fit(X_train, y_train)
y_predict_lasso = lasso.predict(X_test)
r2_score_lasso = r2_score(y_test, y_predict_lasso)  # 使用正确的函数名

print('SCORE:{:.4f}'.format( lasso.score(X_test, y_test)))#模型评分
print('RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_test,y_predict_lasso))))#RMSE(标准误差)
print('Lasso模型的R-squared值为:',r2_score_lasso)
# SCORE:0.6479
# RMSE:5.3203
# Lasso模型的R-squared值为: 0.647920077736116

# ElasticNet 回归
# ElasticNet回归是Lasso回归和岭回归的组合
enet = ElasticNet()
enet.fit(X_train,y_train)
y_predict_enet = enet.predict(X_test)
r2_score_enet = r2_score(y_test,y_predict_enet)

print('SCORE:{:.4f}'.format( enet.score(X_test, y_test)))#模型评分
print('RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_test,y_predict_enet))))#RMSE(标准误差)
print("ElasticNet模型的R-squared值为:",r2_score_enet)
# SCORE:0.6555
# RMSE:5.2624
# ElasticNet模型的R-squared值为: 0.6555396296415932

# Support Vector Regression (SVR)
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import r2_score as r2, mean_squared_error as mse, mean_absolute_error as mae


def svr_model(kernel):
    svr = SVR(kernel=kernel)
    svr.fit(X_train, y_train)
    y_predict = svr.predict(X_test)

    # score(): Returns the coefficient of determination R^2 of the prediction.
    print(kernel, ' SVR的默认衡量评估值值为：', svr.score(X_test, y_test))
    print(kernel, ' SVR的R-squared值为：', r2(y_test, y_predict))
    print(kernel, ' SVR的均方误差（mean squared error）为：', mse(y_test, y_predict))
    print(kernel, ' SVR的平均绝对误差（mean absolute error）为：', mae(y_test, y_predict))
    # print(kernel,' SVR的均方误差（mean squared error）为：',mse(scalery.inverse_transform(y_test), scalery.inverse_transform(y_predict)))
    # print(kernel,' SVR的平均绝对误差（mean absolute error）为：',mae(scalery.inverse_transform(y_test),scalery.inverse_transform(y_predict)))

    return svr

# linear 线性核函数
linear_svr = svr_model(kernel='linear')
# linear  SVR的默认衡量评估值值为： 0.6576483301899768
# linear  SVR的R-squared值为： 0.6576483301899768
# linear  SVR的均方误差（mean squared error）为： 27.52363803765411
# linear  SVR的平均绝对误差（mean absolute error）为： 3.3442088039251567

# poly 多项式核
poly_svr = svr_model(kernel='poly')
# poly  SVR的默认衡量评估值值为： 0.24271635926675106
# poly  SVR的R-squared值为： 0.24271635926675106
# poly  SVR的均方误差（mean squared error）为： 60.8824277998851
# poly  SVR的平均绝对误差（mean absolute error）为： 4.867734182346076

# rbf（Radial Basis Function） 径向基函数
rbf_svr = svr_model(kernel='rbf')
# rbf  SVR的默认衡量评估值值为： 0.22902950937663613
# rbf  SVR的R-squared值为： 0.22902950937663613
# rbf  SVR的均方误差（mean squared error）为： 61.98279311272342
# rbf  SVR的平均绝对误差（mean absolute error）为： 5.079009719161011

# SVM（支持向量机）回归-- 线性核
from sklearn.svm import SVR
linear_svr = SVR(kernel="linear")
linear_svr.fit(X_train, y_train)
linear_svr_pre = linear_svr.predict(X_test)#预测值
print('SCORE:{:.4f}'.format(linear_svr.score(X_test, y_test)))#模型评分
print('RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_test, linear_svr_pre))))#RMSE(标准误差)
# SCORE:0.6576
# RMSE:5.2463

# SVM（支持向量机）回归-- 多项式核
from sklearn.preprocessing import StandardScaler
ss_x = StandardScaler()
X_train = ss_x.fit_transform(X_train)
X_test = ss_x.transform(X_test)
ss_y = StandardScaler()
y_train = ss_y.fit_transform(y_train.values.reshape(-1, 1))
y_test = ss_y.transform(y_test.values.reshape(-1, 1))

poly_svr = SVR(kernel="poly")
poly_svr.fit(X_train, y_train)
poly_svr_pre = poly_svr.predict(X_test)#预测值
print('SCORE:{:.4f}'.format(poly_svr.score(X_test, y_test)))#模型评分
print('RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_test, poly_svr_pre))))#RMSE(标准误差)
# SCORE:0.7977
# RMSE:0.4339

# 决策树回归
from sklearn.tree import DecisionTreeRegressor
tree_reg=DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X_train, y_train)
tree_reg_pre = tree_reg.predict(X_test)#预测值
print('SCORE:{:.4f}'.format( tree_reg.score(X_test, y_test)))#模型评分
print('RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_test,tree_reg_pre))))#RMSE(标准误差)
# SCORE:0.6537
# RMSE:0.5676
