import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv(R'C:\Users\cjy\PycharmProjects\PythonProject\learn_machine_learning\Titanic\dataset\train.csv')
test = pd.read_csv(R'C:\Users\cjy\PycharmProjects\PythonProject\learn_machine_learning\Titanic\dataset\test.csv')

train['Age']=train['Age'].fillna(train['Age'].mean()) # 用平均值填充
train.drop(['Cabin'],axis=1,inplace=True) # 删去Cabin的那一列数据
train.Embarked = train.Embarked.fillna('S') # 用’S'填补缺失值

test['Age']=test['Age'].fillna(test['Age'].mean())
test['Fare']=test['Fare'].fillna(test['Fare'].mean())
test.drop('Cabin',axis=1,inplace=True)

dataset = pd.concat([train, test], ignore_index=True, sort=False)

sexdict = {'male':1, 'female':0}
dataset.Sex = dataset.Sex.map(sexdict)

embarked2 = pd.get_dummies(dataset.Embarked, prefix = 'Embarked')

dataset = pd.concat([dataset,embarked2], axis = 1) ## 将编码好的数据添加到原数据上
dataset.drop(['Embarked'], axis = 1, inplace=True) ## 过河拆桥

pclass = pd.get_dummies(dataset.Pclass, prefix = 'Pclass')

dataset = pd.concat([dataset,pclass], axis = 1)
dataset.drop(['Pclass'], axis = 1, inplace=True)

dataset['family']=dataset.SibSp+dataset.Parch+1

dataset.drop(['Ticket','Name'], axis = 1, inplace=True)
x_train = dataset.iloc[0:891, :]
y_train = x_train.Survived
x_train.drop(['Survived'], axis=1, inplace =True)

x_test = dataset.iloc[891:, :]
x_test.drop(['Survived'], axis=1, inplace =True)

y_test = pd.read_csv(R'C:\Users\cjy\PycharmProjects\PythonProject\learn_machine_learning\Titanic\dataset\gender_submission.csv')#测试集
y_test=np.squeeze(y_test)

x_train.shape,y_train.shape,x_test.shape, y_test.shape

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression()

model.fit(x_train.iloc[0:-100,:],y_train.iloc[0:-100])
accuracy_score(model.predict(x_train.iloc[-100:,:]),y_train.iloc[-100:].values.reshape(-1,1))

prediction1 = model.predict(x_test)
result = pd.DataFrame({'PassengerId':y_test['PassengerId'].values, 'Survived':prediction1.astype(np.int32)})
result.to_csv("./results/predictions1.csv", index=False)

model2 = LogisticRegression()
model2.fit(x_train,y_train)

prediction2 = model2.predict(x_test)

result = pd.DataFrame({'PassengerId':y_test['PassengerId'].values, 'Survived':prediction2.astype(np.int32)})
result.to_csv("./results/predictions2.csv", index=False)

# 随机森林
from sklearn.ensemble import RandomForestClassifier
model3 = RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=5, min_samples_split=2,
                                min_samples_leaf=1, max_features='sqrt',    bootstrap=False, oob_score=False, n_jobs=1, random_state=0,
                                verbose=0)
model3.fit(x_train, y_train)
prediction3 =  model3.predict(x_test)

result = pd.DataFrame({'PassengerId':y_test['PassengerId'].values, 'Survived':prediction3.astype(np.int32)})
result.to_csv("./results/predictions3.csv", index=False)

from sklearn.tree import DecisionTreeClassifier
model4 = DecisionTreeClassifier(criterion='entropy', max_depth=7, min_impurity_decrease=0.0)
model4.fit(x_train, y_train)
prediction4 = model4.predict(x_test)

result = pd.DataFrame({'PassengerId':y_test['PassengerId'].values, 'Survived':prediction4.astype(np.int32)})
result.to_csv("./results/predictions4.csv", index=False)

prediciton_vote = pd.DataFrame({'PassengerId': y_test['PassengerId'],
                                 'Vote': prediction1.astype(int)+prediction2.astype(int)+prediction3.astype(int)})
vote = { 0:False,1:False,2:True,3:True}

prediciton_vote['Survived']=prediciton_vote['Vote'].map(vote)

prediciton_vote.head()

result = pd.DataFrame({'PassengerId':y_test['PassengerId'].values, 'Survived':prediciton_vote.Survived.astype(np.int32)})
result.to_csv("./results/predictions5.csv", index=False)
