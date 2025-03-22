import pandas as pd
# 读取数据
train = pd.read_csv(R'C:\Users\cjy\PycharmProjects\PythonProject\learn_machine_learning\Allstate_Claims_Severity\datasets\train.csv')
test = pd.read_csv(R'C:\Users\cjy\PycharmProjects\PythonProject\learn_machine_learning\Allstate_Claims_Severity\datasets\test.csv')

# 统计离散型变量的列名
cat_features = list(train.select_dtypes(include=['object']).columns)

# 统一编码 train 和 test 数据集中的分类变量
for c in cat_features:
    # 合并 train 和 test 以保证类别一致
    combined = pd.concat([train[c], test[c]], axis=0).astype('category')

    # 将 train 和 test 按照相同类别编码
    train[c] = train[c].astype('category')
    train[c] = train[c].cat.set_categories(combined.cat.categories).cat.codes

    test[c] = test[c].astype('category')
    test[c] = test[c].cat.set_categories(combined.cat.categories).cat.codes
# 确保所有特征都是数值型
print(train.dtypes)  # 检查是否还有 object 类型
print(test.dtypes)

# 训练 XGBoost 模型
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_absolute_error as mae

X = train.drop(columns=['loss'])  # 删除目标变量 loss
y = train['loss']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练 XGBoost
reg = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01)
reg.fit(X_train, y_train)

# 预测并计算误差
reg_predict = reg.predict(X_test)
print('MAE = {}'.format(mae(reg_predict, y_test)))

# 预测 test 数据集
test_pred = reg.predict(test)

# 生成提交文件
submission = pd.DataFrame({
    "id": test["id"],  # 确保使用 test 的 ID
    "loss": test_pred
})
submission.to_csv(R'C:\Users\cjy\PycharmProjects\PythonProject\learn_machine_learning\Allstate_Claims_Severity\result\Submission_xgboost.csv', index=False)
