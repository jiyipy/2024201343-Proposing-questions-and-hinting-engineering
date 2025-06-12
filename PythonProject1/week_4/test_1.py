from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
# 加载数据集
iris = load_iris()
X = iris.data
y = iris.target
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3,random_state=42
)
# 创建预处理和建模的Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # 标准化处理
    ('knn', KNeighborsClassifier())
])
# 设置参数网格
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 11],  # 不同的k值
    'knn__weights': ['uniform', 'distance'],  # 是否启用距离加权
    'knn__p': [1, 2]  # 距离度量标准（1:曼哈顿距离，2:欧氏距离）
}
# 创建GridSearchCV对象
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,  # 5折交叉验证
    scoring='accuracy',
    n_jobs=-1
)
# 执行网格搜索
grid_search.fit(X_train, y_train)
# 获取最佳模型
best_model = grid_search.best_estimator_
# 生成预测结果
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)
# 输出评估报告
print("最佳参数组合:", grid_search.best_params_)
print("\n训练集评估报告:")
print(classification_report(y_train, y_train_pred, target_names=iris.target_names))
print("\n测试集评估报告:")
print(classification_report(y_test, y_test_pred, target_names=iris.target_names))