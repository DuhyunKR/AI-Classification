# -*- coding: utf-8 -*-
"""XGB

# Google Colab
# **import**
"""

import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
import xgboost as xgb

from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import recall_score

"""# **csv파일 불러오기**"""

df = pd.read_csv('path')
df.shape
df['label'].value_counts()

# 모델에 넣을 파일 정리
df_1 = df.drop(columns=['FILENAME', 'URL', 'Domain','Title','TLD'])
df_1

"""# **x값 y값 나누기**"""

X = df_1.drop(columns=['label']) # 타겟값
X

y = df_1['label'].values
y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=156)

#모델 훈련
XGB = XGBClassifier(max_depth=5, reg_lambda=0.1)  #파라미터 설정
XGB.fit(X_train, y_train)

y_train_pred = XGB.predict(X_train)
y_test_pred = XGB.predict(X_test)

"""# **평가하기**"""

#Accuracy
print("- Accuracy (Train)           :  {:.4}". format(accuracy_score(y_train, y_train_pred)))
print("- Accuracy (Test) : {:.4}".format(accuracy_score(y_test, y_test_pred)))
#f1_scores
print("- F1 score (Train)           :  {:.4}".format(f1_score(y_train, y_train_pred)))
print("- F1 score (Test) : {:.4}".format(f1_score(y_test, y_test_pred)))

"""# **교차검증**"""

start_time = time.time()
cross_val = cross_validate(
    XGB,
    X,
    y,
    cv=5,
    return_train_score=True,
    scoring='accuracy'
)
end_time = time.time()
total_cross_validation_time = end_time - start_time

#전체 교차 검증 시간 출력
total_cross_validation_time_minutes = total_cross_validation_time / 60
print("총 교차 검증 시간(분) :", total_cross_validation_time_minutes)
# 전체 교차 검증 점수 출력
print("훈련 점수:", np.mean(cross_val['train_score']))
print("검증 점수:", np.mean(cross_val['test_score']))

"""# **정밀도**"""

from sklearn.metrics import precision_score
precision = precision_score(y_test, y_test_pred) 
print("XGB 정밀도:", precision) # 정밀도

"""# **혼동행렬**"""

#혼동행렬 시각화 Confusion Matrix 시각화
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

"""# **피처 중요도**"""

fig, ax = plt.subplots(figsize=(20, 24))
plot_importance(XGB, ax=ax)

XGB_importances =XGB.feature_importances_
feature_names = [f'Feature {i}' for i in range(X.shape[1])]
feature_importance_XGB = pd.DataFrame({'Feature': feature_names, 'Importance': XGB_importances})
feature_importance_XGB = feature_importance_XGB.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_XGB, palette='viridis')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

"""# **AUC ROC 커브**"""

y_roc_pred = XGB.predict_proba(X_test)[:, 1]


#ROC 곡선 계산
fpr, tpr, thresholds = roc_curve(y_test, y_roc_pred)

#AUC 계산
auc_score = auc(fpr, tpr)

#ROC 곡선 시각화
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGB- (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

print("XGB AUC Score :", auc_score)

"""# **하이퍼 파라미터 튜닝**"""
#하이퍼 파라미터 튜닝 세팅
from sklearn.model_selection import GridSearchCV


# XGB 모델 생성
Best_XGB = XGBClassifier(random_state=10)

start_time = time.time()
# 탐색할 하이퍼파라미터 범위 설정
param_grid = {

    'max_depth': [None, 10, 30,],
    'min_child_weight': [1 ,2, 3, 5, 7, 10],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'gamma':[0, 0.1, 0.2, 0.3]
}


# GridSearchCV를 사용하여 하이퍼파라미터 탐색
grid_search = GridSearchCV(estimator=Best_XGB, param_grid=param_grid,
                           scoring='accuracy', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

end_time = time.time()
total_Grid_search_time = end_time - start_time
# 최적의 하이퍼파라미터와 성능 출력
total_Grid_search_time_minutes = total_Grid_search_time / 60
print("총 하이퍼 파라미터 튜닝 시간(분) :", total_Grid_search_time_minutes)
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

#최적의 하이퍼 파라미터 모델에 넣기
best_XGB = XGBClassifier(colsample_bytree=0.7, gamma=0, max_depth=None, min_child_weight=1, subsample=1)
best_XGB.fit(X_train, y_train)

y_pred = best_XGB.predict(X_test)

print("------최종 XGB_test값으로 모델 성능평가-----------")
print('잘못 분류된 샘플 개수: %d' % (y_test != y_pred).sum())
print('정확도: %.3f' % accuracy_score(y_test, y_pred))
print('정밀도: %.3f' % precision_score(y_test, y_pred))
print('재현율: %.3f' % recall_score(y_test, y_pred))
print('F1: %.3f' % f1_score(y_test, y_pred))


# 하이퍼 파라미터 튜닝 후 XGB AUC Score 출력 
y_best_roc_pred = best_XGB.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_best_roc_pred)
auc_score = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='Best_XGB-ROC (AUC = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Best_XGB (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

print("Best_XGB AUC Score :", auc_score)
# 하이퍼 파라미터 튜닝 후 혼동행렬
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Best_XGB-Confusion Matrix')
plt.show()

"""# **하이퍼 파라미터 후 퓨처 중요도**"""

dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtest = xgb.DMatrix(data=X_test, label=y_test)

## xgboost 주요 하이퍼 파라미터 설정하기
## 하이퍼 파라미터는 매우 중요하기 때문에
param = {
    'max_depth': None,
    'eta': 0.3,
    'min_child_weight': 1,
    'gamma': 0,
    'sub_sample': 0.5,
    'colsample_bytree': 0.55
    }

num_rounds = 500

# train 데이터 세트는 'train', evaluation(test) 데이터 세트는 'eval' 로 명기
wlist = [(dtrain, 'train'), (dtest,'eval')]
# 하이퍼 파라미터와 early stopping 파라미터를 train() 함수의 파라미터로 전달
xgb_model = xgb.train(params = param, dtrain=dtrain, num_boost_round=num_rounds, evals=wlist)

fig, ax = plt.subplots(figsize=(20, 24))
plot_importance(xgb_model, ax=ax)
