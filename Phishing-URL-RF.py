# -*- coding: utf-8 -*-
"""Phishing-url-randomforest

Automatically generated by Colab.

# **import**
"""

import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

"""# **데이터셋 불러오기**"""

df = pd.read_csv('path')

df_1 = df.drop(columns=['FILENAME', 'URL', 'Domain','Title','TLD'])
df_1

"""# **데이터 상관관계 확인하기**"""

plt.figure(figsize=(30,16)) #상관관계 -1, 1에 가까울수록 상관관계가 높음 0에 가까울수록 관계없음
sns.heatmap(df_1.corr(), cmap=sns.color_palette("coolwarm", 10), annot=df_1.corr())

"""# **훈련용 테스트용 나누기**"""

X = df_1.drop(columns=['label'])
X

y = df_1['label'].values
y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, stratify=y)

#파라미터 설정 및 모델 훈련
dt = RandomForestClassifier(criterion='gini', max_depth=5)
dt.fit(X_train, y_train)

"""# **모델 점수확인**"""

#모델 점수 확인하기
from sklearn.metrics import accuracy_score, f1_score
y_train_pred = dt.predict(X_train)
y_test_pred = dt.predict(X_test)

#accuracy_scores
print("- Accuracy (Train)           :  {:.4}". format(accuracy_score(y_train, y_train_pred)))
print("- Accuracy (Test) : {:.4}".format(accuracy_score(y_test, y_test_pred)))
#f1_scores
print("- F1 score (Train)           :  {:.4}".format(f1_score(y_train, y_train_pred)))
print("- F1 score (Test) : {:.4}".format(f1_score(y_test, y_test_pred)))

"""# **정밀도 확인**"""

from sklearn.metrics import precision_score
#정밀도를 보는 이유 = 스펨 메일 같은 여부를 판단을 하는 경우 정밀도가 중요하다 (실제로 postive하는지 본다)
precision = precision_score(y_test, y_test_pred)
print("랜덤포레스트 정밀도:", precision)

#혼동행렬 시각화 Confusion Matrix 시각화
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

#실제값과 예측의 산점도 시각화
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')  # 대각선
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.show()

"""# **교차검증**"""

start_time = time.time()
cross_val = cross_validate(
    dt,
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

"""# **피처 중요도**"""

dt_importances =dt.feature_importances_
feature_names = [f'Feature {i}' for i in range(X.shape[1])]
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': dt_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

"""# **AUC커브 ROC커브**"""

#예측 및 양성 클래스의 예측 확률 얻기
y_roc_pred = dt.predict_proba(X_test)[:, 1]


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
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

print("Random Forest AUC Score :", auc_score)

"""# **하이퍼 파라미터 튜닝**"""
# 그리드 서치로 하이퍼 파라미터 튜닝 세팅
from sklearn.model_selection import GridSearchCV


# Random Forest 모델 생성
Rf = RandomForestClassifier(random_state=10)

start_time = time.time()
# 탐색할 하이퍼파라미터 범위 설정
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 30,],
    'min_samples_split': [2, 3, 5, 7, 10],
    'min_samples_leaf': [10, 20, 30]
}


# GridSearchCV를 사용하여 하이퍼파라미터 탐색
grid_search = GridSearchCV(estimator=Rf, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

end_time = time.time()
total_Grid_search_time = end_time - start_time
# 최적의 하이퍼파라미터와 성능 출력
total_Grid_search_time_minutes = total_Grid_search_time / 60
print("총 하이퍼 파라미터 튜닝 시간(분) :", total_Grid_search_time_minutes)
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

"""# **파라미터 튜닝 후 최종 모델 선정**"""

best_tree = grid_search.best_estimator_
best_tree

best_tree.fit(X_train, y_train)

y_pred = best_tree.predict(X_test)

print("------최종 test값으로 모델 성능평가-----------")
print('잘못 분류된 샘플 개수: %d' % (y_test != y_pred).sum())
print('정확도: %.3f' % accuracy_score(y_test, y_pred))
print('정밀도: %.3f' % precision_score(y_test, y_pred))
print('재현율: %.3f' % recall_score(y_test, y_pred))
print('F1: %.3f' % f1_score(y_test, y_pred))

# 하이퍼 파라미터 튜닝 후 Auc , Roc 출력
y_best_roc_pred = best_tree.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_best_roc_pred)
auc_score = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='Best_RF-ROC (AUC = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Best_Rf (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

print("Random Forest AUC Score :", auc_score)
#하이퍼 파라미터 튜닝 후 혼동행렬 출력


conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Best_RF-Confusion Matrix')
plt.show()