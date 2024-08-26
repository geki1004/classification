import os
from pprint import pprint
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    make_scorer,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

ROOT_DIR = "C:/Users/shc/Downloads/LGdata"  # 이 안에 train, test, submission csv 데이터 전부 넣어 사용
RANDOM_STATE = 110

# train.csv 불러옴
train_data = pd.read_csv(os.path.join(ROOT_DIR, "train.csv"))
train_data
column_to_modify = "HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Dam"
train_data[column_to_modify].replace("OK", np.nan, inplace=True)

df_org_train, df_val = train_test_split(
    # train과 val을 언더샘플링 전에 미리 나눠둠(val data를 test data처럼 unbalance한 상태 유지, train data만 언더샘플링 하기 위함)
    train_data,
    test_size=0.3,
    stratify=train_data["target"],
    random_state=RANDOM_STATE,
)

normal_ratio = 8.0  # 1.0 means 1:1 ratio, 언더샘플링 과정
# Define normal_ratios to test

df_normal = train_data[train_data["target"] == "Normal"] #train data를 val로 따로 빼지 않은 더 많은 데이터셋 사용용
df_abnormal = train_data[train_data["target"] == "AbNormal"]

num_normal = len(df_normal)
num_abnormal = len(df_abnormal)
print(f"  Total: Normal: {num_normal}, AbNormal: {num_abnormal}")

#df_normal = df_normal.sample(n=int(num_abnormal * normal_ratio), replace=False, random_state=RANDOM_STATE) #언더샘플 제거
df_concat = pd.concat([df_normal, df_abnormal], axis=0).reset_index(drop=True)
df_concat.value_counts("target")
df_train = df_concat


def print_stats(df: pd.DataFrame):
    num_normal = len(df[df["target"] == "Normal"])
    num_abnormal = len(df[df["target"] == "AbNormal"])

    print(f"  Total: Normal: {num_normal}, AbNormal: {num_abnormal}" + f" ratio: {num_abnormal / num_normal}")


# Print statistics
print(f"  \tAbnormal\tNormal")
print_stats(df_train)
print_stats(df_val)

# 모델 지정, 파리미터 수정
model = CatBoostClassifier(
    iterations=5000,  # Number of trees
    depth=10,  # Dep6th of the trees
    learning_rate=0.01,  # Learning rate
    class_weights=[10, 1],  # Class weights to handle imbalance
    random_seed=RANDOM_STATE,
    verbose=False  # Reduce verbosity
)

features = []

for col in df_train.columns:
    try:
        df_train[col] = df_train[col].astype(int)
        features.append(col)
    except:
        continue


# 혼동행렬 및 precision, recall, f1score 등의 결과를 나타내기 위함
def print_metrics(y_true, y_pred, dataset_name):
    print(f"Metrics for {dataset_name} dataset:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["AbNormal", "Normal"]))
    print("\n")


# Define the custom grid search

# Train dataset metrics
train_x = df_train[features]
train_y = df_train["target"]
# 오버샘플링 과정, Normal과 Abnormal의 개수 맞춰줌
#smote = SMOTE(random_state=RANDOM_STATE)
#train_x, train_y = smote.fit_resample(train_x, train_y)

# 모델 학습
model.fit(train_x, train_y)

# 예측값 뽑아냄
train_y_pred = model.predict(train_x)

# 실제 타겟과 예측값으로 성능지표 측정
print_metrics(train_y, train_y_pred, "Training")

# Validation dataset metrics
val_x = df_val[features]
val_y = df_val["target"]

# 검증데이터로 예측 및 성능지표 측정정
val_y_pred = model.predict(val_x)
print_metrics(val_y, val_y_pred, "Validation")

# test

test_data = pd.read_csv(os.path.join(ROOT_DIR, "test.csv"))
#test_data[column_to_modify].replace("OK", np.nan, inplace=True)

df_test_x = test_data[features]

for col in df_test_x.columns:
    try:
        df_test_x.loc[:, col] = df_test_x[col].astype(int)
    except:
        continue
# 학습한 모델로 예측
test_pred = model.predict(df_test_x)
test_pred

# 제출 데이터 읽어오기 (df_test는 전처리된 데이터가 저장됨)
df_sub = pd.read_csv(os.path.join(ROOT_DIR, "submission.csv"))
df_sub["target"] = test_pred

# 제출 파일 저장
df_sub.to_csv("submission.csv", index=False)
