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

ROOT_DIR = "C:/Users/shc/Downloads/LGdata"
RANDOM_STATE = 110

# Load data
train_data = pd.read_csv(os.path.join(ROOT_DIR, "train.csv"))
train_data
column_to_modify = "HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Dam"
train_data[column_to_modify].replace("OK", np.nan, inplace=True)

df_org_train, df_val = train_test_split(
    train_data,
    test_size=0.3,
    stratify=train_data["target"],
    random_state=RANDOM_STATE,
)

normal_ratio = 1.0  # 1.0 means 1:1 ratio
# Define normal_ratios to test
normal_ratios = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
df_normal = df_org_train[df_org_train["target"] == "Normal"]
df_abnormal = df_org_train[df_org_train["target"] == "AbNormal"]

num_normal = len(df_normal)
num_abnormal = len(df_abnormal)
print(f"  Total: Normal: {num_normal}, AbNormal: {num_abnormal}")

df_normal = df_normal.sample(n=int(num_abnormal * normal_ratio), replace=False, random_state=RANDOM_STATE)
df_concat = pd.concat([df_normal, df_abnormal], axis=0).reset_index(drop=True)
df_concat.value_counts("target")
df_train = df_concat


def print_stats(df: pd.DataFrame):
    num_normal = len(df[df["target"] == "Normal"])
    num_abnormal = len(df[df["target"] == "AbNormal"])

    print(f"  Total: Normal: {num_normal}, AbNormal: {num_abnormal}" + f" ratio: {num_abnormal/num_normal}")


# Print statistics
print(f"  \tAbnormal\tNormal")
print_stats(df_train)
print_stats(df_val)

model = CatBoostClassifier(
    random_seed=RANDOM_STATE,
    verbose=False           # 학습 과정의 출력을 줄이기 위함
)

features = []


for col in df_train.columns:
    try:
        df_train[col] = df_train[col].astype(int)
        features.append(col)
    except:
        continue
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
smote = SMOTE(random_state=RANDOM_STATE)
train_x,train_y = smote.fit_resample(train_x,train_y)

model.fit(train_x, train_y)

train_y_pred = model.predict(train_x)
print_metrics(train_y, train_y_pred, "Training")

# Validation dataset metrics
val_x = df_val[features]
val_y = df_val["target"]

val_y_pred = model.predict(val_x)
print_metrics(val_y, val_y_pred, "Validation")


# test

test_data = pd.read_csv(os.path.join(ROOT_DIR, "test.csv"))
test_data[column_to_modify].replace("OK", np.nan, inplace=True)

df_test_x = test_data[features]

for col in df_test_x.columns:
    try:
        df_test_x.loc[:, col] = df_test_x[col].astype(int)
    except:
        continue

test_pred = model.predict(df_test_x)
test_pred

# 제출 데이터 읽어오기 (df_test는 전처리된 데이터가 저장됨)
df_sub = pd.read_csv(os.path.join(ROOT_DIR, "submission.csv"))
df_sub["target"] = test_pred

# 제출 파일 저장
df_sub.to_csv("submission.csv", index=False)