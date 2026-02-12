import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv("diabetes.csv")

zero_missing_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in zero_missing_cols:
    zero_count = (df[col] == 0).sum()
    zero_percent = (zero_count / len(df)) * 100
    print(f"{col:20s}: {zero_percent:6.2f}%")

df_a = df.copy()  # Mean
df_b = df.copy()  # GroupMean
df_c = df.copy()  # KNN

for col in zero_missing_cols:
    df_a[col] = df_a[col].replace(0, np.nan)
    df_b[col] = df_b[col].replace(0, np.nan)
    df_c[col] = df_c[col].replace(0, np.nan)

for col in zero_missing_cols:
    nan_count = df_a[col].isna().sum()
    zero_percent = (nan_count / len(df_a)) * 100
    print(f"{col:20s}: {zero_percent:6.2f}%")

original_glucose = df_a["Glucose"].astype(float).mean(skipna=True)
print(f"original_glucose: {original_glucose:6.2f}")

imputer_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
df_a[zero_missing_cols] = imputer_mean.fit_transform(df_a[zero_missing_cols].astype(float))
# print("type :\n", df_a[zero_missing_cols].columns.dtype)
print(f"imputer_mean: {df_a[zero_missing_cols].astype(float):6.2f}")

new_glucose_mean = df_b["Glucose"].mean(skipna=True)
print(f"new_glucose_mean: {new_glucose_mean:6.2f}")
