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

df = pd.read_csv("C:\\Users\\viroo\\PycharmProjects\\AI_HW\\HW11\\diabetes.csv")

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

original_glucose_mean = df["Glucose"].mean(skipna=True)
print(f"original_glucose: {original_glucose_mean:6.2f}")

imputer_mean = SimpleImputer(strategy="mean")
print("befor Glucose null: ", df_a[zero_missing_cols].isna().sum().sum())
df_a[zero_missing_cols] = imputer_mean.fit_transform(df_a[zero_missing_cols])
print(f"Remaining NaN values: {df_a[zero_missing_cols].isna().sum().sum()}")
# todo
new_glucose_mean = df_a['Glucose'].mean()
print(f"New Glucose mean after imputation: {new_glucose_mean:.2f}")
print(f"Change in Glucose mean: {new_glucose_mean - original_glucose_mean:.2f}")
imputer_knn = KNNImputer(n_neighbors=5)
print("before: ", df_c[zero_missing_cols].isna().sum().sum())
df_c[zero_missing_cols] = imputer_knn.fit_transform(df_c[zero_missing_cols])
print("after: ", df_c[zero_missing_cols].isna().sum().sum())

df_fe = df_c.copy()


def categorize_glucose(x):
    if x < 100:
        return "normal"
    elif x < 125:
        return "prediabetes"
    else:
        return "diabetic"


df_fe["Glucose_Category"] = df_fe["Glucose"].apply(categorize_glucose)
[print("for ", a, "len is ", len(df_fe[df_fe["Glucose_Category"] == a])) for a in ["normal", "prediabetes", "diabetic"]]

glucose_analysis = df_fe.groupby("Glucose_Category")["Outcome"].mean() * 100

print("glucose_analysis:", glucose_analysis)
[print(f"{category:15s}: {percentage:.1f}%") for category, percentage in glucose_analysis.items()]

# plt.figure(figsize=(10, 8))
# sns.countplot(
#     x="Glucose_Category",
#     hue="Outcome",
#     data=df_fe,
# )
# plt.title("Glucose Outcome Distribution")
# plt.xlabel("Glucose Outcome")
# plt.ylabel("Count")
# plt.legend(["None-Diabetic", "Diabetic"])
# plt.tight_layout()
# plt.show()


def categorize_bmi(x):
    if x < 18.50:
        return "Underweight"
    elif x < 25:
        return "Normal"
    elif x < 30:
        return "Overweight"
    else:
        return "Obese"


df_fe["BMI_Category"] = df_fe["BMI"].apply(categorize_bmi)

bmi_analysis = df_fe.groupby("BMI_Category")["Outcome"].mean() * 100
[print(f"{category:12}: {percentage:12.2f}")
 for category, percentage in bmi_analysis.items()]

max_bmi_category = bmi_analysis.idxmax()
print(f"max_bmi_category: {max_bmi_category:>12}({bmi_analysis[max_bmi_category]:.2f}%)")


def categorize_age(x):
    if x < 20:
        return 'Young'
    elif x <= 40:
        return 'Middle_Aged'
    elif x <= 60:
        return 'Senior'
    else:
        return 'Elderly'


df_fe['Aged_Group'] = df_fe["Age"].apply(categorize_age)
age_analysis = df_fe.groupby('Aged_Group')['Outcome'].mean() * 100
print("age_analysis:")
[print(f"{category:15s}:{percentage:.2f}") for category, percentage in age_analysis.items()]
print(age_analysis.sum())

total_patient = df_fe[df_fe["Outcome"] == 1].shape[0]
age_distribution = df_fe[df_fe["Outcome"] == 1].groupby('Aged_Group').size() / total_patient * 100
print(f"age_distribution: {age_distribution}")

df_fe["Insulin_to_Glucose_Ratio"] = df_fe["Insulin"] / df_fe["Glucose"]

correlation = df_fe["Insulin_to_Glucose_Ratio"].corr(df_fe["Outcome"])
print(f"correlation with outcome: {correlation:.2f}")


def category_bp(x):
    if x < 80:
        return "Low"
    elif x <= 90:
        return "Normal"
    else:
        return "High"


df_fe["BP_Status"] = df_fe["BloodPressure"].apply(category_bp)

total_diabetic_patients = df_fe['Outcome'].sum()
print(f"total_diabetic_patients: {total_diabetic_patients}")
diabetic_distribution = df_fe[df_fe['Outcome'] == 1].groupby('BP_Status').size()
print(f"diabetic_distribution: {diabetic_distribution}")
diabetic_distribution_pct = (diabetic_distribution / total_diabetic_patients) * 100
for category, percentage in diabetic_distribution_pct.items():
    print(f"{category:15s}: {percentage:.1f}%")


def prepare_data(df_input, strategy_name):
    df_model = df_input.copy()

    df_model["Glucose_Category"] = df_model["Glucose"].apply(categorize_glucose)
    df_model["BMI_Category"] = df_model["BMI"].apply(categorize_bmi)
    df_model["Age_Group"] = df_model["Age"].apply(categorize_age)
    df_model["Insulin_to_Glucose_Ratio"] = df_model["Insulin"] / df_model["Glucose"]
    df_model["BP_Status"] = df_model["BloodPressure"].apply(category_bp)

    categorical_cols = ['Glucose_Category', 'BMI_Category', 'Age_Group', 'BP_Status']
    df_encoded = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

    X = df_encoded.drop('Outcome', axis=1)
    y = df_encoded['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    continues_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
                       'Insulin_to_Glucose_Ratio']
    
    scalre = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[continues_cols] = scalre.fit_transform(X_train[continues_cols])
    X_test_scaled[continues_cols] = scalre.transform(X_train[continues_cols])
    print(f"Train shape: {X_train_scaled.shape}")
    print(f"Test shape: {X_test_scaled.shape}")
    print(f"Using stratify=y ensures class distribution is preserved in train/test splits")

    return X_train_scaled, X_train_scaled, y_train

strategies = {
    'A': (df_a, "Simple Mean Imputation"),
    'B': (df_b, "Group Mean Imputation"),
    'C': (df_c, "KNN Imputation")
}

results = {}

for strategy_key, (df_strategy, strategy_name) in strategies.items():
    print(f"STRATEGY {strategy_key}: {strategy_name}")
    X_train, X_test,y_train,y_test = prepare_data(df_strategy, strategy_name)