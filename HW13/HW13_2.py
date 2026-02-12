from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, precision_score, balanced_accuracy_score, roc_auc_score, \
    classification_report
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

print("=" * 80)
print("بخش 1 – داده اولیه و مدل پایه")
print("=" * 80)

# ساخت دیتاست نامتوازن
X0 = np.random.normal(0, 1, (900, 2))  # کلاس 0 با 900 نمونه، 2 ویژگی
X1 = np.random.normal(2, 1, (100, 2))  # کلاس 1 با 100 نمونه، 2 ویژگی

X = np.vstack([X0, X1])
y = np.array([0] * 900 + [1] * 100)

# نمایش توزیع کلاس‌ها
print(f"توزیع کلاس‌ها در کل دیتاست: {np.bincount(y)}")
print(f"درصد کلاس اقلیت: {100 * np.mean(y):.2f}%\n")

# تقسیم به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print(f"تعداد نمونه‌های آموزش: {len(X_train)}")
print(f"توزیع کلاس‌ها در آموزش: {np.bincount(y_train)}")
print(f"تعداد نمونه‌های تست: {len(X_test)}")
print(f"توزیع کلاس‌ها در تست: {np.bincount(y_test)}\n")

# مدل پایه - RandomForest
model_base = RandomForestClassifier(random_state=42, n_estimators=100)
model_base.fit(X_train, y_train)

# پیش‌بینی و ارزیابی مدل پایه
y_pred_base = model_base.predict(X_test)
y_prob_base = model_base.predict_proba(X_test)[:, 1]

print("📊 نتایج مدل پایه (بدون SMOTE):")
print("Confusion Matrix:")
cm_base = confusion_matrix(y_test, y_pred_base)
print(cm_base)
print(f"Recall (کلاس مثبت): {recall_score(y_test, y_pred_base):.4f}")
print(f"Precision (کلاس مثبت): {precision_score(y_test, y_pred_base):.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred_base):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_prob_base):.4f}")
print("-" * 80)

print("\n" + "=" * 80)
print("بخش 2 – اعمال SMOTE")
print("=" * 80)

# اعمال SMOTE فقط روی داده آموزش
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"تعداد نمونه‌های آموزش بعد از SMOTE: {len(X_train_smote)}")
print(f"توزیع کلاس‌ها در آموزش بعد از SMOTE: {np.bincount(y_train_smote)}\n")

# مدل با SMOTE
model_smote = RandomForestClassifier(random_state=42, n_estimators=100)
model_smote.fit(X_train_smote, y_train_smote)

# ارزیابی مدل با SMOTE
y_pred_smote = model_smote.predict(X_test)
y_prob_smote = model_smote.predict_proba(X_test)[:, 1]

print("📊 نتایج مدل با SMOTE:")
print("Confusion Matrix:")
cm_smote = confusion_matrix(y_test, y_pred_smote)
print(cm_smote)
recall_smote = recall_score(y_test, y_pred_smote)
precision_smote = precision_score(y_test, y_pred_smote)
print(f"Recall (کلاس مثبت): {recall_smote:.4f}")
print(f"Precision (کلاس مثبت): {precision_smote:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred_smote):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_prob_smote):.4f}")

# مقایسه مفهومی
print("\n📈 مقایسه مفهومی SMOTE با مدل پایه:")
print(f"✅ بهبود در Recall: {recall_smote - recall_score(y_test, y_pred_base):+.4f}")
print(f"❌ تغییر در Precision: {precision_smote - precision_score(y_test, y_pred_base):+.4f}")
print("🔍 تفسیر: SMOTE باعث افزایش Recall (پیدا کردن نمونه‌های مثبت بیشتر) شده،")
print("   اما Precision کاهش یافته (یعنی False Positive افزایش پیدا کرده).")
print("-" * 80)

print("\n" + "=" * 80)
print("بخش 3 – تغییر Threshold بعد از SMOTE")
print("=" * 80)

# آستانه‌های مختلف
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
recalls = []
precisions = []
threshold = 0.3  # آستانه دلخواه

y_pred_custom = (y_prob_smote >= threshold).astype(int)
cm_custom = confusion_matrix(y_test, y_pred_custom)

print(f"📊 نتایج با آستانه {threshold}:")
print("Confusion Matrix:")
print(cm_custom)
recall_custom = recall_score(y_test, y_pred_custom)
precision_custom = precision_score(y_test, y_pred_custom)
print(f"Recall (کلاس مثبت): {recall_custom:.4f}")
print(f"Precision (کلاس مثبت): {precision_custom:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred_custom):.4f}")

print(f"\n📈 مقایسه با آستانه پیش‌فرض 0.5:")
print(f"Recall افزایش یافت: {recall_custom:.4f} > {recall_smote:.4f}")
print(f"Precision کاهش یافت: {precision_custom:.4f} < {precision_smote:.4f}")
print("🔍 تفسیر: با کاهش آستانه، Recall افزایش و Precision کاهش می‌یابد.")
print("   این trade-off بین Recall و Precision است.")

# نمایش تأثیر آستانه‌های مختلف
print("\n📊 تأثیر آستانه‌های مختلف روی Recall و Precision:")
for thresh in thresholds[::2]:  # نمایش هر دو آستانه یکی در میان
    y_tmp = (y_prob_smote >= thresh).astype(int)
    rec = recall_score(y_test, y_tmp)
    prec = precision_score(y_test, y_tmp)
    print(f"   Threshold={thresh}: Recall={rec:.4f}, Precision={prec:.4f}")

print("-" * 80)

print("\n" + "=" * 80)
print("بخش 4 – شبیه‌سازی Drift")
print("=" * 80)

# شبیه‌سازی Drift روی داده تست
print("🔄 اعمال Feature Drift: اضافه کردن 1.5 واحد به میانگین ویژگی‌ها")
X_test_drift = X_test.copy()
X_test_drift[:, 0] += 1.5  # افزایش میانگین ویژگی اول
X_test_drift[:, 1] += 1.5  # افزایش میانگین ویژگی دوم

# ارزیابی مدل SMOTE روی داده Drift
y_pred_drift = model_smote.predict(X_test_drift)
y_prob_drift = model_smote.predict_proba(X_test_drift)[:, 1]

print("\n📊 نتایج مدل SMOTE روی داده Drift:")
cm_drift = confusion_matrix(y_test, y_pred_drift)
print("Confusion Matrix:")
print(cm_drift)
recall_drift = recall_score(y_test, y_pred_drift)
precision_drift = precision_score(y_test, y_pred_drift)
print(f"Recall (کلاس مثبت): {recall_drift:.4f}")
print(f"Precision (کلاس مثبت): {precision_drift:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred_drift):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_prob_drift):.4f}")

print(f"\n📉 کاهش عملکرد نسبت به قبل از Drift:")
print(f"   کاهش Recall: {recall_smote - recall_drift:.4f}")
print(
    f"   کاهش Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred_smote) - balanced_accuracy_score(y_test, y_pred_drift):.4f}")

# پاسخ به سؤالات
print("\n🔍 پاسخ به سؤالات بخش 4:")
print("1. این drift از چه نوعی است؟")
print("   ✅ این یک Feature Drift (یا Covariate Shift) است.")
print("   دلیل: توزیع ویژگی‌های ورودی (X) تغییر کرده، اما رابطه بین X و y ثابت مانده.")
print("   (یعنی p(x) تغییر کرده ولی p(y|x) ثابت است)")

print("\n2. چرا SMOTE کمکی به این مشکل نمیکند؟")
print("   ❌ SMOTE فقط توزیع کلاس‌ها را در داده‌های آموزش متوازن می‌کند.")
print("   ❌ SMOTE هیچ تغییری در مقاومت مدل نسبت به تغییر توزیع ویژگی‌ها ایجاد نمی‌کند.")
print("   ❌ مشکل Drift مربوط به تغییر توزیع داده‌های ورودی است، نه نامتوازنی کلاس‌ها.")
print("   ✅ راه‌حل Drift: آموزش مجدد مدل، Domain Adaptation، یا استفاده از داده‌های جدیدتر")

print("\n" + "=" * 80)
print("📌 جمع‌بندی نهایی")
print("=" * 80)
print("""
1. SMOTE:
   - ✅ افزایش Recall و Balanced Accuracy
   - ❌ کاهش Precision (افزایش False Positive)

2. تغییر Threshold:
   - ✅ امکان تنظیم trade-off بین Recall و Precision
   - ✅ بهبود Recall با کاهش آستانه
   - ❌ کاهش Precision با کاهش آستانه

3. Drift:
   - ❌ SMOTE در برابر Drift مقاومت ایجاد نمی‌کند
   - ❌ عملکرد مدل به شدت کاهش می‌یابد
   - ✅ نیاز به راه‌حل‌های دیگر مانند retraining یا domain adaptation
""")

# نمایش بصری (اختیاری)
try:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # داده‌های اصلی
    axes[0, 0].scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], alpha=0.5, label='کلاس 0', c='blue')
    axes[0, 0].scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], alpha=0.5, label='کلاس 1', c='red')
    axes[0, 0].set_title('داده‌های تست اصلی')
    axes[0, 0].legend()

    # داده‌های Drift
    axes[0, 1].scatter(X_test_drift[y_test == 0, 0], X_test_drift[y_test == 0, 1], alpha=0.5, label='کلاس 0', c='blue')
    axes[0, 1].scatter(X_test_drift[y_test == 1, 0], X_test_drift[y_test == 1, 1], alpha=0.5, label='کلاس 1', c='red')
    axes[0, 1].set_title('داده‌های تست با Drift')
    axes[0, 1].legend()

    # مقایسه Recall
    axes[1, 0].bar(['پایه', 'SMOTE', 'SMOTE+Threshold', 'Drift'],
                   [recall_score(y_test, y_pred_base), recall_smote, recall_custom, recall_drift])
    axes[1, 0].set_title('مقایسه Recall')
    axes[1, 0].set_ylabel('Recall')

    # مقایسه Precision
    axes[1, 1].bar(['پایه', 'SMOTE', 'SMOTE+Threshold', 'Drift'],
                   [precision_score(y_test, y_pred_base), precision_smote, precision_custom, precision_drift])
    axes[1, 1].set_title('مقایسه Precision')
    axes[1, 1].set_ylabel('Precision')

    plt.tight_layout()
    plt.savefig('smote_drift_analysis.png', dpi=100)
    plt.show()
    print("\n📊 نمودارها در فایل smote_drift_analysis.png ذخیره شدند.")
except:
    print("\n⚠️ برای نمایش نمودارها نیاز به نصب matplotlib و seaborn است.")