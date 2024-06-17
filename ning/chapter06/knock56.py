import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 特徴量データの読み込み
X_test = joblib.load('test.feature.pkl')

# ラベルデータの読み込み
test_data = pd.read_csv('test.txt', sep='\t', header=None)
y_test = test_data[0]

# モデルの読み込み
clf = joblib.load('model.joblib')

# 評価データ上の予測値
y_test_pred = clf.predict(X_test)

# 適合率、再現率、F1スコアの計測と表示
report = classification_report(y_test, y_test_pred, target_names=clf.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print(report_df)

# 適合率、再現率、F1スコアのマイクロ平均とマクロ平均を表示
print("Micro Average:")
print(f"Precision: {report['micro avg']['precision']:.4f}")
print(f"Recall: {report['micro avg']['recall']:.4f}")
print(f"F1 Score: {report['micro avg']['f1-score']:.4f}")

print("\nMacro Average:")
print(f"Precision: {report['macro avg']['precision']:.4f}")
print(f"Recall: {report['macro avg']['recall']:.4f}")
print(f"F1 Score: {report['macro avg']['f1-score']:.4f}")
