import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# データの読み込み
train_df = pd.read_csv('train.feature.txt', sep='\t')
valid_df = pd.read_csv('valid.feature.txt', sep='\t')

# 特徴量とラベルの分割
X_train = train_df.drop(columns=['Category', 'Title'])
y_train = train_df['Category']

X_valid = valid_df.drop(columns=['Category', 'Title'])
y_valid = valid_df['Category']

# 学習済みモデルの読み込み
model = joblib.load('logistic_regression_model.pkl')

# 学習データおよび評価データ上での予測
y_train_pred = model.predict(X_train)
y_valid_pred = model.predict(X_valid)

# 混同行列の作成
labels = y_train.unique()
conf_matrix_train = confusion_matrix(y_train, y_train_pred, labels=labels)
conf_matrix_valid = confusion_matrix(y_valid, y_valid_pred, labels=labels)

# 混同行列をデータフレームに変換
conf_matrix_train_df = pd.DataFrame(data=conf_matrix_train, index=labels, columns=labels)
conf_matrix_valid_df = pd.DataFrame(data=conf_matrix_valid, index=labels, columns=labels)

# 混同行列の表示関数
def plot_confusion_matrix(conf_matrix, title, filename):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, square=True, cbar=True, annot=True, cmap='Blues', fmt='d')
    plt.xlabel("Predict")
    plt.ylabel("True")
    plt.title(title)
    plt.savefig(filename)
    plt.close()

# 学習データ上の混同行列の表示
plot_confusion_matrix(conf_matrix_train_df, 'Confusion Matrix - Training Data', 'Train_confusion.png')

# 評価データ上の混同行列の表示
plot_confusion_matrix(conf_matrix_valid_df, 'Confusion Matrix - Validation Data', 'Validation_confusion.png')
