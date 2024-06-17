import joblib
import pandas as pd
import numpy as np

# 保存済みのモデルとベクトライザの読み込み
clf = joblib.load('model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# 各カテゴリに対する特徴量の重みを取得
feature_names = vectorizer.get_feature_names_out()
coefs = clf.coef_

# 重みの高い特徴量トップ10と重みの低い特徴量トップ10をカテゴリごとに表示
for i, category in enumerate(clf.classes_):
    print(f"Category: {category}")
    
    # 重みの高い特徴量トップ10
    top10_indices = np.argsort(coefs[i])[-10:]
    top10_features = feature_names[top10_indices]
    top10_weights = coefs[i][top10_indices]
    print("Top 10 positive features:")
    for feature, weight in zip(top10_features, top10_weights):
        print(f"{feature}: {weight:.4f}")
    
    # 重みの低い特徴量トップ10
    bottom10_indices = np.argsort(coefs[i])[:10]
    bottom10_features = feature_names[bottom10_indices]
    bottom10_weights = coefs[i][bottom10_indices]
    print("Top 10 negative features:")
    for feature, weight in zip(bottom10_features, bottom10_weights):
        print(f"{feature}: {weight:.4f}")
    
    print("\n")
