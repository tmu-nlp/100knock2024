from knock51 import preprocess_text, load_data, extract_features
from knock52 import train_model

def predict_category(model, vectorizer, headline):
    # 記事見出しの前処理
    preprocessed_headline = preprocess_text(headline)
    
    # 特徴量の抽出
    headline_features = vectorizer.transform([preprocessed_headline])
    
    # カテゴリの予測
    predicted_category = model.predict(headline_features)[0]
    
    # 予測確率の計算
    predicted_proba = model.predict_proba(headline_features)[0]
    
    return predicted_category, predicted_proba

def main():
    # データの読み込み
    train_df = load_data("train.txt")
    
    # 特徴量の抽出
    train_features, vectorizer = extract_features(train_df)
    
    # モデルの学習
    model = train_model(train_features, train_df["CATEGORY"])
    
    # 記事見出しの入力
    headline = input("記事見出しを入力してください: ")
    
    # カテゴリと予測確率の計算
    predicted_category, predicted_proba = predict_category(model, vectorizer, headline)
    
    # 結果の表示
    print(f"予測されたカテゴリ: {predicted_category}")
    print(f"予測確率: {predicted_proba}")

if __name__ == "__main__":
    main()