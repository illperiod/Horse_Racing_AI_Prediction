# これは何？
　機械学習で競馬の予想を行うプログラムです。flaskを用いたwebアプリケーション上で動作します。

　大きく分けて、各馬のスコア付けと、そこから導き出される馬券購入戦略の二つに対して機械学習が行われます。
前者はDCNを組み合わせたLSTMモデルで行われ、後者はLightGBMを用いたモデルで行われます。

　生データは配布しておりません。生データの用意はJRA-VAN データラボのTARGET frontier JVを想定しています。

# 使い方

## 予測モデルを作成する
1. `docker`でコンテナを作成
2. `TARGET frontier JV`から血統データとレース結果の生データを生成
3. `features_engineered.py`で特徴量エンジニアリング
4. `keiba_analysis.py`でモデル情報を作成
5. `preprocess_sequences.py`で時系列データを定義
6. 再度`keiba_analysis.py`を実行しモデルを作成

## 評価する
1. `/test/create_rl_table.py`を実行（スコアの算出には時間がかかる場合があります）
2. `merge_features_score.py`を実行
3. 各評価プログラムを実行

## 予測する
1. `app.py`を実行
2. `http://127.0.0.1:5000` をブラウザで開く
3. 予測したい `netkeiba` の `URL` を `netkeiba URL` に貼る
4. 予測を実行を入力
5. しばし待つ
6. 予測結果が出る

随時更新予定……
