# 各プログラムの説明

## 予測を行う部分

- app.py
  - メインプログラム 予測を行う
- predict_logic.py
  - 予測プログラムの予測部分
- features_engineed.py
  - 特徴量生成プログラム 予測に必要な特徴量を生成
- scraper.py
  - netkeibaからスクレイピングを行い、出馬表を出力

## 予測前に実行しておく部分
- features_engineed.py
  - 生データの特徴量からモデル生成に必要な特徴量を生成（特徴量エンジニアリング）
- keiba_analysis.py
  - モデル生成に使用 DCNとLSTMを組み合わせたモデルを使用
- preprocess_sequences.py
  - モデル生成に使用 時系列データの前処理

## 用意しておく必要のある部分
以下のファイルを`app.py`などと同階層に配置する必要がある
- 2010_2025_data.csv
  - 2010年から2025年のレース結果を表示する
   
    TARGET frontier JVにおいて、レース検索→コース芝ダート→1~18着→期間選択からレース結果を表示し、★項目設定から以下のように項目済み項目を設定する

    ![レース結果](https://github.com/illperiod/Horse_Racing_AI_Prediction/blob/main/screenshot/TARGET_data.png)

    出力タブから★画面イメージ(CSV形式)を選択し、指定階層に`2010_2025_data.csv`という名前で保存する

- 2005_2025_Pedigree.csv
  - 2005年から2025年の馬の血統データを表示する

    TARGET frontier JVにおいて、馬データ検索→年齢2~30歳から血統データを表示し、★項目設定から以下のように項目済み項目を設定する
    
    ![血統データ](https://github.com/illperiod/Horse_Racing_AI_Prediction/blob/main/screenshot/TARGET_pedigree.png)

    出力タブから★画面イメージ(CSV形式)を選択し、指定階層に`2005_2025_pedigree.csv`という名前で保存する
  
## モデルの性能評価を行う部分
`test`フォルダの中にあるプログラムがモデルの評価を行う
- create_rl_table.py
  - 全レースについて動的な特徴量生成を行い、予測スコアを計算する
- merge_features_score.py
  - 予測スコアを表示したものに、血統データとレース結果をマージする
- evaluate.py
  - モデル自体の順位正解率評価
- train_lightgbm_combined.py
  - ケリー基準とランキング学習を用いて最適な馬券戦略を探索
- train_ppo.py
  - 強化学習を用いて最適な馬券戦略を探索 `environment.py`や`test_only.py`もこれに付随