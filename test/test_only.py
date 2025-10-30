# ファイル名: test_only.py
# (7次元状態 + 終了フラグ対応)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import torch
import warnings

# environment.py から KeibaBetEnv をインポート
from environment import KeibaBetEnv


def load_and_preprocess_data(csv_path, split_date_str):
    """
    (train_ppo.py からコピー)
    CSVを読み込み、7次元の状態 + 11Rフラグ を準備する
    """
    print(f"データをロード中: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["日付"])
    required_cols = [
        "race_id",
        "日付",
        "予測スコア",
        "単勝オッズ",
        "単勝オッズ_実",
        "馬番",
    ]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"エラー: 必要なカラム {missing} がCSVにありません。")
        return None, None
    df["予測スコア"] = df["予測スコア"].fillna(0)
    df["単勝オッズ"] = df["単勝オッズ"].replace([np.inf, -np.inf], 0).fillna(0)
    df["単勝オッズ"] = df["単勝オッズ"].clip(lower=1.0)
    df["単勝オッズ_log"] = np.log(df["単勝オッズ"])
    df["単勝オッズ_実"] = df["単勝オッズ_実"].fillna(0)
    df["is_11R"] = df["race_id"].astype(str).str.endswith("_11")
    print("11Rフラグ (is_11R) カラムを追加しました。")
    split_date = pd.to_datetime(split_date_str)
    df_train = df[df["日付"] <= split_date].copy()
    df_test = df[df["日付"] > split_date].copy()
    if len(df_train) == 0 or len(df_test) == 0:
        print(f"エラー: {split_date_str} でデータを分割できませんでした。")
        return None, None
    print(f"学習データ: {len(df_train)}件 (スケーラー学習用)")
    print(f"テストデータ: {len(df_test)}件 (評価対象)")

    # --- レース内特徴量 ---
    print("レース内の相対特徴量（順位、1位との差）を計算中...")
    df_train["score_rank"] = df_train.groupby("race_id")["予測スコア"].rank(
        method="min", ascending=False
    )
    df_test["score_rank"] = df_test.groupby("race_id")["予測スコア"].rank(
        method="min", ascending=False
    )
    df_train["score_max"] = df_train.groupby("race_id")["予測スコア"].transform("max")
    df_test["score_max"] = df_test.groupby("race_id")["予測スコア"].transform("max")
    df_train["score_diff_from_top"] = df_train["予測スコア"] - df_train["score_max"]
    df_test["score_diff_from_top"] = df_test["予測スコア"] - df_test["score_max"]
    print("状態 5-7 (頭数, スコア平均, スコア標準偏差) を計算中...")
    df_train["headcount"] = df_train.groupby("race_id")["馬番"].transform("count")
    df_test["headcount"] = df_test.groupby("race_id")["馬番"].transform("count")
    df_train["mean_score"] = df_train.groupby("race_id")["予測スコア"].transform("mean")
    df_test["mean_score"] = df_test.groupby("race_id")["予測スコア"].transform("mean")
    df_train["std_score"] = (
        df_train.groupby("race_id")["予測スコア"].transform("std").fillna(0)
    )
    df_test["std_score"] = (
        df_test.groupby("race_id")["予測スコア"].transform("std").fillna(0)
    )

    # --- 正規化 ---
    print("7つの状態特徴量を正規化中...")
    scaler_score = StandardScaler()
    scaler_odds = StandardScaler()
    scaler_rank = StandardScaler()
    scaler_diff = StandardScaler()
    scaler_headcount = StandardScaler()
    scaler_mean = StandardScaler()
    scaler_std = StandardScaler()

    # ★ 学習データ (df_train) だけで .fit() ★
    scaler_score.fit(df_train[["予測スコア"]])
    scaler_odds.fit(df_train[["単勝オッズ_log"]])
    scaler_rank.fit(df_train[["score_rank"]])
    scaler_diff.fit(df_train[["score_diff_from_top"]])
    scaler_headcount.fit(df_train[["headcount"]])
    scaler_mean.fit(df_train[["mean_score"]])
    scaler_std.fit(df_train[["std_score"]])

    # ★ テストデータ (df_test) を .transform() ★
    df_test["score_scaled"] = scaler_score.transform(df_test[["予測スコア"]])
    df_test["odds_scaled"] = scaler_odds.transform(df_test[["単勝オッズ_log"]])
    df_test["rank_scaled"] = scaler_rank.transform(df_test[["score_rank"]])
    df_test["diff_scaled"] = scaler_diff.transform(df_test[["score_diff_from_top"]])
    df_test["headcount_scaled"] = scaler_headcount.transform(df_test[["headcount"]])
    df_test["mean_scaled"] = scaler_mean.transform(df_test[["mean_score"]])
    df_test["std_scaled"] = scaler_std.transform(df_test[["std_score"]])

    # (中間列の削除は test_only では不要だが、念のため)
    cols_to_drop = ["score_max", "headcount", "mean_score", "std_score"]
    df_test = df_test.drop(
        columns=[col for col in cols_to_drop if col in df_test.columns]
    )

    # (df_train はスケーラー学習にしか使わないので、df_test のみ返す)
    return df_test


def evaluate_model_on_test_env(model, test_env_instance):
    """
    (train_ppo.py からコピー)
    (終了フラグを監視するように修正)
    """
    print("\n" + "=" * 50)
    print("--- テスト環境での評価開始 ---")
    print("★★★ 11R (race_id が '_11' で終了) は強制的にベットします ★★★")

    test_env_vec = DummyVecEnv([lambda: test_env_instance])

    total_investment_all = 0
    total_payout_all = 0
    total_bets_all = 0
    total_races_processed = 0

    raw_env = test_env_vec.envs[0]

    try:
        obs = test_env_vec.reset()
    except Exception as e:
        print(f"エラー: テスト環境の初回リセットに失敗: {e}")
        return 0

    while True:
        if raw_env.all_test_races_done:
            print("\n--- 全テストレースの処理が完了 (フラグ検知) ---")
            break

        action, _states = model.predict(obs, deterministic=True)

        try:
            current_row = raw_env.current_race_df.loc[raw_env.current_horse_index]
            if current_row["is_11R"]:
                action[0] = 1  # 強制ベット
        except Exception:
            pass

        obs, reward, done, infos = test_env_vec.step(action)

        if done[0]:
            info = infos[0]

            if not raw_env.all_test_races_done:
                total_investment_all += info.get("total_investment_race", 0)
                total_payout_all += info.get("total_payout_race", 0)
                total_bets_all += info.get("total_bets_race", 0)
                total_races_processed += 1

    # --- 最終結果の表示 ---
    print("--- テスト環境での評価終了 ---")
    print(f"  処理した総レース数: {total_races_processed} レース")
    print(f"  総投資額: {total_investment_all:,.0f} 円")
    print(f"  総払戻額: {total_payout_all:,.0f} 円")

    # ▼▼▼ 【★★★ エラー修正 (L165相当) ★★★】 ▼▼▼
    print(f"  総ベット回数: {total_bets_all:,.0f} 回")
    # ▲▲▲ 【★★★ エラー修正 (L165相当) ★★★】 ▲▲▲

    print("-" * 30)
    if total_investment_all > 0:
        final_return_rate = (total_payout_all / total_investment_all) * 100
    else:
        final_return_rate = 0.0
    print(f"  ★ 最終単勝回収率: {final_return_rate:.2f} % ★")
    print("=" * 50)
    return final_return_rate


if __name__ == "__main__":

    # --- 1. 設定 (train_ppo.py と合わせる) ---
    CSV_PATH = "./data/processed/backtest_predictions_raw_features_pit.csv"
    SPLIT_DATE = "2024-12-31"

    # ★ 読み込むモデル (7d)
    MODEL_PATH = "ppo_keiba_betting_model_v_race_7d.zip"

    # --- 2. データ準備 (テストデータのみ) ---
    # (スケーラー学習のため、内部で df_train もロードされる)
    print("--- テストデータ準備中 ---")
    try:
        df_test = load_and_preprocess_data(CSV_PATH, SPLIT_DATE)
    except Exception as e:
        print(f"データ準備中にエラー: {e}")
        df_test = None

    if df_test is None:
        print("テストデータの準備に失敗したため、処理を終了します。")
        exit()

    # ★ is_training=False でテスト環境を作成
    test_env = KeibaBetEnv(df_test, is_training=False)

    # --- 3. 学習済みモデルの読み込み ---
    if not os.path.exists(MODEL_PATH):
        print(f"エラー: 学習済みモデル {MODEL_PATH} が見つかりません。")
        exit()

    print(f"\n--- 学習済みモデル {MODEL_PATH} を読み込み中... ---")
    try:
        model = PPO.load(MODEL_PATH)
        print("モデルの読み込み完了。")
    except Exception as e:
        print(f"モデルの読み込みに失敗しました: {e}")
        exit()

    # --- 4. 評価の実行 ---
    evaluate_model_on_test_env(model, test_env)
