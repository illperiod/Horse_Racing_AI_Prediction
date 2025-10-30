# ファイル名: train_lightgbm_rank_threshold_11R12R.py

import pandas as pd
import numpy as np
import lightgbm as lgb

# from sklearn.model_selection import train_test_split # 時系列分割は手動
from sklearn.metrics import ndcg_score
import warnings
import os
import polars as pl
import unicodedata

# --- 設定 ---
CSV_PATH = "data/processed/backtest_predictions_raw_features_pit.csv"
SPLIT_DATE = "2024-12-31"
VALIDATION_SPLIT_RATIO = 0.8

# --- 特徴量リスト (期待値スコア含む) ---
FEATURE_COLS = [
    "予測スコア",
    "単勝オッズ",
    "score_rank",
    "score_diff_from_top",
    "headcount",
    "mean_score",
    "std_score",
    "value_gap",
    "expected_value_score",
]
TARGET_COL = "profit_rank"  # 目的変数 (利益ランク: 0-5)


# --- 前処理関数 (変更なし / 'R'カラム必須) ---
def preprocess_and_feature_engineer(df):  # 省略 (前回のコードと同じ)
    print("データの前処理と特徴量計算を開始...")
    df["予測スコア"] = df["予測スコア"].fillna(0)
    df["単勝オッズ"] = (
        df["単勝オッズ"].replace([np.inf, -np.inf], 0).fillna(1.0).clip(lower=1.0)
    )
    df["単勝オッズ_実"] = df["単勝オッズ_実"].fillna(0)
    df["着順_num"] = (
        pd.to_numeric(
            df["着順"].astype(str).str.normalize("NFKC").str.extract(r"(\d+)")[0],
            errors="coerce",
        )
        .fillna(99)
        .astype(int)
    )
    if "Ｒ" in df.columns:
        df["Ｒ"] = (
            pd.to_numeric(
                df["Ｒ"].astype(str).str.normalize("NFKC").str.extract(r"(\d+)")[0],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )
    else:
        print("警告: 'R'カラムなし")
        df["Ｒ"] = df["race_id"].astype(str).str.split("_").str[-1].astype(int)
    return_ratio = df["単勝オッズ_実"] / 100.0
    condlist = [
        return_ratio <= 0,
        (return_ratio > 0) & (return_ratio <= 1.2),
        (return_ratio > 1.2) & (return_ratio <= 3.0),
        (return_ratio > 3.0) & (return_ratio <= 7.0),
        (return_ratio > 7.0) & (return_ratio <= 15.0),
        return_ratio > 15.0,
    ]
    choicelist = [0, 1, 2, 3, 4, 5]
    df[TARGET_COL] = np.select(condlist, choicelist, default=0).astype(int)
    print(
        f"目的変数 '{TARGET_COL}' 作成 (分布:\n{df[TARGET_COL].value_counts(normalize=True).sort_index()})"
    )
    print("特徴量を計算中...")
    df_pl = pl.from_pandas(df)
    df_pl = df_pl.with_columns(
        pl.col("馬番").n_unique().over("race_id").alias("headcount")
    )
    df_pl = df_pl.with_columns(
        pl.col("予測スコア")
        .rank(method="min", descending=True)
        .over("race_id")
        .alias("score_rank")
    )
    df_pl = df_pl.with_columns(
        pl.col("予測スコア").max().over("race_id").alias("score_max")
    )
    df_pl = df_pl.with_columns(
        (pl.col("予測スコア") - pl.col("score_max")).alias("score_diff_from_top")
    )
    df_pl = df_pl.with_columns(
        pl.col("予測スコア").mean().over("race_id").alias("mean_score")
    )
    df_pl = df_pl.with_columns(
        pl.col("予測スコア").std().over("race_id").fill_null(0).alias("std_score")
    )
    df_pl = df_pl.with_columns(
        pl.col("単勝オッズ")
        .rank(method="min", descending=False)
        .over("race_id")
        .alias("odds_rank")
    )
    df_pl = df_pl.with_columns(
        (pl.col("odds_rank") - pl.col("score_rank")).alias("value_gap")
    )
    df_pl = df_pl.with_columns(
        (pl.col("予測スコア") * pl.col("単勝オッズ")).alias("expected_value_score")
    )
    df = df_pl.to_pandas()
    cols_to_keep = [
        "race_id",
        "日付",
        "馬番",
        "単勝オッズ_実",
        TARGET_COL,
        "Ｒ",
    ] + FEATURE_COLS
    cols_to_keep = list(set(cols_to_keep).intersection(df.columns))
    df = df[cols_to_keep]
    df = df.fillna(0)
    print("前処理と特徴量計算完了。")
    return df


# ▼▼▼ 【★★★ 閾値探索用の評価関数 (11R/12R フィルタリング追加) ★★★】 ▼▼▼
def simulate_return_rate_rank1_threshold_11R12R(df_results, threshold):
    """
    与えられた閾値で、Rank1位かつ11R/12Rのベット回収率を計算
    """
    pred_rank_col = "predict_rank_score"
    bets_df = pd.DataFrame()  # 初期化
    rank1_bets = pd.DataFrame()  # 初期化

    # 1. 各レースで予測スコア1位を特定
    try:
        idx = df_results.groupby("race_id")[pred_rank_col].idxmax(skipna=True)
        rank1_df = df_results.loc[idx]
    except ValueError:
        rank1_df = pd.DataFrame()  # 空にする

    if not rank1_df.empty:
        # 2. ★★★ 11R または 12R の結果のみにフィルタリング ★★★
        if "Ｒ" not in rank1_df.columns:
            print("エラー: 評価データに 'R' カラムなし")
            rank1_df_filtered = pd.DataFrame()
        else:
            rank1_df_filtered = rank1_df[rank1_df["Ｒ"].isin([11, 12])].copy()

        # 3. 予測スコアが閾値以上の馬券のみ選択
        if not rank1_df_filtered.empty:
            bets_df = rank1_df_filtered[
                rank1_df_filtered[pred_rank_col] >= threshold
            ].copy()

    total_bets = len(bets_df)
    if total_bets == 0:
        return 0.0, 0.0, 0, 0, 0  # rate, hit_rate, bets, investment, payout

    total_investment = total_bets * 100
    total_payout = bets_df["単勝オッズ_実"].sum()
    total_wins = (bets_df["単勝オッズ_実"] > 0).sum()

    return_rate = (
        (total_payout / total_investment) * 100 if total_investment > 0 else 0.0
    )
    hit_rate = (total_wins / total_bets) * 100 if total_bets > 0 else 0.0

    return return_rate, hit_rate, total_bets, total_investment, total_payout


# ▲▲▲ 【★★★ 閾値探索用の評価関数 (11R/12R フィルタリング追加) ★★★】 ▲▲▲


# ▼▼▼ 【★★★ 最適閾値探索関数 (11R/12R 対象) ★★★】 ▼▼▼
def find_optimal_threshold_11R12R(df_val_results):
    """
    検証データの11R/12Rのみを対象に、最適な閾値を見つける
    """
    print("\n" + "=" * 50)
    print("--- 最適閾値探索 (Rank1位 & 11R/12R / 検証データ) ---")
    pred_rank_col = "predict_rank_score"

    # (探索する閾値候補 - 変更なし)
    thresholds_to_try = np.quantile(
        df_val_results[pred_rank_col], np.arange(0, 1.05, 0.05)
    )
    thresholds_to_try = np.unique(thresholds_to_try)
    print(
        f"探索閾値候補 ({len(thresholds_to_try)}個): {np.round(thresholds_to_try, 3)}"
    )

    best_threshold = -np.inf
    best_return_rate = -np.inf
    results = []

    for threshold in thresholds_to_try:
        # ★★★ 11R/12R 限定のシミュレーション関数を呼び出す ★★★
        return_rate, hit_rate, num_bets, investment, payout = (
            simulate_return_rate_rank1_threshold_11R12R(df_val_results, threshold)
        )
        results.append(
            {
                "threshold": threshold,
                "return_rate": return_rate,
                "hit_rate": hit_rate,
                "num_bets": num_bets,
            }
        )
        if return_rate > best_return_rate:
            best_return_rate = return_rate
            best_threshold = threshold

    results_df = pd.DataFrame(results).sort_values(by="threshold")
    print("\n閾値ごとの検証結果 (11R/12Rのみ):")
    print(results_df)
    print(
        f"\n---> 最適閾値 (検証@11R/12R 回収率最大): {best_threshold:.4f} (回収率: {best_return_rate:.2f}%)"
    )
    print("=" * 50)
    return best_threshold


# ▲▲▲ 【★★★ 最適閾値探索関数 (11R/12R 対象) ★★★】 ▲▲▲


# ▼▼▼ 【★★★ テスト用評価関数 (11R/12R 限定 + 閾値適用) ★★★】 ▼▼▼
def evaluate_betting_strategy_rank1_threshold_11R12R(
    df_test_results, optimal_threshold
):
    """
    最適閾値を使ってテストデータの11R/12Rでベット戦略をシミュレーション
    """
    print("\n" + "=" * 50)
    print(
        f"--- ベット戦略 (Rank1位 & 11R/12R & 閾値: {optimal_threshold:.4f} / テスト) ---"
    )

    # ★★★ 11R/12R 限定のシミュレーション関数を呼び出す ★★★
    return_rate, hit_rate, total_bets, total_investment, total_payout = (
        simulate_return_rate_rank1_threshold_11R12R(df_test_results, optimal_threshold)
    )

    # (表示用にレース数と的中数を再計算 - simulate関数内で計算済みだが、念のため)
    pred_rank_col = "predict_rank_score"
    bets_df = pd.DataFrame()
    try:
        idx = df_test_results.groupby("race_id")[pred_rank_col].idxmax(skipna=True)
        rank1_df = df_test_results.loc[idx]
        if not rank1_df.empty:
            rank1_df_filtered = rank1_df[rank1_df["Ｒ"].isin([11, 12])].copy()
            if not rank1_df_filtered.empty:
                bets_df = rank1_df_filtered[
                    rank1_df_filtered[pred_rank_col] >= optimal_threshold
                ].copy()
    except Exception as e:
        print(f"警告: 最終評価中のデータ処理エラー: {e}")

    num_races = bets_df["race_id"].nunique() if not bets_df.empty else 0
    total_wins = (bets_df["単勝オッズ_実"] > 0).sum() if not bets_df.empty else 0

    print(f"  購入対象ベット数: {total_bets} 件 (対象レース数: {num_races} レース)")
    print(f"  総投資額: {total_investment:,.0f} 円")
    print(f"  総払戻額: {total_payout:,.0f} 円")
    print(f"  的中ベット数: {total_wins} 件")
    print("-" * 30)
    print(f"  ★ 的中率 (ベット単位): {hit_rate:.2f} % ★")
    print(f"  ★ 単勝回収率: {return_rate:.2f} % ★")
    print("=" * 50)

    return return_rate, hit_rate


# ▲▲▲ 【★★★ テスト用評価関数 (11R/12R 限定 + 閾値適用) ★★★】 ▲▲▲


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    # --- 1. データロード & 前処理 ---
    print(f"データロード: {CSV_PATH}")
    df = None
    try:
        df_full = pd.read_csv(CSV_PATH, parse_dates=["日付"])
        df = preprocess_and_feature_engineer(df_full)
    except FileNotFoundError:
        print(f"エラー: {CSV_PATH}なし")
        exit()
    except Exception as e:
        print(f"エラー: データ処理中 {e}")
        import traceback

        traceback.print_exc()
        exit()
    if df is None:
        print("エラー: データ準備失敗")
        exit()

    # --- 2. 時系列分割 (Train -> Train_sub / Val, Test) ---
    print(f"\nデータ分割: {SPLIT_DATE}")
    split_date = pd.to_datetime(SPLIT_DATE)
    df_train_full = (
        df[df["日付"] <= split_date].copy().sort_values(by=["日付", "race_id", "馬番"])
    )
    df_test = df[df["日付"] > split_date].copy().sort_values(by=["race_id", "馬番"])
    if len(df_train_full) == 0 or len(df_test) == 0:
        print("エラー: データ分割失敗")
        exit()
    train_cutoff_date = df_train_full["日付"].quantile(
        VALIDATION_SPLIT_RATIO, interpolation="nearest"
    )
    print(f"学習データを訓練/検証に分割 (カットオフ日: {train_cutoff_date.date()})")
    df_train_sub = df_train_full[df_train_full["日付"] <= train_cutoff_date].copy()
    df_val = df_train_full[df_train_full["日付"] > train_cutoff_date].copy()
    if len(df_train_sub) == 0 or len(df_val) == 0:
        print("エラー: 訓練/検証データ分割失敗")
        exit()
    train_sub_group = df_train_sub.groupby("race_id").size().tolist()
    val_group = df_val.groupby("race_id").size().tolist()
    test_group = df_test.groupby("race_id").size().tolist()
    print(f"訓練データ件数: {len(df_train_sub)} (レース数: {len(train_sub_group)})")
    print(f"検証データ件数: {len(df_val)} (レース数: {len(val_group)})")
    print(f"テストデータ件数: {len(df_test)} (レース数: {len(test_group)})")
    X_train_sub = df_train_sub[FEATURE_COLS]
    y_train_sub = df_train_sub[TARGET_COL]
    X_val = df_val[FEATURE_COLS]
    y_val = df_val[TARGET_COL]
    X_test = df_test[FEATURE_COLS]
    y_test = df_test[TARGET_COL]  # 早期停止用

    # --- 3. LightGBM ランキングモデル学習 ---
    print("\nLightGBM ランキングモデル学習開始...")
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3],
        "boosting_type": "gbdt",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "seed": 42,
        "n_jobs": -1,
        "verbose": -1,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
    }
    params["label_gain"] = [pow(2, i) - 1 for i in range(6)]
    print(f"  Label Gain: {params['label_gain']}")
    model = lgb.LGBMRanker(**params)
    eval_set = [(X_val, y_val)]  # ★ 検証データで早期停止
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
    model.fit(
        X_train_sub,
        y_train_sub,
        group=train_sub_group,
        eval_set=eval_set,
        eval_group=[val_group],
        eval_metric="ndcg",
        callbacks=callbacks,
    )
    print("学習完了。")

    # --- 4. 最適閾値の探索 (11R/12R 限定) ---
    print("\n--- 検証データで予測スコアを計算 ---")
    val_pred_scores = model.predict(X_val)
    df_val["predict_rank_score"] = val_pred_scores
    # ★★★ 11R/12R 限定で最適閾値を探す ★★★
    optimal_threshold = find_optimal_threshold_11R12R(df_val)
    # ▲▲▲ 11R/12R 限定で最適閾値を探す ▲▲▲

    # --- 5. テストデータでの最終評価 (11R/12R 限定) ---
    print("\n--- テストデータで予測スコアを計算 ---")
    test_pred_scores = model.predict(X_test)
    df_test["predict_rank_score"] = test_pred_scores
    # ★★★ 最適閾値を使って 11R/12R 限定で最終評価 ★★★
    optimal_threshold = 0.04
    evaluate_betting_strategy_rank1_threshold_11R12R(df_test, optimal_threshold)
    # ▲▲▲ 最適閾値を使って 11R/12R 限定で最終評価 ▲▲▲
