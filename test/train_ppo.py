# ファイル名: train_ppo.py
# (日付デバッグ Print 文を追加)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os
import torch
import warnings
import joblib
import polars as pl
import unicodedata
import datetime  # ★ datetime をインポート
import polars.selectors as cs  # ★ Selector をインポート

# environment.py から KeibaBetEnv をインポート
from environment import KeibaBetEnv  # (environment.py も後で修正が必要)


def normalize_nkfc_pl(text: str | None) -> str | None:
    if isinstance(text, str):
        return unicodedata.normalize("NFKC", text)
    return text


# ▼▼▼ 【★★★ `load_and_preprocess_data` 関数を修正 ★★★】 ▼▼▼
def load_and_preprocess_data(
    rl_table_path, raw_race_path, split_date_str, scaler_save_path
):
    """
    (日付デバッグ Print 文を追加)
    """
    print(f"RLテーブルをロード中: {rl_table_path}")
    try:
        df_rl = pl.read_csv(rl_table_path).with_columns(
            pl.col("日付").str.strptime(pl.Date)
        )
        required_rl_cols = [
            "race_id",
            "日付",
            "馬名",
            "予測スコア",
            "単勝オッズ",
            "単勝オッズ_実",
            "着順",
            "馬番",
        ]
        if not all(col in df_rl.columns for col in required_rl_cols):
            missing = [col for col in required_rl_cols if col not in df_rl.columns]
            print(f"エラー: RLテーブルに必要なカラム {missing} がありません。")
            return None, None, None, None
    except Exception as e:
        print(f"エラー: RLテーブルの読み込みに失敗しました。 {e}")
        return None, None, None, None

    print(f"生レースデータをロード中: {raw_race_path}")
    try:
        raw_cols_needed = [
            "日付S",
            "場所",
            "Ｒ",
            "馬名",
            "騎手",
            "芝・ダ",
            "PCI",
            "4角",
            "上り3F",
            "馬番",
            "着順",
        ]
        df_raw = pl.read_csv(
            raw_race_path,
            encoding="cp932",
            columns=raw_cols_needed,
            schema_overrides={"着順": pl.Utf8},
        )
        df_raw = df_raw.with_columns(
            pl.col("日付S")
            .str.strptime(pl.Date, format="%Y.%m.%d", strict=False)
            .alias("日付")
        ).drop_nulls("日付")
        df_raw = df_raw.with_columns(
            pl.col("Ｒ")
            .cast(pl.Utf8)
            .map_elements(normalize_nkfc_pl, return_dtype=pl.Utf8)
            .str.extract(r"(\d+)", 0)
            .cast(pl.Int64, strict=False)
            .alias("Ｒ")
        )
        df_raw = df_raw.with_columns(
            (
                pl.col("日付S").cast(pl.Utf8)
                + "_"
                + pl.col("場所").cast(pl.Utf8)
                + "_"
                + pl.col("Ｒ").cast(pl.Utf8)
            ).alias("race_id")
        )
        for col in ["馬名", "騎手"]:
            df_raw = df_raw.with_columns(
                pl.col(col)
                .cast(pl.Utf8)
                .map_elements(normalize_nkfc_pl, return_dtype=pl.Utf8)
                .str.strip_chars()
                .str.replace_all(r"\s+", "")
                .str.replace_all(r"[▲△▽☆]", "")
                .alias(col)
            )
        cols_to_num = ["PCI", "4角", "上り3F", "馬番"]
        for col in cols_to_num:
            df_raw = df_raw.with_columns(
                pl.col(col)
                .cast(pl.Utf8)
                .map_elements(normalize_nkfc_pl, return_dtype=pl.Utf8)
                .str.extract(r"(\d+\.?\d*)", 0)
                .cast(pl.Float64, strict=False)
                .alias(col)
            )
        df_raw = df_raw.with_columns(
            pl.col("着順")
            .map_elements(normalize_nkfc_pl, return_dtype=pl.Utf8)
            .str.extract(r"(\d+\.?\d*)", 0)
            .cast(pl.Float64, strict=False)
            .alias("着順")
        )
        if "芝・ダ" not in df_raw.columns:
            print("警告: 生データに '芝・ダ' カラムがありません。")

        print("RLテーブルに生データをマージ中...")
        raw_cols_to_merge = [
            c
            for c in raw_cols_needed
            if c not in ["日付S", "場所"] + list(df_rl.columns)
        ]
        df_merged = df_rl.join(
            df_raw.select(["race_id", "馬名"] + raw_cols_to_merge),
            on=["race_id", "馬名"],
            how="left",
        )
        print(f"マージ後のデータ件数: {len(df_merged)}")
        total_null_count = (
            df_merged.select(raw_cols_to_merge).null_count().sum().sum_horizontal()[0]
        )
        if total_null_count > 0:
            print(f"警告: マージ後に {total_null_count} 件のNullが発生しました。")
        df = df_merged
    except Exception as e:
        print(f"エラー: 生データの読み込みまたはマージに失敗しました。 {e}")
        import traceback

        traceback.print_exc()
        return None, None, None, None

    # --- 基本前処理 ---
    df = df.with_columns(
        pl.col("単勝オッズ").replace([np.inf, -np.inf], 0).fill_null(0)
    )
    df = df.with_columns(
        pl.when(pl.col("単勝オッズ") < 1.0)
        .then(1.0)
        .otherwise(pl.col("単勝オッズ"))
        .alias("単勝オッズ")
    )
    df = df.with_columns(pl.col("単勝オッズ").log().alias("単勝オッズ_log"))
    df = df.with_columns(pl.col("単勝オッズ_実").fill_null(0))
    df = df.with_columns(pl.col("race_id").str.ends_with("_11").alias("is_11R"))
    df = df.with_columns(
        pl.col("着順")
        .cast(pl.Utf8)
        .map_elements(normalize_nkfc_pl)
        .str.extract(r"(\d+)", 0)
        .cast(pl.Float64, strict=False)
        .fill_null(99)
        .cast(pl.Int64)
        .alias("着順_num")
    )
    df = df.with_columns(pl.col("PCI").cast(pl.Float64, strict=False).alias("PCI_num"))
    df = df.with_columns(pl.col("4角").cast(pl.Float64, strict=False).alias("4角_num"))
    df = df.with_columns(
        pl.col("上り3F").cast(pl.Float64, strict=False).alias("上り3F_num")
    )
    df = df.with_columns(pl.col("馬番").n_unique().over("race_id").alias("headcount"))
    df = df.with_columns(
        (pl.col("4角_num") / pl.col("headcount")).alias("4角順位率_num")
    )
    if "Ｒ" not in df.columns:
        print("エラー: マージ後のDFに 'Ｒ' カラムがありません。")
        return None, None, None, None

    # ▼▼▼ 【★★★ デバッグ Print 1: マージ直後の日付カラム ★★★】 ▼▼▼
    print("\n--- [DEBUG] マージ直後の日付カラム情報 ---")
    print(f"  df['日付'] の型: {df['日付'].dtype}")
    # (最初の5つと最後の5つの日付を表示)
    if df.height > 10:
        print(f"  最初の日付5件: {df['日付'].head(5).to_list()}")
        print(f"  最後の日付5件: {df['日付'].tail(5).to_list()}")
    else:
        print(f"  全日付: {df['日付'].to_list()}")
    print("-" * 30)
    # ▲▲▲ 【★★★ デバッグ Print 1 ★★★】 ▲▲▲

    # --- データ分割 (全レース) ---
    split_date = pl.lit(split_date_str).str.strptime(pl.Date)
    df_train_full = df.filter(pl.col("日付") <= split_date)  # 全レース
    df_test_full = df.filter(pl.col("日付") > split_date)  # 全レース
    if df_train_full.height == 0 or df_test_full.height == 0:
        print(f"エラー: 分割失敗")
        return None, None, None, None
    print(f"学習データ(全レース): {df_train_full.height}件")
    print(f"テストデータ(全レース): {df_test_full.height}件")

    # --- 特徴量計算 (全レース) ---
    processed_dfs_full = {}
    for name, df_proc in [("train", df_train_full), ("test", df_test_full)]:
        print(f"\n--- {name} データの静的特徴量を計算中 ---")
        df_proc = df_proc.with_columns(
            pl.col("予測スコア")
            .rank(method="min", descending=True)
            .over("race_id")
            .alias("score_rank")
        )
        df_proc = df_proc.with_columns(
            pl.col("予測スコア").max().over("race_id").alias("score_max")
        )
        df_proc = df_proc.with_columns(
            (pl.col("予測スコア") - pl.col("score_max")).alias("score_diff_from_top")
        )
        df_proc = df_proc.with_columns(
            pl.col("予測スコア").mean().over("race_id").alias("mean_score")
        )
        df_proc = df_proc.with_columns(
            pl.col("予測スコア").std().over("race_id").fill_null(0).alias("std_score")
        )
        df_proc = df_proc.with_columns(
            pl.col("単勝オッズ")
            .rank(method="min", descending=False)
            .over("race_id")
            .alias("odds_rank")
        )
        df_proc = df_proc.with_columns(
            (pl.col("odds_rank") - pl.col("score_rank")).alias("value_gap")
        )
        processed_dfs_full[name] = df_proc
    df_train_featured = processed_dfs_full["train"]
    df_test_featured = processed_dfs_full["test"]

    # --- 動的特徴量計算 (全学習データ / スケーラー学習用) ---
    print("\n--- 全学習データで動的特徴量計算 ---")
    df_train_dynamic = df_train_featured.sort(["日付", "race_id", "馬番"])
    df_train_dynamic = df_train_dynamic.with_columns(
        [
            (pl.col("着順_num") == 1).cast(pl.Int32).alias("_win"),
            (pl.col("着順_num") <= 3).cast(pl.Int32).alias("_place"),
        ]
    )
    df_train_dynamic = df_train_dynamic.with_columns(
        [
            pl.col("_win")
            .cum_sum()
            .over(["日付", "騎手"])
            .shift(1)
            .fill_null(0)
            .alias("_j_wins_cum"),
            pl.col("_place")
            .cum_sum()
            .over(["日付", "騎手"])
            .shift(1)
            .fill_null(0)
            .alias("_j_places_cum"),
            pl.int_range(0, pl.len())
            .over(["日付", "騎手"])
            .shift(1)
            .fill_null(0)
            .alias("_j_rides_cum"),
        ]
    )
    df_train_dynamic = df_train_dynamic.with_columns(
        [
            (pl.col("_j_wins_cum") / (pl.col("_j_rides_cum") + 1e-6)).alias(
                "daily_jockey_win_rate"
            ),
            (pl.col("_j_places_cum") / (pl.col("_j_rides_cum") + 1e-6)).alias(
                "daily_jockey_place_rate"
            ),
        ]
    ).drop(["_win", "_place", "_j_wins_cum", "_j_places_cum", "_j_rides_cum"])
    turf_mask = pl.col("芝・ダ") == "芝"
    dirt_mask = pl.col("芝・ダ") == "ダ"
    winner_mask = pl.col("着順_num") == 1
    df_train_dynamic = df_train_dynamic.with_columns(
        [
            pl.when(turf_mask & winner_mask)
            .then(pl.col("PCI_num"))
            .otherwise(None)
            .mean()
            .over("日付")
            .shift(1)
            .alias("daily_track_winner_avg_pci"),
            pl.when(turf_mask & winner_mask)
            .then(pl.col("上り3F_num"))
            .otherwise(None)
            .mean()
            .over("日付")
            .shift(1)
            .alias("daily_track_winner_avg_last3f"),
            pl.when(dirt_mask & winner_mask)
            .then(pl.col("4角順位率_num"))
            .otherwise(None)
            .mean()
            .over("日付")
            .shift(1)
            .alias("daily_track_winner_avg_4c_pos_rate"),
        ]
    )
    dynamic_cols_to_fill = [
        "daily_jockey_win_rate",
        "daily_jockey_place_rate",
        "daily_track_winner_avg_pci",
        "daily_track_winner_avg_last3f",
        "daily_track_winner_avg_4c_pos_rate",
    ]
    df_train_dynamic = df_train_dynamic.with_columns(
        [pl.col(c).fill_null(0.0) for c in dynamic_cols_to_fill]
    )
    print("動的特徴量計算完了")

    # --- 正規化 (スケーラー学習・保存 / 全特徴量) ---
    print("\nスケーラー学習・保存中...")
    scalers = {}
    static_features_orig = [
        "予測スコア",
        "単勝オッズ_log",
        "score_rank",
        "score_diff_from_top",
        "headcount",
        "mean_score",
        "std_score",
        "value_gap",
    ]
    scaled_static_cols = [
        "score_scaled",
        "odds_scaled",
        "rank_scaled",
        "diff_scaled",
        "headcount_scaled",
        "mean_scaled",
        "std_scaled",
        "gap_scaled",
    ]
    static_mapping = dict(zip(static_features_orig, scaled_static_cols))
    dynamic_features = [
        "daily_jockey_win_rate",
        "daily_jockey_place_rate",
        "daily_track_winner_avg_pci",
        "daily_track_winner_avg_last3f",
        "daily_track_winner_avg_4c_pos_rate",
    ]
    all_features_to_scale = static_features_orig + dynamic_features
    df_train_dynamic_pd = df_train_dynamic.to_pandas()
    for feature in all_features_to_scale:
        scaler = StandardScaler()
        feature_data_np = df_train_dynamic_pd[[feature]].fillna(0).values
        if feature_data_np.ndim == 1:
            feature_data_np = feature_data_np.reshape(-1, 1)
        scalers[feature] = scaler.fit(feature_data_np)
    try:
        joblib.dump(scalers, scaler_save_path)
        print(f"スケーラー保存: {scaler_save_path}")
    except Exception as e:
        print(f"エラー: スケーラー保存 {e}")
        return None, None, None, None

    # --- スケーリング適用 (静的特徴量のみ / 学習・テスト両方) ---
    print("\n静的特徴量スケーリング適用中...")
    train_scaled_exprs = []
    test_scaled_exprs = []
    df_train_featured_pd = df_train_featured.to_pandas()
    df_test_featured_pd = df_test_featured.to_pandas()
    for feature_orig, scaled_col_short in static_mapping.items():
        train_data_np = df_train_featured_pd[[feature_orig]].fillna(0).values
        test_data_np = df_test_featured_pd[[feature_orig]].fillna(0).values
        if train_data_np.ndim == 1:
            train_data_np = train_data_np.reshape(-1, 1)
        if test_data_np.ndim == 1:
            test_data_np = test_data_np.reshape(-1, 1)
        train_scaled_data = scalers[feature_orig].transform(train_data_np)
        test_scaled_data = scalers[feature_orig].transform(test_data_np)
        train_scaled_exprs.append(
            pl.Series(scaled_col_short, train_scaled_data.flatten())
        )
        test_scaled_exprs.append(
            pl.Series(scaled_col_short, test_scaled_data.flatten())
        )
    df_train_featured = df_train_featured.with_columns(train_scaled_exprs)
    df_test_featured = df_test_featured.with_columns(test_scaled_exprs)

    # --- 学習データをフィルタリング ---
    df_train_final_filtered = df_train_featured.filter(pl.col("Ｒ").is_in([11, 12]))
    print(f"\n学習データフィルタリング後件数: {df_train_final_filtered.height}")
    df_test_final = df_test_featured  # テストは全レース

    # ▼▼▼ 【★★★ デバッグ Print 2 & 3 ★★★】 ▼▼▼
    print("\n--- [DEBUG] フィルタリング後 学習データの日付情報 ---")
    print(
        f"  df_train_final_filtered['日付'] の型: {df_train_final_filtered['日付'].dtype}"
    )
    unique_train_dates = df_train_final_filtered["日付"].unique().sort()
    print(f"  ユニークな日付の数: {len(unique_train_dates)}")
    if len(unique_train_dates) > 10:
        print(f"  最初の日付5件: {unique_train_dates.head(5).to_list()}")
        print(f"  最後の日付5件: {unique_train_dates.tail(5).to_list()}")
    else:
        print(f"  全日付: {unique_train_dates.to_list()}")
    print("-" * 30)

    print("\n--- [DEBUG] 特徴量計算後 全学習データの日付情報 (Env用) ---")
    print(f"  df_train_featured['日付'] の型: {df_train_featured['日付'].dtype}")
    unique_full_train_dates = df_train_featured["日付"].unique().sort()
    print(f"  ユニークな日付の数: {len(unique_full_train_dates)}")
    if len(unique_full_train_dates) > 10:
        print(f"  最初の日付5件: {unique_full_train_dates.head(5).to_list()}")
        print(f"  最後の日付5件: {unique_full_train_dates.tail(5).to_list()}")
    else:
        print(f"  全日付: {unique_full_train_dates.to_list()}")
    print("-" * 30)
    # ▲▲▲ 【★★★ デバッグ Print 2 & 3 ★★★】 ▲▲▲

    # --- 不要列削除 ---
    env_needed_cols = [
        "race_id",
        "日付",
        "馬番",
        "馬名",
        "単勝オッズ",
        "騎手",
        "芝・ダ",
        "着順_num",
        "Ｒ",
        "単勝オッズ_実",
        "is_11R",
        "headcount",
        "PCI_num",
        "4角順位率_num",
        "上り3F_num",
    ] + scaled_static_cols
    cols_to_keep_train = [
        col for col in df_train_final_filtered.columns if col in env_needed_cols
    ]
    cols_to_keep_test = [col for col in df_test_final.columns if col in env_needed_cols]
    df_train_final = df_train_final_filtered.select(cols_to_keep_train)
    df_test_final = df_test_final.select(cols_to_keep_test)
    print(f"\n  最終学習データカラム (一部): {df_train_final.columns[:15]}...")
    print(f"  最終テストデータカラム (一部): {df_test_final.columns[:15]}...")

    # --- 最終チェック ---
    if not all(col in df_train_final.columns for col in scaled_static_cols):
        missing = [c for c in scaled_static_cols if c not in df_train_final.columns]
        print(f"警告: 最終学習DF {missing}")
    if not all(col in df_test_final.columns for col in scaled_static_cols):
        missing = [c for c in scaled_static_cols if c not in df_test_final.columns]
        print(f"警告: 最終テストDF {missing}")
    train_nulls = (
        df_train_final.select(scaled_static_cols).null_count().sum_horizontal()[0]
    )
    test_nulls = (
        df_test_final.select(scaled_static_cols).null_count().sum_horizontal()[0]
    )
    if train_nulls > 0 or test_nulls > 0:
        print("警告: 最終DF NaN残存。0埋め。")
        df_train_final = df_train_final.with_columns(
            [pl.col(c).fill_null(0.0) for c in scaled_static_cols]
        )
        df_test_final = df_test_final.with_columns(
            [pl.col(c).fill_null(0.0) for c in scaled_static_cols]
        )

    return df_train_final, df_test_final, scaler_save_path, df_train_featured


# ▲▲▲ 【★★★ `load_and_preprocess_data` 関数ここまで ★★★】 ▲▲▲


def evaluate_model_on_test_env(model, test_env_instance):  # 省略
    print("\n" + "=" * 50)
    print("--- テスト評価開始 ---")
    print("★★★ 11R 強制ベット ★★★")  # 以下省略


if __name__ == "__main__":
    # ... (設定、メイン処理は変更なし) ...
    RL_TABLE_PATH = "data/processed/backtest_predictions_raw_features_pit.csv"
    RAW_RACE_PATH = "2010_2025_data_v2.csv"
    SPLIT_DATE = "2024-12-31"
    SCALER_PATH = "filtered_dynamic_scalers.joblib"
    MODEL_PATH = "ppo_keiba_betting_model_v_race_filtered_dynamic.zip"
    LOG_DIR = "./ppo_keiba_tensorboard_monitor_v_race_filtered_dynamic/"
    os.makedirs(LOG_DIR, exist_ok=True)
    df_train_filtered, df_test, scaler_path, df_train_full_featured = (
        load_and_preprocess_data(RL_TABLE_PATH, RAW_RACE_PATH, SPLIT_DATE, SCALER_PATH)
    )
    if df_train_filtered is None:
        print("データ準備失敗")
        exit()
    N_EPOCHS_APPROX = 50
    TOTAL_TIMESTEPS = df_train_filtered.height * N_EPOCHS_APPROX
    print(f"学習ステップ数: {TOTAL_TIMESTEPS:,}")
    try:  # 省略
        env_instance = KeibaBetEnv(
            df_train_filtered,
            df_train_full_featured,
            scaler_path=scaler_path,
            is_training=True,
        )
        train_env_monitored = Monitor(env_instance, LOG_DIR)
        train_env = DummyVecEnv([lambda: train_env_monitored])
        test_env = KeibaBetEnv(
            df_test, df_train_full_featured, scaler_path=scaler_path, is_training=False
        )
    except Exception as e:
        print(f"環境作成エラー: {e}")
        exit()
    device = "cpu"
    print(f"--- デバイス: {device} ---")
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device=device,
        n_steps=1024,
        ent_coef=0.005,
    )
    print(f"\n--- 学習開始 (Steps = {TOTAL_TIMESTEPS:,}) ---")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tb_log_name="PPO_Keiba_Race_Run_FilteredDynamic",
        progress_bar=True,
    )
    print("--- 学習完了 ---")
    model.save(MODEL_PATH)
    print(f"モデル保存: {MODEL_PATH}")
    evaluate_model_on_test_env(model, test_env)
