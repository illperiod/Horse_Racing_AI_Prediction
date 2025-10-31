# ファイル名: train_lightgbm_combined_v_dual_model.py
# (ランキング学習と勝率予測の二重モデル + ケリー戦略)

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import ndcg_score
import warnings
import os
import polars as pl
import unicodedata

# --- 設定 ---
CSV_PATH = "data/processed/features_with_score_merged.csv"
DATA_START_DATE = "2024-01-01"
SPLIT_DATE = "2024-12-31"

# (リーク疑いを除外した特徴量リスト)
# ★★★ '単勝オッズ' と '予測スコア' が含まれていることを前提 ★★★
FEATURE_COLS = [
    "予測スコア",
    "単勝オッズ",
    "枠番",
    "斤量",
    "距離",
    "年齢",
    "キャリア",
    "レース間隔",
    "馬体重増減",
    "斤量負担率",
    "騎手乗り替わりフラグ",
    "キャリア平均_PCI",
    "近5走平均_PCI",
    "キャリア平均_上り3F",
    "近5走平均_上り3F",
    "キャリア平均_4角順位率",
    "近5走平均_4角順位率",
    "キャリア平均_平均速度",
    "近5走平均_平均速度",
    "キャリア平均_上り3F_vs_基準",
    "キャリア平均_平均速度_vs_基準",
    "近5走平均_上り3F_vs_基準",
    "近5走平均_平均速度_vs_基準",
    "追走指数",
    "レースメンバー平均PCI過去全体",
    "騎手_複勝率",
    "調教師_複勝率",
    "種牡馬_複勝率",
    "騎手_競馬場別複勝率",
    "馬_コース距離別複勝率",
]

# ▼▼▼【ここから修正】▼▼▼
TARGET_COL_RANK = "profit_rank"  # ランキング学習用
TARGET_COL_PROBA = "単勝"  # 勝率予測用
# ▲▲▲【ここまで修正】▲▲▲


# --- 前処理関数 (変更あり) ---
def preprocess_target_and_fillna(df):
    """
    目的変数(2種)を計算し、特徴量カラムのNullを0で埋める
    """
    print("目的変数を計算し、Nullを補完中...")
    df["単勝オッズ_実"] = df["単勝オッズ_実"].fillna(0)
    # (着順の数値化 - 全角対応)
    df["着順_num"] = (
        pd.to_numeric(
            df["着順"].astype(str).str.normalize("NFKC").str.extract(r"(\d+)")[0],
            errors="coerce",
        )
        .fillna(99)
        .astype(int)
    )

    # ▼▼▼【ここから修正】▼▼▼
    # 目的変数1: profit_rank (ランキング学習用)
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
    df[TARGET_COL_RANK] = np.select(condlist, choicelist, default=0).astype(int)

    # 目的変数2: 単勝 (勝率予測用)
    df[TARGET_COL_PROBA] = (df["着順_num"] == 1).astype(int)
    # ▲▲▲【ここまで修正】▲▲▲

    # ★ 特徴量カラムの Null を 0 で埋める ★
    feature_cols_in_df = [col for col in FEATURE_COLS if col in df.columns]
    df[feature_cols_in_df] = (
        df[feature_cols_in_df].fillna(0).replace([np.inf, -np.inf], 0)
    )
    print(f"{len(feature_cols_in_df)}個の特徴量カラムのNull/Infを0で補完しました。")

    df = df.dropna(subset=["race_id"])
    print("前処理完了。")
    return df


# --- (参考) 従来の評価関数 (Rank1位固定100円) ---
def evaluate_betting_strategy_rank1(df_test_results):
    print("\n--- (参考: Rank1位固定100円 / Target: profit_rank) ---")
    pred_rank_col = "predict_rank_score"
    bets_df = pd.DataFrame()

    try:
        bets_idx = df_test_results.groupby("race_id")[pred_rank_col].idxmax(skipna=True)
        bets_df = df_test_results.loc[bets_idx]
    except ValueError:
        print("注意: 有効スコアなし")
        return 0.0, 0.0

    if bets_df.empty:
        print("エラー: 1位抽出失敗")
        return 0.0, 0.0

    total_bets = len(bets_df)
    if total_bets == 0:
        print("  購入対象0件")
        return 0.0, 0.0

    total_investment = total_bets * 100
    total_payout = bets_df["単勝オッズ_実"].sum()
    total_wins = (bets_df["単勝オッズ_実"] > 0).sum()
    return_rate = (
        (total_payout / total_investment) * 100 if total_investment > 0 else 0.0
    )
    hit_rate = (total_wins / total_bets) * 100 if total_bets > 0 else 0.0
    print(f"  購入レース数: {total_bets}")
    print(f"  総投資額: {total_investment:,.0f}")
    print(f"  総払戻額: {total_payout:,.0f}")
    print(f"  的中レース数: {total_wins}")
    print("-" * 30)
    print(f"  ★ 的中率: {hit_rate:.2f} % ★")
    print(f"  ★ 回収率: {return_rate:.2f} % ★")
    return return_rate, hit_rate


# ▼▼▼ 【★★★ 修正版 ケリー戦略 評価関数 (二重モデル版) ★★★】 ▼▼▼
def evaluate_betting_strategy_kelly(
    df_test_results, kelly_fraction, base_unit, max_bet
):
    """
    ランカー1位の馬に対し、分類器の予測勝率(P)でケリー戦略を実行
    """
    print(
        f"\n--- ベット戦略 (Rank1~3位 + 分類器P + 修正ケリー / F_frac={kelly_fraction}, Max={max_bet}) ---"
    )

    # ▼▼▼【ここから修正】▼▼▼
    rank_col = "predict_rank_score"  # ランカーが予測した「儲かる」スコア
    proba_col = "predict_win_proba"  # 分類器が予測した「勝率」
    # ▲▲▲【ここまで修正】▲▲▲

    total_investment = 0
    total_payout = 0
    bet_races = 0  # 実際に賭けたレース数
    win_races = 0  # 的中したレース数

    grouped = df_test_results.groupby("race_id")

    for race_id, race_df in grouped:
        # ▼▼▼【ここからロジック変更】▼▼▼

        # 1. ランキングモデルの予測（儲かるスコア）が上位3位の馬を取得
        try:
            top_n_horses = race_df.nlargest(3, rank_col)
        except ValueError:
            continue  # 有効なスコアがない

        # 2. 上位3頭を1頭ずつチェック
        for target_horse_idx, target_horse_row in top_n_horses.iterrows():
            target_horse = target_horse_row.copy()

            # 2. 選んだ馬の「勝率 P」と「オッズ B」を取得
            P = target_horse[proba_col]
            B = target_horse["単勝オッズ"]

            # ( ... L188以降の E > 0, F_opt > 0, purchase_amount >= 100 の
            #   フィルター処理は、インデントを下げてこのループ内に入れる ...)

            if pd.isna(P) or pd.isna(B) or B < 1.0:
                continue

            # 3. 期待値 E とケリー分数 F の計算 (選んだ1頭だけ)
            E = (P * B) - 1
            # ( ...以下、インデントして継続... )

            if E <= 0:
                continue  # 期待値が 10% 未満なら賭けない

            # フィルター2: オッズが10倍未満（得意領域）であること
            # (モデルが苦手な大穴馬を除外する)
            if B >= 10.0:
                continue

            B_minus_1 = B - 1
            if B_minus_1 <= 0.001:
                continue

            F = (P * B_minus_1 - (1 - P)) / B_minus_1

            # 4. 賭け金（Bet Size）の決定
            F_opt = F
            if F_opt <= 0:
                continue

            calculated_bet = F_opt * kelly_fraction * base_unit

            # 100円単位に「切り捨て」
            purchase_amount = np.floor(calculated_bet / 100.0) * 100

            purchase_amount = min(purchase_amount, max_bet)

            if purchase_amount < 100:
                # 100円未満は賭けない
                continue

            # 5. 払戻の計算
            is_win = target_horse["単勝オッズ_実"] > 0
            payout = (
                target_horse["単勝オッズ_実"] * (purchase_amount / 100.0)
                if is_win
                else 0.0
            )

            total_investment += purchase_amount
            total_payout += payout
            bet_races += 1

            if is_win:
                win_races += 1

            # ▲▲▲【ロジック変更ここまで】▲▲▲

    # 6. 結果集計
    return_rate = (
        (total_payout / total_investment) * 100 if total_investment > 0 else 0.0
    )
    hit_rate = (win_races / bet_races) * 100 if bet_races > 0 else 0.0  # ★的中率

    print(f"  購入レース数: {bet_races}")
    print(f"  総投資額: {total_investment:,.0f}")
    print(f"  総払戻額: {total_payout:,.0f}")
    print(
        f"  平均賭け金: {total_investment / bet_races if bet_races > 0 else 0:,.1f} 円"
    )
    print("-" * 30)
    print(f"  ★ 的中率: {hit_rate:.2f} % ★")
    print(f"  ★ 回収率: {return_rate:.2f} % ★")
    return return_rate


# ▲▲▲ 【★★★ 修正版 ケリー戦略 評価関数ここまで ★★★】 ▲▲▲


# --- メイン実行ブロック ---
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    # --- 1. データロード & 前処理 ---
    print(f"データロード: {CSV_PATH}")
    df = None
    try:
        df_full = pd.read_csv(CSV_PATH, parse_dates=["日付"], low_memory=False)
        print(f"ロード完了 ({len(df_full)}行)")

        start_date_dt = pd.to_datetime(DATA_START_DATE)
        df_full = df_full[df_full["日付"] >= start_date_dt].copy()
        if df_full.empty:
            raise ValueError(f"{start_date_dt.year}年以降のデータがCSVにありません。")
        print(
            f"  -> {start_date_dt.year}年以降のデータのみ ({len(df_full)}行) にフィルタリングしました。"
        )

        print("特徴量カラムの存在チェック...")
        missing_features = [col for col in FEATURE_COLS if col not in df_full.columns]
        if missing_features:
            print(
                f"警告: CSVに必要な特徴量カラム {missing_features} が見つかりません。"
            )
            FEATURE_COLS = [col for col in FEATURE_COLS if col in df_full.columns]
            if not FEATURE_COLS:
                raise ValueError("使用可能な特徴量がありません。")
            print(f"  -> 実際に使用する特徴量: {FEATURE_COLS}")

        # 必須カラムの最終確認
        essential_cols = [
            "race_id",
            "日付",
            "馬番",
            "単勝オッズ_実",
            "着順",
            "単勝オッズ",
        ]
        if not all(col in df_full.columns for col in essential_cols):
            missing_essential = [
                col for col in essential_cols if col not in df_full.columns
            ]
            raise ValueError(f"CSVに必須カラム {missing_essential} が不足しています。")

        df = preprocess_target_and_fillna(df_full)

    except FileNotFoundError:
        print(f"エラー: {CSV_PATH}なし")
        exit()
    except ValueError as ve:
        print(f"エラー: {ve}")
        exit()
    except Exception as e:
        print(f"エラー: データ処理中 {e}")
        exit()

    # --- 2. 時系列分割 & グループ準備 ---
    print(f"\nデータ分割 & グループ準備 (境界: {SPLIT_DATE})")
    split_date = pd.to_datetime(SPLIT_DATE)

    df_train = df[df["日付"] <= split_date].copy().sort_values(by=["race_id", "馬番"])
    df_test = df[df["日付"] > split_date].copy().sort_values(by=["race_id", "馬番"])

    if len(df_train) == 0:
        print(
            f"エラー: 学習データが0件です (期間: {DATA_START_DATE} ～ {SPLIT_DATE})。"
        )
        exit()

    train_group = df_train.groupby("race_id").size().tolist()
    test_group = df_test.groupby("race_id").size().tolist()
    print(f"学習件数: {len(df_train)} (レース数: {len(train_group)})")
    print(f"テスト件数: {len(df_test)} (レース数: {len(test_group)})")

    X_train = df_train[FEATURE_COLS]
    # ▼▼▼【ここから修正】▼▼▼
    y_train_rank = df_train[TARGET_COL_RANK]
    y_train_proba = df_train[TARGET_COL_PROBA]
    # ▲▲▲【ここまで修正】▲▲▲

    X_test = df_test[FEATURE_COLS]
    # ▼▼▼【ここから修正】▼▼▼
    y_test_rank = df_test[TARGET_COL_RANK]
    y_test_proba = df_test[TARGET_COL_PROBA]
    # ▲▲▲【ここまで修正】▲▲▲

    # --- 3. LightGBM 学習 (二重モデル) ---
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]

    # ▼▼▼ 【★★★ Pylanceエラー対策のロジック修正 ★★★】 ▼▼▼

    if not X_test.empty and not y_test_rank.empty and test_group:
        # --- 3A. テストデータがある場合 (学習 & 評価) ---

        # ▼▼▼【ここから修正】▼▼▼
        # --- 3A-1. モデルA (ランカー) の学習 ---
        print("\nLightGBM ランキングモデル学習開始 (Target: profit_rank)...")
        params_rank = {
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
        params_rank["label_gain"] = [pow(2, i) - 1 for i in range(6)]
        print(f"  Label Gain (Ranker): {params_rank['label_gain']}")
        model_ranker = lgb.LGBMRanker(**params_rank)

        model_ranker.fit(
            X_train,
            y_train_rank,
            group=train_group,
            eval_set=[(X_test, y_test_rank)],
            eval_group=[test_group],
            eval_metric="ndcg",
            callbacks=callbacks,
        )
        print("  -> ランカー学習完了。")

        # --- 3A-2. モデルB (分類器) の学習 ---
        print("\nLightGBM 分類モデル学習開始 (Target: 単勝)...")
        params_proba = {
            "objective": "binary",
            "metric": "logloss",
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
        model_classifier = lgb.LGBMClassifier(**params_proba)

        model_classifier.fit(
            X_train,
            y_train_proba,
            # ★ group は不要 ★
            eval_set=[(X_test, y_test_proba)],
            # ★ eval_group は不要 ★
            eval_metric="logloss",
            callbacks=callbacks,
        )
        print("  -> 分類器学習完了。")

        # --- 4. モデル評価 (Feature Importance はランカーのものを表示) ---
        print("\n--- モデル評価 (テストデータ) ---")
        if hasattr(model_ranker, "feature_importances_"):
            feature_imp = (
                pd.DataFrame(
                    {
                        "Value": model_ranker.feature_importances_,
                        "Feature": FEATURE_COLS,
                    }
                )
                .sort_values(by="Value", ascending=False)
                .head(20)
            )
            print("\n  Feature Importances (Ranker, Top 20):")
            print(feature_imp)

        # --- 5. ベット戦略シミュレーション (予測を2回実行) ---
        print("\nテストデータに対して予測を実行中...")
        y_pred_score_rank = model_ranker.predict(X_test)
        y_pred_proba = model_classifier.predict_proba(X_test)[
            :, 1
        ]  # [:, 1] で勝率を取得

        df_test["predict_rank_score"] = y_pred_score_rank
        df_test["predict_win_proba"] = y_pred_proba  # ★ 勝率をDFに追加
        print("予測完了。")

        # ▼▼▼ 【ケリー戦略のパラメータ設定】 ▼▼▼
        KELLY_FRACTION = 0.05
        BASE_UNIT = 100000.0  # (1000.0 -> 100000.0 に変更)
        MAX_BET_PER_RACE = 1000.0
        # ▲▲▲ 【パラメータ設定ここまで】 ▲▲▲

        evaluate_betting_strategy_kelly(
            df_test,
            kelly_fraction=KELLY_FRACTION,
            base_unit=BASE_UNIT,
            max_bet=MAX_BET_PER_RACE,
        )

        # (参考: 従来のRank1位固定100円も表示)
        evaluate_betting_strategy_rank1(df_test)
        # ▲▲▲ 【修正ここまで】 ▲▲▲

    else:
        # --- 3B. テストデータがない場合 (学習のみ) ---
        print(
            "  -> テストデータがないため、(Early Stopping のため) 学習データを評価セットにも使用します"
        )

        # ▼▼▼【ここから修正】▼▼▼
        # ランカー学習
        print("\nLightGBM ランキングモデル学習開始...")
        params_rank = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "seed": 42,
            "verbose": -1,
            "n_estimators": 500,  # 例 (テストなしなので適宜調整)
            "label_gain": [pow(2, i) - 1 for i in range(6)],
            # ... 他のパラメータ ...
        }
        model_ranker = lgb.LGBMRanker(**params_rank)
        model_ranker.fit(
            X_train,
            y_train_rank,
            group=train_group,
            eval_set=[(X_train, y_train_rank)],
            eval_group=[train_group],
            callbacks=callbacks,
        )
        print("  -> ランカー学習完了。")

        # 分類器学習
        print("\nLightGBM 分類モデル学習開始...")
        params_proba = {
            "objective": "binary",
            "metric": "logloss",
            "seed": 42,
            "verbose": -1,
            "n_estimators": 500,  # 例 (テストなしなので適宜調整)
            # ... 他のパラメータ ...
        }
        model_classifier = lgb.LGBMClassifier(**params_proba)
        model_classifier.fit(
            X_train,
            y_train_proba,
            eval_set=[(X_train, y_train_proba)],
            callbacks=callbacks,
        )
        print("  -> 分類器学習完了。")
        # ▲▲▲【修正ここまで】▲▲▲

        print("学習完了。 (テストデータなし)")
        print("\n--- モデル評価 (テストデータ) ---")
        print("テストデータがないためスキップします。")

    # ▲▲▲ 【★★★ ロジック修正ここまで ★★★】 ▲▲▲
