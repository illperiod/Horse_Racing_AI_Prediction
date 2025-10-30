# ファイル名: train_lightgbm_combined_v8_umaren_debug_final.py

import pandas as pd
import numpy as np
import warnings
import os

# import lightgbm as lgb # 学習スキップ

# --- 設定 ---
CSV_PATH = "data/processed/features_with_score_merged.csv"
DATA_START_DATE = "2024-01-01"
SPLIT_DATE = "2024-12-31"
# UMAREN_PAYOUT_COL はメイン関数内で定義

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
TARGET_COL = "profit_rank"


# --- 前処理関数 (再掲) ---
def preprocess_target_and_fillna(df, umaren_col_name):
    print("目的変数を計算し、Nullを補完中...")
    df["単勝オッズ_実"] = df["単勝オッズ_実"].fillna(0)
    df["馬番"] = pd.to_numeric(df["馬番"], errors="coerce").fillna(0).astype(int)
    df["着順_num"] = (
        pd.to_numeric(
            df["着順"].astype(str).str.normalize("NFKC").str.extract(r"(\d+)")[0],
            errors="coerce",
        )
        .fillna(99)
        .astype(int)
    )

    # 馬連払戻カラムを float に変換 (存在するなら)
    if umaren_col_name in df.columns:
        # シミュレーション用なので、0やNaNをそのまま残す (fillnaはしない)
        df[umaren_col_name] = pd.to_numeric(
            df[umaren_col_name], errors="coerce"
        ).fillna(0)

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

    feature_cols_in_df = [col for col in FEATURE_COLS if col in df.columns]
    df[feature_cols_in_df] = (
        df[feature_cols_in_df].fillna(0).replace([np.inf, -np.inf], 0)
    )
    print(f"{len(feature_cols_in_df)}個の特徴量カラムのNull/Infを0で補完しました。")

    df = df.dropna(subset=["race_id"])
    print("前処理完了。")
    return df


# --- 馬連軸流し評価関数 (再掲) ---
def evaluate_betting_strategy_umaren(
    df_test_results, rank_threshold, umaren_payout_col
):

    # 評価するカラムがデータに存在するか確認
    if umaren_payout_col not in df_test_results.columns:
        print(
            f"\n--- ベット戦略 (馬連 1位軸流し / N={rank_threshold} / カラム'{umaren_payout_col}'が見つかりません) ---"
        )
        return 0.0, 0.0

    print(
        f"\n--- ベット戦略 (馬連 1位軸流し / N={rank_threshold} / カラム'{umaren_payout_col}') ---"
    )

    pred_rank_col = "predict_rank_score"

    # レースIDごとの最大払戻額を確実に取得
    race_payouts = df_test_results.groupby("race_id")[umaren_payout_col].max().to_dict()

    df_temp = df_test_results.copy()
    df_temp["_rank_temp"] = df_temp["着順_num"].rank(method="first")
    actual_winners = df_temp[df_temp["_rank_temp"].isin([1, 2])]

    total_investment = 0
    total_payout = 0
    total_races = len(df_test_results["race_id"].unique())
    races_with_win = 0

    grouped = df_test_results.groupby("race_id")

    for race_id, race_df in grouped:
        top_n = race_df.nlargest(rank_threshold, pred_rank_col)
        if len(top_n) < 2:
            continue

        pivot_horse = top_n.iloc[0]["馬番"]
        flow_horses = top_n.iloc[1:rank_threshold]["馬番"].tolist()

        combinations_to_bet = set()
        for h in flow_horses:
            combinations_to_bet.add(tuple(sorted((pivot_horse, h))))

        num_bets = len(combinations_to_bet)
        if num_bets == 0:
            continue

        total_investment += num_bets * 100

        actual_pair_df = actual_winners[actual_winners["race_id"] == race_id]
        if len(actual_pair_df) < 2:
            continue

        actual_winning_pair = tuple(sorted(actual_pair_df["馬番"].tolist()))

        if actual_winning_pair in combinations_to_bet:
            payout = race_payouts.get(race_id, 0)
            total_payout += payout
            races_with_win += 1

    if total_investment == 0:
        print("  購入対象0件")
        return 0.0, 0.0

    return_rate = (
        (total_payout / total_investment) * 100 if total_investment > 0 else 0.0
    )
    hit_rate = (races_with_win / total_races) * 100 if total_races > 0 else 0.0

    print(
        f"  購入組み合わせ数: {total_investment / 100:,.0f} (総投資額: {total_investment:,.0f} 円)"
    )
    print(f"  総払戻額: {total_payout:,.0f}")
    print(f"  的中レース数: {races_with_win}")
    print("-" * 30)
    print(f"  ★ 的中レース率: {hit_rate:.2f} % ★")
    print(f"  ★ 回収率: {return_rate:.2f} % ★")
    return return_rate, hit_rate


# --- メイン処理 ---
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    # --- 1. データロード & 前処理 ---
    # ... (データロード処理 - 既存のコードを使用)
    try:
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError("CSVファイルが見つかりません。")

        df_full = pd.read_csv(CSV_PATH, parse_dates=["日付"], low_memory=False)
        print(f"ロード完了 ({len(df_full)}行)")

        start_date_dt = pd.to_datetime(DATA_START_DATE)
        df_full = df_full[df_full["日付"] >= start_date_dt].copy()
        print(
            f"  -> {start_date_dt.year}年以降のデータのみ ({len(df_full)}行) にフィルタリングしました。"
        )

        # 必須カラムチェックを緩める (デバッグのため)
        essential_cols = ["race_id", "日付", "馬番", "単勝オッズ_実", "着順"]
        if not all(col in df_full.columns for col in essential_cols):
            missing_essential = [
                col for col in essential_cols if col not in df_full.columns
            ]
            raise ValueError(f"CSVに必須カラム {missing_essential} が不足しています。")

        # dfのコピーを後で使うため、前処理は一旦最小限に
        df_base = df_full.copy()

    except Exception as e:
        print(f"エラー: データ処理中 {e}")
        exit()

    # --- 2. 時系列分割 & グループ準備 ---
    split_date = pd.to_datetime(SPLIT_DATE)

    # --- 3. 評価カラム名探索 ---

    # ★★★ 評価するカラム名の候補リスト ★★★
    PAYOUT_COL_CANDIDATES = ["馬連払戻"]
    # ★★★★★★★★★★★★★★★★★★★★★★★★★

    final_results = {}

    for candidate_col in PAYOUT_COL_CANDIDATES:
        if candidate_col not in df_base.columns:
            continue

        print(f"\n==============================================")
        print(f"  ▶︎▶︎▶︎ カラム名 '{candidate_col}' でシミュレーション開始")
        print(f"==============================================")

        # 候補カラムを使って前処理
        df = preprocess_target_and_fillna(df_base.copy(), candidate_col)

        df_train = (
            df[df["日付"] <= split_date].copy().sort_values(by=["race_id", "馬番"])
        )
        df_test = df[df["日付"] > split_date].copy().sort_values(by=["race_id", "馬番"])

        if len(df_test) == 0:
            print("テストデータがないためスキップ。")
            continue

        # --- 4. 予測スコア (ダミー予測) ---
        if not df_test.empty:
            np.random.seed(42)
            df_test["predict_rank_score"] = (
                1.0 / df_test["馬番"] + np.random.rand(len(df_test)) * 0.1
            )

        # --- 5. ベット戦略シミュレーション (Nを探索) ---
        current_candidate_results = {}

        for N in [2, 3, 4, 5]:
            return_rate, hit_rate = evaluate_betting_strategy_umaren(
                df_test,
                rank_threshold=N,
                umaren_payout_col=candidate_col,  # 候補カラム名を使用
            )

            # 結果を最終辞書に格納
            final_results[f"{candidate_col}_N{N}"] = {
                "カラム": candidate_col,
                "N": N,
                "回収率": return_rate,
                "的中レース率": hit_rate,
            }

    # --- 総合結果 ---
    print("\n\n=== 馬連払戻カラム探索 総合結果 ===")
    if final_results:
        results_df = pd.DataFrame(final_results).T
        results_df = results_df.sort_values(by="回収率", ascending=False)
        print(results_df)
    else:
        print(
            "シミュレーション対象のカラム（'馬連払戻', '馬連配当', '馬連'）がCSVに見つかりませんでした。"
        )
