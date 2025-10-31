import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import warnings

# --- [グローバル設定] ---
warnings.filterwarnings("ignore")
pd.set_option("display.width", 1000)
pd.set_option("display.max_columns", None)

# ▼▼▼【修正】DCN_LSTM_Hybridモデルの予測結果CSVを入力とする▼▼▼
INPUT_CSV_PATH = "./evaluation_results_lstm/predictions_lstm.csv"
MODEL_NAME = "DCN_LSTM_Hybrid (score)"
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

EVAL_START_YEAR = 2024  # 予測CSVがこの年以降のデータを含んでいるか確認

# --- [メイン処理] ---


def calculate_accuracy_metrics(df_test_raw):
    print("\n--- ランキング精度を計算中... ---")

    # 日付列を datetime に変換 (フィルタリング用)
    if "日付" in df_test_raw.columns:
        df_test_raw["日付"] = pd.to_datetime(df_test_raw["日付"])
        df_test = df_test_raw[df_test_raw["日付"].dt.year >= EVAL_START_YEAR].copy()
        if df_test.empty:
            print(
                f"--- [Error] {EVAL_START_YEAR}年以降の評価データがCSV内に見つかりません。 ---"
            )
            return
    else:
        print("警告: 日付列が見つからないため、全期間で評価します。")
        df_test = df_test_raw.copy()

    hits_top1, hits_top3, hits_top5 = 0, 0, 0
    total_precision_at_5, total_ndcg, num_races = 0, 0, 0

    # 必要な列を確認
    required_cols = ["race_id", "prediction", "着順", "馬番"]
    if not all(col in df_test.columns for col in required_cols):
        print(
            f"--- [Error] CSVに必要な列が不足しています。必要な列: {required_cols} ---"
        )
        return

    # (評価ロジックは元のスクリプトから流用)
    for race_id, race_df in tqdm(df_test.groupby("race_id"), desc="精度計算中"):
        if len(race_df) < 5:
            continue

        # 着順が 0 or NaN の馬は評価から除外 (元のロジック)
        race_df = race_df.dropna(subset=["着順"])
        race_df = race_df[race_df["着順"] > 0].copy()  # 0を除外

        if race_df.empty:
            continue

        # prediction が NaN の馬も除外
        race_df = race_df.dropna(subset=["prediction"])

        if race_df.empty:
            continue

        num_races += 1

        sorted_by_pred = race_df.sort_values("prediction", ascending=False)
        top1_pred_horse = sorted_by_pred.iloc[0]

        if top1_pred_horse["着順"] == 1:
            hits_top1 += 1
        if top1_pred_horse["着順"] <= 3:
            hits_top3 += 1
        if top1_pred_horse["着順"] <= 5:
            hits_top5 += 1

        pred_top5_set = set(sorted_by_pred.head(5)["馬番"])
        actual_top5_set = set(race_df[race_df["着順"] <= 5]["馬番"])
        correct_in_top5 = len(pred_top5_set.intersection(actual_top5_set))
        total_precision_at_5 += correct_in_top5 / 5.0

        ideal_top5 = race_df.nsmallest(5, "着順")
        if ideal_top5.empty:
            num_races -= 1  # 理想的な順位が計算できないレースは除外
            continue

        # 理想的なDCG (ideal_top5 が 5頭未満の場合も考慮)
        ideal_relevance = 1.0 / ideal_top5["着順"]
        ideal_dcg = (ideal_relevance / np.log2(np.arange(2, len(ideal_top5) + 2))).sum()

        # 予測のDCG (pred_top5 が 5頭未満の場合も考慮)
        pred_top5 = sorted_by_pred.head(5)
        actual_relevance = 1.0 / pred_top5["着順"]
        actual_dcg = (
            actual_relevance / np.log2(np.arange(2, len(pred_top5) + 2))
        ).sum()

        total_ndcg += actual_dcg / ideal_dcg if ideal_dcg > 0 else 0

    if num_races == 0:
        print("評価対象レースがありません。")
        return

    print("\n\n" + "=" * 80)
    print(" " * 28 + " Ranking Model 精度評価サマリー ")
    print("=" * 80)
    print(f"評価対象モデル: {MODEL_NAME}")
    print(f"評価対象レース数: {num_races} (開始年: {EVAL_START_YEAR})")
    print("-" * 80)
    print(f"1着的中率 (予測1位が実際に1着): {hits_top1 / num_races * 100:.2f}%")
    print(f"複勝率 (予測1位が3着以内): {hits_top3 / num_races * 100:.2f}%")
    print(f"5着以内率 (予測1位が5着以内): {hits_top5 / num_races * 100:.2f}%")
    print(
        f"Top-5 正答率 (予測上位5頭のうち実際に5着以内): {total_precision_at_5 / num_races:.3f} (平均 {total_precision_at_5 / num_races * 5 :.2f} / 5 頭)"
    )
    print(f"NDCG@5 (順位予測の正確性): {total_ndcg / num_races:.3f}")
    print("=" * 80)


if __name__ == "__main__":

    try:
        print(f"--- 予測結果ファイル {INPUT_CSV_PATH} を読み込み中 ---")
        df_preds = pd.read_csv(INPUT_CSV_PATH)
        df_preds["着順"] = pd.to_numeric(df_preds["着順"], errors="coerce")
        df_preds["馬番"] = pd.to_numeric(df_preds["馬番"], errors="coerce")
        print("--- 読み込み完了 ---")

    except FileNotFoundError:
        print(f"--- [Error] {INPUT_CSV_PATH} が見つかりません。 ---")
        print(
            "--- [Error] まず evaluate.py を実行して予測結果CSVを生成してください。 ---"
        )
        exit()
    except Exception as e:
        print(f"--- [Error] CSVの読み込みに失敗: {e} ---")
        exit()

    calculate_accuracy_metrics(df_preds)
