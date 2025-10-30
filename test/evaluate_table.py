import pandas as pd
import os
import sys

# --- 設定 ---
INPUT_CSV_PATH = "data/processed/backtest_predictions_raw_features_pit.csv"


def calculate_top_scorer_return_rate_11R_only(csv_path):
    """
    CSVを読み込み、各レースで最も予測スコアが高い馬のうち、
    「race_id の末尾が '_11' のレース」のみを対象として
    単勝馬券を買い続けた場合の単勝回収率を計算する。
    """
    try:
        # --- 1. CSVファイルの読み込み ---
        if not os.path.exists(csv_path):
            print(f"エラー: ファイルが見つかりません: {csv_path}", file=sys.stderr)
            print(
                "まず create_rl_table.py を実行して、CSVファイルを生成してください。",
                file=sys.stderr,
            )
            return

        df = pd.read_csv(csv_path)
        print(f"CSVファイルを読み込みました: {csv_path} (総馬匹数: {len(df)})")

        # --- 2. 必要なカラムの存在チェック ---
        required_cols = ["race_id", "予測スコア", "単勝オッズ_実", "着順"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(
                f"エラー: 必要なカラム {missing_cols} がCSVに存在しません。",
                file=sys.stderr,
            )
            print(f"存在するカラム: {df.columns.tolist()}", file=sys.stderr)
            return

        # --- 3. 各レースで予測スコア1位の馬を抽出 ---
        print("各レースで「予測スコア」が1位の馬を抽出中...")

        try:
            top_scorers_idx = df.groupby("race_id")["予測スコア"].idxmax()
            top_scorers_df = df.loc[top_scorers_idx]
        except ValueError as e:
            print(
                f"エラー: データを処理できませんでした（DataFrameが空の可能性があります）: {e}",
                file=sys.stderr,
            )
            return

        # ▼▼▼ 【★★★ 変更点 ★★★】 ▼▼▼
        # race_id の末尾が '_11' のレースのみに絞り込む
        print("「race_id」が '_11' で終わるレース（11R）のみに絞り込みます。")
        # .copy() をつけて SettingWithCopyWarning を回避
        top_scorers_df = top_scorers_df[
            top_scorers_df["race_id"].astype(str).str.endswith("_11")
        ].copy()
        # ▲▲▲ 【★★★ 変更点ここまで ★★★】 ▲▲▲

        # --- 4. 回収率の計算 ---
        # 投資レース数（＝スコア1位の馬がいて、かつ11Rだったレース数）
        total_bets = len(top_scorers_df)

        if total_bets == 0:
            print("分析対象のレース（11R）が0件でした。")
            return

        # 総投資額 (1レース100円と仮定)
        total_investment = total_bets * 100

        # 総払戻額
        total_payout = top_scorers_df["単勝オッズ_実"].fillna(0).sum()

        # 単勝回収率 (Return Rate)
        if total_investment == 0:
            return_rate = 0.0
        else:
            return_rate = (total_payout / total_investment) * 100

        # --- (参考) 的中率の計算 ---
        total_wins = (top_scorers_df["単勝オッズ_実"].fillna(0) > 0).sum()

        if total_bets == 0:
            hit_rate = 0.0
        else:
            hit_rate = (total_wins / total_bets) * 100

        # --- 5. 結果表示 ---
        print("\n" + "=" * 50)
        print("--- 単勝回収率シミュレーション結果 (11Rのみ) ---")
        print("ロジック: 各レースで「予測スコア」が最も高い馬の単勝を100円ずつ購入")
        print("  対象R: 'race_id' の末尾が '_11' のレース")
        print("=" * 50)
        print(f"  対象レース数 (購入レース数): {total_bets} レース")
        print(f"  総投資額: {total_investment:,.0f} 円")
        print(f"  総払戻額: {total_payout:,.0f} 円")
        print(f"  的中レース数: {total_wins} レース")
        print("-" * 50)
        print(f"  単勝回収率: {return_rate:.2f} %")
        print(f"  的中率: {hit_rate:.2f} %")
        print("=" * 50)

    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {csv_path}", file=sys.stderr)
        print(
            "create_rl_table.py を実行して、まずCSVファイルを生成してください。",
            file=sys.stderr,
        )
    except pd.errors.EmptyDataError:
        print(f"エラー: ファイルは存在しますが、空です: {csv_path}", file=sys.stderr)
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # スクリプト名を 'calculate_baseline_return_11R.py' などとして保存することを推奨
    calculate_top_scorer_return_rate_11R_only(INPUT_CSV_PATH)
