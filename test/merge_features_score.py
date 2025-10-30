# ファイル名: merge_features_score.py
# ('Ｒ' カラム読み込みを修正)

import polars as pl
import os
import unicodedata
import warnings

# --- 設定 ---
FEATURES_CSV_PATH = "features_engineered_pathB.csv"
SCORE_CSV_PATH = "data/processed/backtest_predictions_raw_features_pit.csv"
OUTPUT_CSV_PATH = "data/processed/features_with_score_merged.csv"


# --- ヘルパー関数 ---
def normalize_nkfc_pl(text: str | None) -> str | None:
    if isinstance(text, str):
        return unicodedata.normalize("NFKC", text)
    return text


def main():
    # warnings.filterwarnings('ignore', category=pl.PolarsInefficientMapWarning) # コメントアウトしても良い

    print(f"--- データマージ処理を開始 ---")
    print(f"特徴量データ: {FEATURES_CSV_PATH}")
    print(f"スコアデータ: {SCORE_CSV_PATH}")

    # --- 1. 特徴量データ読み込み ---
    try:
        df_features = pl.read_csv(FEATURES_CSV_PATH, infer_schema_length=10000)
        if "日付" in df_features.columns:
            df_features = df_features.with_columns(
                pl.col("日付").str.strptime(pl.Date, strict=False)
            )
        if "馬名" in df_features.columns:
            df_features = df_features.with_columns(
                pl.col("馬名")
                .cast(pl.Utf8)
                .map_elements(normalize_nkfc_pl, return_dtype=pl.Utf8)
                .str.strip_chars()
                .str.replace_all(r"\s+", "")
                .str.replace_all(r"[▲△▽☆]", "")
                .alias("馬名")
            )
        if "race_id" not in df_features.columns:
            print("  - 特徴量データに race_id がないため生成します...")
            if not all(c in df_features.columns for c in ["日付S", "場所", "Ｒ"]):
                raise ValueError(
                    "race_id 生成に必要なカラム (日付S, 場所, Ｒ) がありません。"
                )
            df_features = df_features.with_columns(
                pl.col("日付S")
                .str.strptime(pl.Date, format="%Y.%m.%d", strict=False)
                .alias("日付")
            ).drop_nulls("日付")
            df_features = df_features.with_columns(
                pl.col("Ｒ")
                .cast(pl.Utf8)
                .map_elements(normalize_nkfc_pl, return_dtype=pl.Utf8)
                .str.extract(r"(\d+)", 0)
                .cast(pl.Int64, strict=False)
                .alias("Ｒ")
            )
            df_features = df_features.with_columns(
                (
                    pl.col("日付S").cast(pl.Utf8)
                    + "_"
                    + pl.col("場所").cast(pl.Utf8)
                    + "_"
                    + pl.col("Ｒ").cast(pl.Utf8)
                ).alias("race_id")
            )
        print(
            f"特徴量データをロード完了 ({df_features.height}行, {df_features.width}列)"
        )
    except Exception as e:
        print(f"エラー: 特徴量データ読み込み失敗: {e}")
        return

    # --- 2. スコアデータ読み込み ---
    try:
        # ▼▼▼ 【★★★ cols_to_load から 'Ｒ' を削除 ★★★】 ▼▼▼
        cols_to_load = [
            "race_id",
            "馬名",
            "予測スコア",
            "単勝オッズ_実",
            "着順",
            "日付",
            "馬番",
            "単勝オッズ",
        ]  # 'Ｒ' を削除
        # ▲▲▲ 【★★★ 修正ここまで ★★★】 ▲▲▲
        df_score = pl.read_csv(SCORE_CSV_PATH, columns=cols_to_load)
        df_score = df_score.with_columns(
            pl.col("日付").str.strptime(pl.Date, strict=False)
        )
        df_score = df_score.with_columns(
            pl.col("馬名")
            .cast(pl.Utf8)
            .map_elements(normalize_nkfc_pl, return_dtype=pl.Utf8)
            .str.strip_chars()
            .str.replace_all(r"\s+", "")
            .str.replace_all(r"[▲△▽☆]", "")
            .alias("馬名")
        )
        print(f"スコアデータをロード完了 ({df_score.height}行, {df_score.width}列)")
    except Exception as e:
        print(f"エラー: スコアデータ読み込み失敗: {e}")
        return

    # --- 3. データをマージ ---
    print(f"\nデータを 'race_id' と '馬名' をキーにマージ中...")
    try:
        df_merged = df_features.join(
            df_score, on=["race_id", "馬名"], how="left", suffix="_score"
        )
        print(f"マージ完了。結果: ({df_merged.height}行, {df_merged.width}列)")
        null_score_count = df_merged["予測スコア"].is_null().sum()
        if null_score_count > 0:
            print(
                f"警告: マージ後、{null_score_count} 件の予測スコアが Null になりました。"
            )
        if "日付_score" in df_merged.columns:
            df_merged = df_merged.drop("日付_score")
        # (Rカラムは df_score にないので、重複処理は不要)
    except Exception as e:
        print(f"エラー: データマージ失敗: {e}")
        return

    # --- 4. マージ済みデータをCSVに保存 ---
    try:
        output_dir = os.path.dirname(OUTPUT_CSV_PATH)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        df_merged.write_csv(OUTPUT_CSV_PATH, include_bom=True)
        print(f"\n✅ マージ済みデータを {OUTPUT_CSV_PATH} に保存しました。")
    except Exception as e:
        print(f"エラー: マージ済みデータ保存失敗: {e}")


if __name__ == "__main__":
    main()
