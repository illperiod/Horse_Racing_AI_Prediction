import polars as pl
import pandas as pd
import numpy as np
import warnings
import os
from tqdm import tqdm
import unicodedata

# from sklearn.cluster import KMeans  # <-- 削除
# from sklearn.preprocessing import StandardScaler  # <-- 削除

# --- グローバル設定 ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- K-Means設定 (削除) ---
# (CLUSTER_N, CLUSTER_FEATURES は削除)


# ▼▼▼【ここから修正】▼▼▼
# normalize_nkfc をトップレベル（preprocess_data の外）に移動
def normalize_nkfc(text: str | None) -> str | None:
    """
    文字列をNFKC正規化するヘルパー関数。
    """
    if isinstance(text, str):
        return unicodedata.normalize("NFKC", text)
    return text


# ▲▲▲【ここまで修正】▲▲▲


def preprocess_data(race_path, pedigree_path):
    """
    レースデータと血統データを読み込み、基本的な前処理を行う関数。
    (pandas版の正規化処理などを移植)
    """
    print("--- データの読み込みと前処理を開始 ---")
    df_race = pl.read_csv(race_path, encoding="cp932", infer_schema_length=10000)
    df_ped = pl.read_csv(pedigree_path, encoding="cp932", infer_schema_length=10000)

    def time_to_seconds(time_str: str | None) -> float | None:
        if not isinstance(time_str, str) or not time_str:
            return None
        try:
            parts = time_str.split(":")
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            return float(parts[0])
        except (ValueError, TypeError):
            return None

    cols_to_process = [
        "着順",
        "人気",
        "斤量",
        "4角",
        "3角",
        "頭数",
        "馬体重",
        "馬番",
        "枠番",
        "距離",
        "PCI",
        "上り3F",
        "Ave-3F",
        "単勝オッズ",
        "走破タイム",
        "平均速度",
        "着差",
    ]
    for col in cols_to_process:
        if col in df_race.columns:
            base_expr = (
                pl.col(col)
                .cast(pl.Utf8)
                .map_elements(normalize_nkfc, return_dtype=pl.Utf8)
            )

            if col == "走破タイム":
                expression = base_expr.map_elements(
                    time_to_seconds, return_dtype=pl.Float64
                )
            else:
                expression = base_expr.str.extract(r"(\d+\.?\d*)", 0).cast(
                    pl.Float64, strict=False
                )

            df_race = df_race.with_columns(expression.alias(col))

    # ▼▼▼【ここから修正】▼▼▼
    print("  - 実配当データを処理中...")

    # 1. 複勝配当 (複勝圏内_実) の処理
    #    (元のCSVに '複勝配当' があり、それが実配当金であると仮定)
    original_fukusho_col = "複勝配当"
    new_fukusho_col = "複勝圏内_実"
    if original_fukusho_col in df_race.columns:
        df_race = df_race.with_columns(
            pl.col(original_fukusho_col)
            .cast(pl.Utf8)
            .map_elements(normalize_nkfc, return_dtype=pl.Utf8)
            .str.extract(r"(\d+\.?\d*)", 0)  # 数値を抽出
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)  # 配当がない(着外)場合は 0.0 に
            .alias(new_fukusho_col)
        )
        print(
            f"    -> {original_fukusho_col} を {new_fukusho_col} として追加しました。"
        )
    else:
        print(
            f"  - 警告: 元データに {original_fukusho_col} カラムが見つかりません。{new_fukusho_col} は生成されません。"
        )

    # 2. 単勝配当 (単勝オッズ_実) の処理
    #    (元の '単勝オッズ' と '着順' から生成)
    original_tansho_odds_col = "単勝オッズ"
    original_rank_col = "着順"
    new_tansho_payout_col = "単勝オッズ_実"

    # '単勝オッズ' と '着順' が正規化された後で実行
    if (
        original_tansho_odds_col in df_race.columns
        and original_rank_col in df_race.columns
    ):
        df_race = df_race.with_columns(
            pl.when(pl.col(original_rank_col) == 1)
            .then(pl.col(original_tansho_odds_col) * 100)  # 1着ならオッズ*100
            .otherwise(0.0)  # それ以外は0
            .fill_null(0.0)  # オッズがNullだった場合も0に
            .alias(new_tansho_payout_col)
        )
        print(
            f"    -> {original_tansho_odds_col} と {original_rank_col} から {new_tansho_payout_col} を生成しました。"
        )
    else:
        print(
            f"  - 警告: {original_tansho_odds_col} または {original_rank_col} が見つからないため、{new_tansho_payout_col} は生成されません。"
        )
    # ▲▲▲【ここまで修正】▲▲▲

    if "Ｒ" in df_race.columns:
        df_race = df_race.with_columns(
            pl.col("Ｒ")
            .cast(pl.Utf8)
            .map_elements(normalize_nkfc, return_dtype=pl.Utf8)
            .str.extract(r"(\d+)", 0)
            .cast(pl.Int64, strict=False)
            .alias("Ｒ")
        )

    df_race = (
        df_race.with_columns(
            pl.col("日付S")
            .str.strptime(pl.Date, format="%Y.%m.%d", strict=False)
            .alias("日付")
        )
        .drop_nulls("日付")
        .sort("日付")
    )

    df_race = df_race.with_columns(
        (
            pl.col("日付S").cast(pl.Utf8)
            + "_"
            + pl.col("場所").cast(pl.Utf8)
            + "_"
            + pl.col("Ｒ").cast(pl.Utf8)
        ).alias("race_id")
    )
    df_race = df_race.with_columns(
        pl.col("馬名").n_unique().over("race_id").alias("頭数")
    )

    def normalize_name(series):
        return (
            series.cast(pl.Utf8)
            .map_elements(normalize_nkfc, return_dtype=pl.Utf8)
            .str.strip_chars()
            .str.replace_all(r"\s+", "")
            .str.replace_all(r"[▲△▽☆]", "")
        )

    for col in ["騎手", "馬名", "調教師"]:
        if col in df_race.columns:
            df_race = df_race.with_columns(normalize_name(pl.col(col)).alias(col))
    for col in ["馬名", "種牡馬", "母父馬"]:
        if col in df_ped.columns:
            df_ped = df_ped.with_columns(normalize_name(pl.col(col)).alias(col))

    mawari_map = {
        "札幌": "右",
        "函館": "右",
        "福島": "右",
        "中山": "右",
        "京都": "右",
        "阪神": "右",
        "小倉": "右",
        "新潟": "左",
        "東京": "左",
        "中京": "左",
    }
    df_race = df_race.with_columns(pl.col("場所").replace(mawari_map).alias("回り"))

    df_race = df_race.with_columns(
        pl.when(pl.col("レース名").str.contains("G1|GI"))
        .then(pl.lit("G1"))
        .when(pl.col("レース名").str.contains("G2|GII"))
        .then(pl.lit("G2"))
        .when(pl.col("レース名").str.contains("G3|GIII"))
        .then(pl.lit("G3"))
        .when(pl.col("レース名").str.contains(r"\(L\)|リステッド"))
        .then(pl.lit("リステッド"))
        .when(pl.col("レース名").str.contains("3勝|３勝|1600万"))
        .then(pl.lit("3勝クラス"))
        .when(pl.col("レース名").str.contains("2勝|２勝|1000万"))
        .then(pl.lit("2勝クラス"))
        .when(pl.col("レース名").str.contains("1勝|１勝|500万"))
        .then(pl.lit("1勝クラス"))
        .when(pl.col("レース名").str.contains("未勝利"))
        .then(pl.lit("未勝利"))
        .when(pl.col("レース名").str.contains("新馬|メイクデビュー"))
        .then(pl.lit("新馬"))
        .otherwise(pl.lit("オープン"))
        .alias("クラスカテゴリ")
    )

    class_map = {
        "新馬": 0,
        "未勝利": 1,
        "1勝クラス": 2,
        "2勝クラス": 3,
        "3勝クラス": 4,
        "オープン": 5,
        "リステッド": 6,
        "G3": 7,
        "G2": 8,
        "G1": 9,
    }
    df_race = df_race.with_columns(
        pl.col("クラスカテゴリ")
        .replace(class_map)
        .cast(pl.Int64, strict=False)
        .fill_null(5)
        .alias("クラスレベル")
    )

    if "母父名" in df_ped.columns:
        df_ped = df_ped.rename({"母父名": "母父馬"})
    ped_cols_to_merge = ["馬名", "種牡馬", "母父馬"]
    df_ped = df_ped.unique(subset=["馬名"], keep="last")
    df = df_race.join(df_ped.select(ped_cols_to_merge), on="馬名", how="left")
    df = df.sort(["日付", "Ｒ", "馬番"])
    df = df.with_columns(
        (pl.col("場所").cast(pl.Utf8) + "_" + pl.col("距離").cast(pl.Utf8)).alias(
            "コース距離"
        )
    )
    print("--- 前処理完了 ---")
    return df


def calculate_dynamic_rates(
    df, key_cols, new_col_name, target_col="複勝圏内", condition_col=None
):
    """
    動的な率（勝率・複勝率など）を計算する汎用関数。
    """
    if condition_col:
        hit_expr = pl.col(target_col) * pl.col(condition_col)
        cnt_expr = pl.col(condition_col).cast(pl.Int32)
    else:
        hit_expr = pl.col(target_col)
        cnt_expr = pl.col(key_cols[0]).is_not_null().cast(pl.Int32)

    df = df.with_columns(
        [
            hit_expr.cum_sum().over(key_cols).shift(1).fill_null(0).alias("_cum_hit"),
            cnt_expr.cum_sum().over(key_cols).shift(1).fill_null(0).alias("_cum_cnt"),
        ]
    )
    df = df.with_columns(
        pl.when(pl.col("_cum_cnt") == 0)
        .then(0.0)
        .otherwise(pl.col("_cum_hit") / pl.col("_cum_cnt"))
        .alias(new_col_name)
    )
    return df.drop(["_cum_hit", "_cum_cnt"])


def feature_engineering(df_polars):
    print("--- 特徴量生成を開始 ---")

    print("--- パフォーマンス指標の0をnullに置換中 ---")
    metrics_to_correct = ["上り3F", "平均速度", "PCI", "Ave-3F", "走破タイム"]
    for col_name in metrics_to_correct:
        if col_name in df_polars.columns:
            df_polars = df_polars.with_columns(
                pl.when(pl.col(col_name) <= 0)
                .then(None)
                .otherwise(pl.col(col_name))
                .alias(col_name)
            )
    print("--- 置換完了 ---")

    df = df_polars.sort(["馬名", "日付"])

    key_entities = [
        "騎手",
        "調教師",
        "種牡馬",
        "母父馬",
        "場所",
        "馬場状態",
        "コース距離",
        "脚質",
        "芝・ダ",
    ]
    for col_name in key_entities:
        if col_name in df.columns:
            df = df.with_columns(pl.col(col_name).fill_null("不明"))

    # --- 基準タイム/速度の計算 ---
    condition_keys = ["場所", "芝・ダ", "距離", "馬場状態"]
    print(f"\n--- 基準タイム/速度の計算 (キー: {condition_keys}) ---")
    null_counts = df.select(
        [pl.col(c).is_null().sum().alias(f"{c}_null_count") for c in condition_keys]
    ).row(0, named=True)
    print(f"  キー列のNull数: {null_counts}")

    for metric in ["上り3F", "平均速度"]:
        if metric in df.columns:
            print(f"  - '{metric}' の基準値を計算中...")
            df = df.with_columns(pl.col(metric).cast(pl.Float64, strict=False))

            # 累積合計: null を 0.0 として合計
            cum_sum_expr = (
                pl.col(metric)
                .fill_null(0.0)
                .cum_sum()
                .over(condition_keys)
                .shift(1)
                .fill_null(0.0)
            )
            # 累積カウント: null でない行を 1、null 行を 0 として合計
            cum_count_expr = (
                pl.col(metric)
                .is_not_null()
                .cast(pl.Int32)
                .cum_sum()
                .over(condition_keys)
                .shift(1)
                .fill_null(0)
            )
            # expanding_mean を計算 (0除算を回避)
            expanding_mean = (
                pl.when(cum_count_expr > 0)
                .then(cum_sum_expr / cum_count_expr)
                .otherwise(None)
            )
            standard_metric_col = f"基準_{metric}"
            df = df.with_columns(expanding_mean.alias(standard_metric_col))

            null_ratio_standard = df.select(
                pl.col(standard_metric_col).is_null().mean() * 100
            ).row(0)[0]
            print(
                f"    -> '{standard_metric_col}' 生成完了 (Null率: {null_ratio_standard:.2f}%)"
            )

            # _vs_基準 の計算 (Null を考慮)
            vs_standard_col = f"{metric}_vs_基準"
            df = df.with_columns(
                pl.when(
                    pl.col(metric).is_not_null()
                    & pl.col(standard_metric_col).is_not_null()
                )
                .then(pl.col(metric) - pl.col(standard_metric_col))
                .otherwise(None)
                .alias(vs_standard_col)
            )
            null_ratio_vs_standard = df.select(
                pl.col(vs_standard_col).is_null().mean() * 100
            ).row(0)[0]
            print(
                f"    -> '{vs_standard_col}' 生成完了 (Null率: {null_ratio_vs_standard:.2f}%)"
            )

    # --- 基本的なフラグと差分 ---
    df = df.with_columns(
        [
            (pl.col("着順") == 1).cast(pl.Int32).alias("単勝"),
            (pl.col("着順") <= 3).cast(pl.Int32).alias("複勝圏内"),
            pl.col("日付")
            .diff()
            .over("馬名")
            .dt.total_days()
            .fill_null(0)
            .alias("レース間隔"),
        ]
    )
    df = df.with_columns(
        [
            (pl.col("レース間隔") >= 90).cast(pl.Int32).alias("長期休養明けフラグ"),
            (pl.col("レース間隔") < 14).cast(pl.Int32).alias("連闘フラグ"),
            pl.col("馬体重")
            .cast(pl.Float64, strict=False)
            .diff()
            .over("馬名")
            .fill_null(0)
            .alias("馬体重増減"),
            (pl.col("斤量").cast(pl.Float64, strict=False) / pl.col("馬体重")).alias(
                "斤量負担率"
            ),
            pl.when(
                (pl.col("騎手").shift(1).over("馬名").is_null())
                | (pl.col("騎手").is_null())
            )
            .then(0)
            .when(
                (pl.col("騎手") == pl.col("騎手").shift(1).over("馬名"))
                | (
                    pl.col("騎手").str.contains(
                        pl.col("騎手").shift(1).over("馬名"), literal=True
                    )
                )
                | (
                    pl.col("騎手")
                    .shift(1)
                    .over("馬名")
                    .str.contains(pl.col("騎手"), literal=True)
                )
            )
            .then(0)
            .otherwise(1)
            .cast(pl.Int32)
            .fill_null(0)
            .alias("騎手乗り替わりフラグ"),
            pl.col("レース間隔")
            .is_between(14, 28, closed="both")
            .cast(pl.Int32)
            .alias("叩き良化型フラグ"),
        ]
    )
    df = df.with_columns(
        (pl.col("馬体重増減") / pl.col("馬体重").shift(1).over("馬名"))
        .fill_null(0)
        .alias("馬体重増減率")
    )

    for col in ["4角", "3角"]:
        if col in df.columns:
            df = df.with_columns(
                (pl.col(col).cast(pl.Float64, strict=False) / pl.col("頭数")).alias(
                    f"{col}順位率"
                )
            )
    df = df.with_columns(
        (pl.col("クラスレベル") >= 7).cast(pl.Int32).alias("is_stakes_race")
    )

    df = df.with_columns(
        [
            (pl.arange(0, pl.len()).over("馬名") + 1).alias("キャリア"),
            pl.arange(0, pl.len()).over(["騎手", "調教師"]).alias("コンビ歴"),
        ]
    )
    df = df.with_columns(
        [
            (
                pl.col("キャリア")
                - pl.col("長期休養明けフラグ")
                .cum_sum()
                .over("馬名")
                .shift(1)
                .fill_null(0)
            ).alias("長期休養明け後レース数"),
            pl.col("単勝")
            .cum_sum()
            .over("馬名")
            .shift(1)
            .fill_null(0)
            .alias("通算勝利数"),
            pl.col("複勝圏内")
            .cum_sum()
            .over("馬名")
            .shift(1)
            .fill_null(0)
            .alias("通算複勝圏内"),
        ]
    )
    df = df.with_columns(
        [
            pl.when(pl.col("キャリア") <= 1)
            .then(0)
            .otherwise(pl.col("通算勝利数") / (pl.col("キャリア") - 1))
            .alias("通算勝率"),
            pl.when(pl.col("キャリア") <= 1)
            .then(0)
            .otherwise(pl.col("通算複勝圏内") / (pl.col("キャリア") - 1))
            .alias("通算複勝率"),
        ]
    )

    # --- キャリア平均/近5走平均の計算 ---
    cols_to_agg = [
        "PCI",
        "上り3F",
        "4角順位率",
        "クラスレベル",
        "平均速度",
        "Ave-3F",
        "上り3F_vs_基準",  # ★ Null率が低いことを確認 ★
        "平均速度_vs_基準",  # ★ Null率が低いことを確認 ★
        "馬体重",
    ]
    print("\n--- キャリア平均/近5走平均の計算 ---")
    for col in cols_to_agg:
        if col in df.columns:
            null_ratio_before = df.select(pl.col(col).is_null().mean() * 100).row(0)[0]
            print(
                f"  - '{col}' の平均を計算中 (計算前のNull率: {null_ratio_before:.2f}%)"
            )

            valid_mask = pl.col(col).is_not_null() & (pl.col(col) != 0)
            cumsum_col = (
                pl.when(valid_mask)
                .then(pl.col(col))
                .otherwise(0.0)
                .cum_sum()
                .over("馬名")
                .shift(1)
                .fill_null(0.0)
            )
            valid_count = (
                valid_mask.cast(pl.Int32).cum_sum().over("馬名").shift(1).fill_null(0)
            )
            career_avg_col_expr = (
                pl.when(valid_count > 0).then(cumsum_col / valid_count).otherwise(0.0)
            )
            rolling_mean_col_expr = (
                pl.when(valid_mask)
                .then(pl.col(col))
                .otherwise(None)
                .shift(1)
                .rolling_mean(5, min_periods=1)
                .over("馬名")
                .fill_null(0.0)
            )
            vs_self_col_expr = (
                pl.when(pl.col(col).is_not_null() & career_avg_col_expr.is_not_null())
                .then(pl.col(col) - career_avg_col_expr)
                .otherwise(None)
            )

            df = df.with_columns(
                [
                    career_avg_col_expr.alias(f"キャリア平均_{col}"),
                    rolling_mean_col_expr.alias(f"近5走平均_{col}"),
                    vs_self_col_expr.alias(f"{col}vs自己平均"),
                ]
            )
            null_ratio_career = df.select(
                pl.col(f"キャリア平均_{col}").is_null().mean() * 100
            ).row(0)[0]
            null_ratio_rolling = df.select(
                pl.col(f"近5走平均_{col}").is_null().mean() * 100
            ).row(0)[0]
            print(
                f"    -> キャリア平均 Null率: {null_ratio_career:.2f}%, 近5走平均 Null率: {null_ratio_rolling:.2f}%"
            )

    # --- 最速上がり3F, 馬体重キャリア平均差 ---
    df = df.with_columns(
        [
            pl.col("上り3F").shift(1).cum_min().over("馬名").alias("最速上がり3F"),
            (pl.col("馬体重") - pl.col("キャリア平均_馬体重")).alias(
                "馬体重キャリア平均差"
            ),
        ]
    )

    # --- キャリア/近5走平均 vs 基準 の特徴量を生成中 ---
    print("\n--- キャリア/近5走平均 vs 基準 の特徴量を生成中 ---")
    for metric in ["上り3F", "平均速度"]:
        standard_metric_col = f"基準_{metric}"
        if standard_metric_col in df.columns:
            print(
                f"  - 基準カラム '{standard_metric_col}' を使用して {metric} の差分を計算..."
            )
            for period in ["キャリア平均", "近5走平均"]:
                period_metric_col = f"{period}_{metric}"
                output_col_name = f"{period}_{metric}_vs_基準"
                if period_metric_col in df.columns:
                    print(
                        f"    - 特徴量 '{output_col_name}' を生成中 ('{period_metric_col}' - '{standard_metric_col}')..."
                    )
                    df = df.with_columns(
                        (
                            pl.col(period_metric_col).fill_null(0.0)
                            - pl.col(standard_metric_col).fill_null(0.0)
                        )
                        .fill_null(0.0)  # ★ Null の場合に 0.0 で埋める処理を追加 ★
                        .alias(output_col_name)
                    )
                else:
                    print(
                        f"    - スキップ: 必要な列 '{period_metric_col}' が見つかりません。"
                    )
        else:
            print(
                f"  - スキップ: {metric} の計算に必要な基準カラム '{standard_metric_col}' が見つかりません。"
            )

    # --- デバッグプリント (ループ直後) ---
    print("\nDEBUG: Immediately after final _vs_基準 loop:")
    cols_to_check = [
        "キャリア平均_上り3F_vs_基準",
        "キャリア平均_平均速度_vs_基準",
        "近5走平均_上り3F_vs_基準",
        "近5走平均_平均速度_vs_基準",
    ]
    existing_cols_after = [col for col in cols_to_check if col in df.columns]
    missing_cols_after = [col for col in cols_to_check if col not in df.columns]
    print(f"  存在する列 (ループ直後): {existing_cols_after}")
    if missing_cols_after:
        print(f"  ★存在しない列 (ループ直後): {missing_cols_after}")
    print("-" * 30)

    # --- レースメンバー平均PCI ---
    if "キャリア平均_PCI" in df.columns:
        df = df.with_columns(
            pl.col("キャリア平均_PCI")
            .mean()
            .over("race_id")
            .alias("レースメンバー平均PCI過去全体")
        )
        df = df.with_columns(
            pl.when(pl.col("レースメンバー平均PCI過去全体") < 40)
            .then(pl.lit("スロー予測"))
            .when(pl.col("レースメンバー平均PCI過去全体") < 60)
            .then(pl.lit("ミドル予測"))
            .otherwise(pl.lit("ハイ予測"))
            .alias("展開予測")
        )

    # --- 動的な率特徴量 ---
    print("  - 動的な率特徴量を生成中...")
    for entity in ["騎手", "調教師", "種牡馬", "母父馬"]:
        if entity in df.columns:
            df = calculate_dynamic_rates(df, [entity], f"{entity}_複勝率")

    combo_keys = [
        (["騎手", "場所"], "騎手_競馬場別複勝率"),
        (["騎手", "馬名"], "騎手・馬コンビ複勝率"),
        (["種牡馬", "距離"], "父馬_距離別複勝率"),
        (["馬名", "コース距離"], "馬_コース距離別複勝率"),
        (["馬名", "馬場状態"], "馬_馬場状態別複勝率"),
        (["騎手", "調教師"], "騎手_調教師相性複勝率"),
        (["種牡馬", "場所"], "種牡馬_競馬場別複勝率"),
        (["騎手", "種牡馬"], "騎手_種牡馬相性複勝率"),
        (["母父馬", "種牡馬"], "BMS産駒複勝率"),
        (["騎手", "馬場状態"], "騎手_馬場状態別複勝率"),
        (["騎手", "コース距離"], "騎手_コース距離別複勝率"),
        (["種牡馬", "距離"], "種牡馬_距離別複勝率"),
        (["種牡馬", "馬場状態"], "種牡馬_馬場状態別複勝率"),
    ]
    for cols, name in combo_keys:
        if all(c in df.columns for c in cols):
            df = calculate_dynamic_rates(df, cols, name)

    conditional_keys = [
        (["調教師"], "調教師_放牧明け成績", "長期休養明けフラグ"),
        (["調教師"], "調教師_連闘成績", "連闘フラグ"),
        (["騎手"], "騎手_重賞成績", "is_stakes_race"),
        (["調教師"], "調教師_重賞成績", "is_stakes_race"),
    ]
    for cols, name, cond_col in conditional_keys:
        if all(c in df.columns for c in cols + [cond_col]):
            df = calculate_dynamic_rates(df, cols, name, condition_col=cond_col)

    if "展開予測" in df.columns:
        for pace_type in ["ハイ予測", "ミドル予測", "スロー予測"]:
            df = df.with_columns(
                (pl.col("展開予測") == pace_type)
                .cast(pl.Int32)
                .alias(f"is_{pace_type}")
            )
            df = calculate_dynamic_rates(
                df, ["馬名"], f"馬_{pace_type}時複勝率", condition_col=f"is_{pace_type}"
            )
            df = df.drop(f"is_{pace_type}")

    # --- 騎手の脚質割合 ---
    if "脚質" in df.columns:
        print("  - 騎手の脚質割合を計算中...")
        style_map = {"逃げ": "逃げ", "先行": "先行", "中団": "差し", "後方": "追込"}
        jockey_total_races = (
            pl.arange(0, pl.len()).over("騎手") + 1
        )  # 0除算回避のため +1
        for style, name in style_map.items():
            style_flag = (pl.col("脚質") == style).cast(pl.Int32)
            style_sum = style_flag.cum_sum().over("騎手").shift(1).fill_null(0)
            df = df.with_columns(
                (style_sum / jockey_total_races)
                .fill_null(0.0)
                .alias(f"騎手_{name}割合")
            )

    # --- 世代と世代内対戦数 ---
    if "年齢" in df.columns:
        df = df.with_columns(
            (
                pl.col("日付").dt.year()
                - pl.col("年齢").cast(pl.Float64, strict=False).fill_null(3.0)
            ).alias("世代")
        )
        df = df.with_columns(
            pl.arange(0, pl.len()).over(["馬名", "世代"]).alias("世代内対戦数")
        )

    # --- クラス昇級/降級フラグ ---
    if "クラスレベル" in df.columns:
        df = df.with_columns(
            pl.col("クラスレベル")
            .cast(pl.Float64, strict=False)
            .shift(1)
            .over("馬名")
            .fill_null(0)
            .alias("前走クラスレベル")
        )
        df = df.with_columns(
            [
                (pl.col("クラスレベル") > pl.col("前走クラスレベル"))
                .cast(pl.Int32)
                .fill_null(0)
                .alias("クラス昇級フラグ"),
                (pl.col("クラスレベル") < pl.col("前走クラスレベル"))
                .cast(pl.Int32)
                .fill_null(0)
                .alias("クラス降級フラグ"),
            ]
        )

    # --- 期待順位スコア (rank_score) ---
    if "着順" in df.columns:
        print("  - 期待順位スコア (rank_score) を生成中...")
        rank_map = {1: 16.0, 2: 8.0, 3: 4.0, 4: 2.0, 5: 1.0}
        df = df.with_columns(
            pl.col("着順").replace(rank_map, default=0.0).alias("rank_score")
        )

    # --- 前走との差分特徴量など ---
    final_exprs = []
    if "着順" in df.columns:
        final_exprs.append(
            pl.col("着順").shift(1).over("馬名").fill_null(0).alias("前走着順")
        )
    if "距離" in df.columns:
        final_exprs.append(
            pl.col("距離")
            .cast(pl.Float64, strict=False)
            .diff(1)
            .over("馬名")
            .fill_null(0)
            .alias("前走との距離差")
        )
    if "上り3F" in df.columns:
        final_exprs.append(
            pl.col("上り3F").diff(1).over("馬名").fill_null(0).alias("前走との上り3F差")
        )
    if "平均速度" in df.columns:
        final_exprs.append(
            pl.col("平均速度")
            .diff(1)
            .over("馬名")
            .fill_null(0)
            .alias("前走との平均速度差")
        )
    if "キャリア平均_4角順位率" in df.columns:
        final_exprs.append(
            (1 - pl.col("キャリア平均_4角順位率").cast(pl.Float64, strict=False)).alias(
                "追走指数"
            )
        )
    if final_exprs:
        df = df.with_columns(final_exprs)

    # --- 一時列の削除 ---
    temp_cols = [
        col
        for col in df.columns
        if col.startswith("_") or col.startswith("is_") or col.startswith("基準_")
    ]
    if temp_cols:
        df = df.drop(temp_cols)

    # --- キャリア初期の補完 ---
    if "キャリア" in df.columns:
        print("  - キャリア初期の馬のデータを補完中...")
        no_history_mask = pl.col("キャリア") == 1
        numeric_cols = df.select(pl.selectors.numeric()).columns  # Use selector
        cols_to_exclude_from_imputation = [
            "単勝",
            "複勝圏内",
            "着順",
            "クラスレベル",
            "枠番",
            "馬番",
            "単勝",
            "複勝配当",
            "馬連",
            "馬単",
            "３連複",
            "３連単",
            "単勝オッズ_実",
            "複勝圏内_実",
            "長期休養明けフラグ",
            "連闘フラグ",
            "is_stakes_race",  # is_stakes_race は一時列なので削除されているはずだが念のため
            "叩き良化型フラグ",
            "騎手乗り替わりフラグ",
            "クラス昇級フラグ",
            "クラス降級フラグ",
            "rank_score",
            "頭数",
            "Ｒ",
            "距離",
            "斤量",  # これらも補完対象外
        ] + [col for col in numeric_cols if col.startswith("キャリア平均_")]

        cols_to_impute = [
            col
            for col in numeric_cols
            if col not in cols_to_exclude_from_imputation and col in df.columns
        ]

        maiden_mask = df["クラスレベル"].is_in([0, 1])
        mean_vals = {}
        for col in cols_to_impute:
            maiden_mean = df.filter(maiden_mask)[col].mean()
            mean_vals[col] = maiden_mean if maiden_mean is not None else df[col].mean()

        impute_exprs = []
        for col, mean_val in mean_vals.items():
            if mean_val is not None:
                impute_exprs.append(
                    pl.when(no_history_mask & pl.col(col).is_null())
                    .then(pl.lit(mean_val))
                    .otherwise(pl.col(col))
                    .alias(col)
                )
        if impute_exprs:
            df = df.with_columns(impute_exprs)

    # --- キャリア平均系特徴量の最終補正 ---
    print("--- キャリア平均系特徴量の最終補正 ---")
    career_avg_cols = [col for col in df.columns if col.startswith("キャリア平均_")]
    maiden_mask = df["クラスレベル"].is_in([0, 1])
    career_avg_defaults = {}
    for col in career_avg_cols:
        maiden_avg = df.filter(maiden_mask & (pl.col(col) > 0))[col].mean()
        career_avg_defaults[col] = maiden_avg if maiden_avg is not None else 0.0

    for col, default_val in career_avg_defaults.items():
        df = df.with_columns(
            pl.when(
                (pl.col("キャリア") > 1) & ((pl.col(col) == 0) | pl.col(col).is_null())
            )
            .then(pl.lit(default_val))
            .otherwise(pl.col(col))
            .alias(col)
        )
    print(f"キャリア平均系のデフォルト値: {career_avg_defaults}")

    # --- K-Meansクラスタリング (★削除★) ---
    print(f"\n--- K-Meansクラスタリング（スキップ） ---")
    print(f"  - (running_style_id の生成は削除されました)")
    print("-" * 30)
    # (L921-L953 の K-Means ロジックを削除)

    # --- デバッグ用 Print 文 (最終確認) ---
    print("\nDEBUG: feature_engineering の最終出力カラム確認:")
    cols_to_check = [
        "キャリア平均_上り3F_vs_基準",
        "キャリア平均_平均速度_vs_基準",
        "近5走平均_上り3F_vs_基準",
        "近5走平均_平均速度_vs_基準",
    ]
    existing_cols = [col for col in cols_to_check if col in df.columns]
    missing_cols = [col for col in cols_to_check if col not in df.columns]
    print(f"  存在する列: {existing_cols}")
    if missing_cols:
        print(f"  ★存在しない列: {missing_cols}")
    print("-" * 30)

    return df.fill_nan(None).fill_null(0)


if __name__ == "__main__":
    RACE_DATA_PATH = "2010_2025_data_v2.csv"
    PEDIGREE_DATA_PATH = "2005_2025_Pedigree.csv"
    OUTPUT_PATH = "features_engineered.csv"

    if not os.path.exists(RACE_DATA_PATH) or not os.path.exists(PEDIGREE_DATA_PATH):
        print(
            f"エラー: データファイルが見つかりません。({RACE_DATA_PATH}, {PEDIGREE_DATA_PATH})"
        )
    else:
        try:
            print("⚡ Polarsベースの高速特徴量生成を開始 (パスB: 回帰スコア版)...")
            main_df = preprocess_data(RACE_DATA_PATH, PEDIGREE_DATA_PATH)

            print("--- 同名馬・長期休養馬のキャリアを補正中... ---")
            gap_threshold_days = 1460
            main_df = main_df.sort(["馬名", "日付"])
            main_df = (
                main_df.with_columns(
                    (pl.col("日付").diff().dt.total_days().over("馬名")).alias(
                        "_前走からの日数"
                    )
                )
                .with_columns(
                    (pl.col("_前走からの日数") >= gap_threshold_days)
                    .fill_null(False)
                    .cast(pl.Int32)
                    .alias("_キャリア区切り")
                )
                .with_columns(
                    pl.col("_キャリア区切り")
                    .cum_sum()
                    .over("馬名")
                    .alias("_キャリア区間ID")
                )
                .with_columns(
                    pl.col("_キャリア区間ID")
                    .max()
                    .over("馬名")
                    .alias("_最新キャリア区間ID")
                )
            )
            main_df = main_df.filter(
                pl.col("_キャリア区間ID") == pl.col("_最新キャリア区間ID")
            )
            main_df = main_df.drop(
                [
                    "_前走からの日数",
                    "_キャリア区切り",
                    "_キャリア区間ID",
                    "_最新キャリア区間ID",
                ]
            )
            main_df = main_df.sort("日付")
            print("--- キャリア補正完了 ---")

            featured_data = feature_engineering(main_df)

            featured_data.write_csv(OUTPUT_PATH, include_bom=True)
            print(
                f"\n✅ 処理が完了しました。特徴量データが {OUTPUT_PATH} に保存されました。"
            )

        except Exception as e:
            print(f"\nエラーが発生しました: {e}")
            import traceback

            traceback.print_exc()
