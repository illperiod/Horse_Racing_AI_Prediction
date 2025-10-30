# preprocess_sequences.py
import polars as pl
import numpy as np
import joblib
import json
import os
from tqdm import tqdm
import gc
from deepctr_torch.inputs import SparseFeat, DenseFeat

# --- 設定 (keiba_analysis.py, evaluate.py と合わせる) ---
INPUT_DATA_PATH = "features_engineered_pathB.csv"
PREPROCESSOR_BASE_DIR = "lstm_preprocessor_score"  # ロード用
MODEL_OUTPUT_DIR = "lstm_models_score"
OUTPUT_NPZ_PATH = "preprocessed_lstm_data.npz"  # 出力ファイル名
N_STEPS = 5
TARGET_COL = "rank_score"

# 回収率計算に必要なメタデータ列 (元データに存在することを確認)
META_COLS_FOR_EVAL = [
    "race_id",
    "馬名",
    "日付",
    "単勝オッズ_実",
    "複勝圏内_実",
    "単勝",  # 単勝的中フラグ (evaluate.py で使用)
    "単勝オッズ",  # レース前の単勝オッズ (evaluate.py で使用)
    "着順",  # 追加
    "馬番",  # 追加
    "馬連",  # 追加 (馬連配当金)
    "馬単",  # 追加 (馬単配当金)
    "３連複",  # 追加 (三連複配当金)
    "３連単",  # 追加 (三連単配当金)
]


# --- ヘルパー関数 ---
def apply_mappings_pl(df, col, mapping):
    """Polars DataFrameにマッピングを適用 (旧バージョン互換の map_elements を使用)"""
    mapping_filled = {str(k): int(v) for k, v in mapping.items()}
    # map_elements を使って辞書マッピングを実行
    return df.with_columns(
        pl.col(col)
        .cast(pl.Utf8)
        .map_elements(
            lambda x: mapping_filled.get(
                x, 0
            ),  # mappingにキーがあれば値を、なければデフォルト値(0)を返す
            return_dtype=pl.Int32,  # 出力の型を指定
        )
        .alias(col)
    )


# --- メイン処理 ---
def main():
    print("--- Polarsによる事前処理プログラムを開始 ---")
    print(f"入力ファイル: {INPUT_DATA_PATH}")
    print(f"設定 N_STEPS: {N_STEPS}")

    # --- 1. 前処理オブジェクトのロード ---
    print(f"--- 前処理オブジェクトを '{PREPROCESSOR_BASE_DIR}' からロード中 ---")
    try:
        scaler = joblib.load(
            os.path.join(PREPROCESSOR_BASE_DIR, "numerical_scaler.pkl")
        )
        with open(
            os.path.join(PREPROCESSOR_BASE_DIR, "categorical_mappings.json"),
            "r",
            encoding="utf-8",
        ) as f:
            all_mappings = json.load(f)

        # ▼▼▼【ここから修正】▼▼▼
        # 'feature_columns.pkl' をロードして、モデルが期待する正しい順序を取得する
        all_feature_columns = joblib.load(
            os.path.join(
                MODEL_OUTPUT_DIR, "feature_columns.pkl"
            )  # MODEL_OUTPUT_DIR を参照
        )

        # pkl からロードしたリストは、名前だけのセットに変換
        static_name_set = set(
            joblib.load(os.path.join(PREPROCESSOR_BASE_DIR, "static_feature_names.pkl"))
        )
        sequence_name_set = set(
            joblib.load(
                os.path.join(PREPROCESSOR_BASE_DIR, "sequence_feature_names.pkl")
            )
        )

        # all_feature_columns (正しい順序) を、static / sequence に振り分ける
        static_fcs_sorted = [
            fc for fc in all_feature_columns if fc.name in static_name_set
        ]
        sequence_fcs_sorted = [
            fc for fc in all_feature_columns if fc.name in sequence_name_set
        ]

        # 使うのは、この「名前の順序付きリスト」
        static_feature_names_ordered = [fc.name for fc in static_fcs_sorted]
        sequence_feature_names_ordered = [fc.name for fc in sequence_fcs_sorted]

        # 元のリストは削除 (または使わない)
        # sequence_feature_names = joblib.load(...) # <-- 不要
        # static_feature_names = joblib.load(...) # <-- 不要

        print(
            f"  - Static features (順序補正済): {len(static_feature_names_ordered)}件"
        )
        print(
            f"  - Sequence features (順序補正済): {len(sequence_feature_names_ordered)}件"
        )
        print("前処理オブジェクトのロード完了")
        # ▲▲▲【ここまで修正】▲▲▲

    except Exception as e:
        print(f"エラー: 前処理オブジェクトのロードに失敗しました: {e}")
        return

    # 特徴量リストの整理 (ここも修正)
    # ▼▼▼【ここから修正】▼▼▼
    categorical_features = list(all_mappings.keys())
    numerical_features_static_scaled = scaler.feature_names_in_.tolist()

    # シーケンス内の数値特徴量 (スケーリング対象外)
    sequence_numerical_names = [
        name
        for name in sequence_name_set  # set を参照
        if name not in all_mappings and name != TARGET_COL
    ]

    numerical_features_all = list(
        set(numerical_features_static_scaled + sequence_numerical_names)
    )
    # ▲▲▲【ここまで修正】▲▲▲
    # --- 2. データ読み込みと初期処理 (Polars) ---
    print(f"--- データ ({INPUT_DATA_PATH}) を Polars で読み込み中 ---")
    try:
        # 型推測を増やし、日付もパース
        df_pl = pl.read_csv(
            INPUT_DATA_PATH, try_parse_dates=True, infer_schema_length=10000
        )

        # ソート (シーケンス作成の前提)
        df_pl = df_pl.sort(["馬名", "日付"])

        # 評価に必要なメタデータ列が存在するか確認
        missing_meta = [col for col in META_COLS_FOR_EVAL if col not in df_pl.columns]
        if missing_meta:
            print(f"警告: 評価に必要なメタデータ列が見つかりません: {missing_meta}")
            # とりあえず続行するが、evaluate.py で問題になる可能性あり
            # 必要ならダミー列を追加
            # for col in missing_meta:
            #     df_pl = df_pl.with_columns(pl.lit(None).alias(col))

    except Exception as e:
        print(f"エラー: データ読み込みまたは初期処理に失敗しました: {e}")
        return

    # --- 3. 前処理の適用 (Polars) ---
    print("--- 前処理を Polars で適用中 ---")
    try:
        # カテゴリ特徴量マッピング
        print("  - カテゴリ特徴量のマッピング...")
        for col in tqdm(categorical_features, desc="Mapping"):
            if col in df_pl.columns:
                mapping = all_mappings.get(col)
                if mapping:
                    df_pl = apply_mappings_pl(df_pl, col, mapping)
                # else: print(f"    - Warning: Mapping not found for {col}") # Debug
            # else: print(f"    - Warning: Column {col} not in df") # Debug

        # 単勝オッズの対数変換 (前処理の一部としてここで実施)
        if (
            "単勝オッズ" in df_pl.columns
            and "単勝オッズ" in numerical_features_static_scaled
        ):
            print("  - 単勝オッズの対数(log1p)変換...")
            # inf/-inf/null を 0 で埋めてから log1p
            df_pl = df_pl.with_columns(
                pl.col("単勝オッズ")
                .fill_nan(0.0)
                .fill_null(0.0)
                .log1p()
                .alias("単勝オッズ")
            )

        # 静的数値特徴量のスケーリング
        print(
            f"  - 静的数値特徴量 ({len(numerical_features_static_scaled)}件) のスケーリング..."
        )
        static_num_cols_in_df = [
            col for col in numerical_features_static_scaled if col in df_pl.columns
        ]
        if static_num_cols_in_df:
            # スケーリングのために NumPy に変換
            print("    - NumPyに変換中...")
            np_data_to_scale = df_pl.select(static_num_cols_in_df).to_numpy()

            # NaN/Infを0で置換 (NumPy側で)
            np_data_to_scale = np.nan_to_num(
                np_data_to_scale, nan=0.0, posinf=0.0, neginf=0.0
            )

            # スケーリング実行
            print("    - スケーリング実行中...")
            scaled_data = scaler.transform(np_data_to_scale)

            # スケーリング結果を Polars DataFrame に戻す
            print("    - Polarsに戻し中...")
            df_scaled = pl.DataFrame(scaled_data, schema=static_num_cols_in_df)

            # 元のDataFrameに結合 (元の列を削除して結合)
            df_pl = df_pl.drop(static_num_cols_in_df).hstack(df_scaled)
            print("    - スケーリング完了")
        else:
            print("    - スケーリング対象の静的数値特徴量がデータに存在しません。")

        # その他の数値特徴量 (シーケンス内など) の NaN/Inf 処理
        other_num_cols = [
            col
            for col in numerical_features_all
            if col not in numerical_features_static_scaled and col in df_pl.columns
        ]
        if other_num_cols:
            print(
                f"  - その他の数値特徴量 ({len(other_num_cols)}件) の NaN/Inf を 0 で置換..."
            )
            df_pl = df_pl.with_columns(
                [
                    pl.col(col).fill_nan(0.0).fill_null(0.0).alias(col)
                    for col in other_num_cols
                ]
            )

        # ターゲット列の NaN/Inf 処理
        if TARGET_COL in df_pl.columns:
            print(f"  - ターゲット列 ({TARGET_COL}) の NaN/Inf を 0 で置換...")
            df_pl = df_pl.with_columns(
                pl.col(TARGET_COL).fill_nan(0.0).fill_null(0.0).alias(TARGET_COL)
            )

        print("前処理適用完了")

    except Exception as e:
        print(f"エラー: 前処理の適用中に失敗しました: {e}")
        import traceback

        traceback.print_exc()
        return

    # --- 4. シーケンスデータの生成 (Polars Window Functions) ---
    print("--- シーケンスデータを Polars で生成中 ---")
    try:
        # 馬ごとのレースインデックス (0始まり) を生成
        # pl.cum_count()ではなく、グループ内の行番号を生成する
        df_pl = df_pl.with_columns(
            pl.int_range(0, pl.len()).over("馬名").alias("_horse_race_idx")
        )  # ← 修正

        # 過去N走分のデータを参照する式を生成
        lag_exprs = []
        for i in range(1, N_STEPS + 1):
            for col_name in sequence_feature_names_ordered:
                if col_name in df_pl.columns:
                    lag_exprs.append(
                        pl.col(col_name)
                        .shift(i)
                        .over("馬名")
                        .alias(f"{col_name}_lag_{i}")
                    )
                # else: print(f"    - Warning: Seq col {col_name} not found, skipping lag {i}") # Debug

        # Shiftを実行して新しい列を追加
        print(f"  - Shift ({len(lag_exprs)}列) を実行中...")
        df_pl = df_pl.with_columns(lag_exprs)

        # N_STEPS 分の過去データがない行を除外
        print(f"  - N_STEPS ({N_STEPS}) 未満の行を除外中...")
        original_rows = len(df_pl)
        df_sequences = df_pl.filter(pl.col("_horse_race_idx") >= N_STEPS)
        print(f"    - {original_rows} -> {len(df_sequences)} 行にフィルタリング")

        # 不要になったlag期間中の行とインデックス列を削除 (メモリ節約のため早めに)
        # df_pl = df_sequences # メモリに余裕があれば df_pl を残しても良い
        del df_pl  # メモリ解放
        gc.collect()
        df_sequences = df_sequences.drop("_horse_race_idx")

        # シーケンスデータをまとめる
        print("  - シーケンスデータを NumPy 配列に変換中...")
        sequence_data_np = []
        pbar_seq = tqdm(
            sequence_feature_names_ordered, desc="Structuring Sequences"
        )  # 順序付きリストを参照
        for col_name in pbar_seq:
            lag_cols = [
                f"{col_name}_lag_{i}" for i in range(N_STEPS, 0, -1)
            ]  # lag_5, lag_4, ..., lag_1 の順
            valid_lag_cols = [col for col in lag_cols if col in df_sequences.columns]

            if not valid_lag_cols:
                print(
                    f"    - 警告: 特徴量 '{col_name}' のlag列が見つかりません。ゼロで埋めます。"
                )
                # Add zeros array of shape (n_rows, N_STEPS)
                seq_np = np.zeros((len(df_sequences), N_STEPS))
            else:
                # Select the lag columns and convert to NumPy
                # fill_null(0) でパディング (shift で生じた Null を 0 埋め)
                seq_np = df_sequences.select(
                    [pl.col(c).fill_null(0) for c in valid_lag_cols]
                ).to_numpy()

                # If some lags were missing, pad with zeros
                if len(valid_lag_cols) < N_STEPS:
                    num_missing = N_STEPS - len(valid_lag_cols)
                    # Assuming lags are missing from the oldest (e.g., lag_5), pad at the beginning
                    padding = np.zeros((len(df_sequences), num_missing))
                    seq_np = np.hstack([padding, seq_np])

            # (n_rows, N_STEPS) -> (n_rows, N_STEPS, 1)
            sequence_data_np.append(seq_np[:, :, np.newaxis])

        # (n_features, n_rows, N_STEPS, 1) -> (n_rows, N_STEPS, n_features)
        X_seq_np = np.concatenate(sequence_data_np, axis=2)

        # メモリ解放: lag列を含む可能性のあるDataFrameを削除
        lag_column_names = [expr.meta.output_name() for expr in lag_exprs]
        df_sequences = df_sequences.drop(lag_column_names)
        del sequence_data_np
        gc.collect()

        print("シーケンスデータ生成完了")
        print(f"  - X_seq_np shape: {X_seq_np.shape}")

    except Exception as e:
        print(f"エラー: シーケンスデータの生成中に失敗しました: {e}")
        import traceback

        traceback.print_exc()
        return

    # --- 5. 静的データ、ラベル、メタデータをNumPy配列に変換 ---
    print("--- 静的データ、ラベル、メタデータを NumPy 配列に変換中 ---")
    try:
        # ▼▼▼【ここから修正】▼▼▼
        # 'static_feature_names_ordered' (正しい順序) と 'static_name_set' (全セット) は
        # セクション1 で .pkl からロード/生成済み。

        # 'static_feature_names_ordered' のうち、DataFrame (df_sequences) に *実際に* 存在する列を
        # *順序を維持したまま* フィルタリングする。
        final_static_cols_to_select = [
            col for col in static_feature_names_ordered if col in df_sequences.columns
        ]

        # チェック: .pkl のセットと、DFに存在するリストを比較
        if len(final_static_cols_to_select) != len(static_name_set):
            print(
                "警告: .pkl の static_feature_names セットと DataFrame の列が一致しません。"
            )
            print(f"  Expected from pkl (set): {static_name_set}")
            print(f"  Found in DataFrame (list): {set(final_static_cols_to_select)}")

            missing_cols = static_name_set - set(final_static_cols_to_select)
            if missing_cols:
                print(f"  Missing from DataFrame (will not be in NPZ): {missing_cols}")

        # DFに存在する列 (正しい順序) で NumPy 配列を生成
        X_static_np = df_sequences.select(final_static_cols_to_select).to_numpy()
        print(f"  - X_static_np shape: {X_static_np.shape}")
        # ▲▲▲【ここまで修正】▲▲▲

        y_np = df_sequences.select(TARGET_COL).to_numpy().squeeze()  # (n_rows,)
        print(f"  - y_np shape: {y_np.shape}")

        meta_cols_in_df = [
            col for col in META_COLS_FOR_EVAL if col in df_sequences.columns
        ]
        meta_data = df_sequences.select(meta_cols_in_df)
        print(f"  - Meta data shape: {meta_data.shape}")

        # (元のコードはここまで)

    except Exception as e:
        print(f"エラー: NumPy配列への変換中に失敗しました: {e}")
        return

    # --- 6. NumPy配列を .npz ファイルに保存 ---
    print(f"--- 処理済みデータを '{OUTPUT_NPZ_PATH}' に保存中 ---")
    try:
        save_dict = {
            "X_seq": X_seq_np,
            "X_static": X_static_np,
            "y": y_np,
            # メタデータも列ごとに保存 (evaluate.py で読み込みやすくするため)
            **{col: meta_data[col].to_numpy() for col in meta_cols_in_df},
        }

        # ▼▼▼【ここから修正】▼▼▼
        # どの静的特徴量がどの列インデックスに対応するかを保存
        # `X_static_np` と `final_static_cols_to_select` の順序と数が一致
        save_dict["static_feature_names_ordered"] = np.array(
            final_static_cols_to_select  # DFに実在する列のリスト (順序維持)
        )
        # どのシーケンス特徴量がどの列インデックスに対応するかを保存
        save_dict["sequence_feature_names_ordered"] = np.array(
            sequence_feature_names_ordered  # セクション1でロードした順序付きリスト
        )
        # ▲▲▲【ここまで修正】▲▲▲

        np.savez_compressed(OUTPUT_NPZ_PATH, **save_dict)
        print("保存完了")
    except Exception as e:
        print(f"エラー: .npz ファイルの保存中に失敗しました: {e}")
        return

    print(
        "\n" + "=" * 50 + "\n🎉 事前処理プログラムが正常に完了しました 🎉\n" + "=" * 50
    )
    print(f"出力ファイル: {OUTPUT_NPZ_PATH}")
    print("このファイルを evaluate.py および app.py から読み込んでください。")


if __name__ == "__main__":
    main()
