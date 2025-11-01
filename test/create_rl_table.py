import pandas as pd
import numpy as np
import joblib
import json
import os
import gc
import torch
import torch.nn as nn
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.layers import CrossNet, DNN
from deepctr_torch.inputs import SparseFeat, DenseFeat, combined_dnn_input
import traceback
import logging
import polars as pl
from datetime import datetime
import unicodedata  # ★ 追加

# --- ログ設定 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- DCN_LSTM_Hybrid モデルクラス定義 ---
# (L28-L277 の DCN_LSTM_Hybrid クラス定義は元のファイルからコピーしてそのままペースト)
class DCN_LSTM_Hybrid(BaseModel):
    # ... (モデル定義は変更なし、元のコードをここにコピー) ...
    def __init__(
        self,
        static_linear_feature_columns,
        static_dnn_feature_columns,
        sequence_feature_columns,
        static_feature_names,  # ★ 引数名修正 (元のコードに合わせる)
        lstm_hidden_size=64,
        lstm_layers=1,
        lstm_dropout=0.2,
        cross_num=2,  # ★ 元のコードに合わせる
        dnn_hidden_units=(256, 128),
        l2_reg_linear=0.00001,
        l2_reg_embedding=0.00001,
        l2_reg_cross=0.00001,
        l2_reg_dnn=0,
        init_std=0.0001,
        seed=1024,
        dnn_dropout=0,
        dnn_activation="relu",
        dnn_use_bn=False,
        task="regression",
        device="cpu",
        gpus=None,
    ):

        all_sparse_fcs = [
            fc
            for fc in static_linear_feature_columns + sequence_feature_columns
            if isinstance(fc, SparseFeat)
        ]

        # ★ BaseModelの引数を修正 (元のコードに合わせる)
        super(DCN_LSTM_Hybrid, self).__init__(
            linear_feature_columns=static_linear_feature_columns,
            dnn_feature_columns=static_dnn_feature_columns,  # ★ 元はこれだけだったはず
            l2_reg_linear=l2_reg_linear,
            l2_reg_embedding=l2_reg_embedding,
            init_std=init_std,
            seed=seed,
            task=task,
            device=device,
            gpus=gpus,
        )

        self.sequence_feature_columns = sequence_feature_columns
        self.static_linear_feature_columns = static_linear_feature_columns
        self.static_dnn_feature_columns = static_dnn_feature_columns
        self.dnn_hidden_units = dnn_hidden_units

        # --- LSTM Part ---
        seq_sparse_feature_columns = [
            fc for fc in sequence_feature_columns if isinstance(fc, SparseFeat)
        ]
        seq_dense_feature_columns = [
            fc for fc in sequence_feature_columns if isinstance(fc, DenseFeat)
        ]

        lstm_input_size = sum(
            self.embedding_dict[fc.embedding_name].embedding_dim
            for fc in seq_sparse_feature_columns
            if fc.embedding_name in self.embedding_dict
        ) + len(seq_dense_feature_columns)

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
        )

        # --- DCN Part ---
        static_dnn_input_dim = self.compute_input_dim(static_dnn_feature_columns)
        self.dnn_input_dim = static_dnn_input_dim + lstm_hidden_size

        self.dnn = DNN(
            self.dnn_input_dim,
            self.dnn_hidden_units,
            activation=dnn_activation,
            l2_reg=l2_reg_dnn,
            dropout_rate=dnn_dropout,
            use_bn=dnn_use_bn,
            init_std=init_std,
            device=device,
        )
        self.dnn_linear = nn.Linear(self.dnn_hidden_units[-1], 1, bias=False).to(device)

        self.cross_input_dim = self.compute_input_dim(static_linear_feature_columns)
        self.crossnet = CrossNet(self.cross_input_dim, cross_num, device=device)
        self.cross_linear = nn.Linear(self.cross_input_dim, 1, bias=False).to(device)

        self.to(device)

    def forward(self, X_seq_tensor, X_static_tensor):
        batch_size = X_static_tensor.shape[0]

        # --- Prepare LSTM Input ---
        seq_sparse_embedding_list = []
        seq_dense_value_list = []
        current_seq_col_idx = 0
        for fc in self.sequence_feature_columns:
            if isinstance(fc, SparseFeat):
                if fc.embedding_name in self.embedding_dict:
                    embedding_layer = self.embedding_dict[fc.embedding_name]
                    try:
                        # ★ .long() を追加 (IDは整数型である必要がある)
                        ids = X_seq_tensor[:, :, current_seq_col_idx].long()
                        emb = embedding_layer(ids)
                        seq_sparse_embedding_list.append(emb)
                    except IndexError:  # ★ エラーハンドリング追加
                        pass
                    except Exception as e:
                        logging.warning(f"LSTM Sparse Input Error ({fc.name}): {e}")
                        pass
                current_seq_col_idx += 1
            elif isinstance(fc, DenseFeat):
                try:
                    val = X_seq_tensor[:, :, current_seq_col_idx].unsqueeze(-1)
                    val = torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
                    seq_dense_value_list.append(val)
                except IndexError:  # ★ エラーハンドリング追加
                    pass
                except Exception as e:
                    logging.warning(f"LSTM Dense Input Error ({fc.name}): {e}")
                    pass
                current_seq_col_idx += 1

        # --- LSTM Processing ---
        if not seq_sparse_embedding_list and not seq_dense_value_list:
            lstm_output_features = torch.zeros((batch_size, self.lstm.hidden_size)).to(
                self.device
            )
        else:
            try:
                lstm_input_tensors = seq_sparse_embedding_list + seq_dense_value_list
                lstm_input = torch.cat(lstm_input_tensors, dim=-1)
                lstm_out, (hn, cn) = self.lstm(lstm_input)
                lstm_output_features = lstm_out[:, -1, :]
            except Exception as e:
                calculated_input_size = (
                    sum(t.shape[-1] for t in lstm_input_tensors)
                    if lstm_input_tensors
                    else 0
                )
                logging.error(
                    f"LSTM Error: Input Size Expected={self.lstm.input_size}, Got={calculated_input_size}. Err: {e}"
                )
                lstm_output_features = torch.zeros(
                    (batch_size, self.lstm.hidden_size)
                ).to(self.device)

        # --- Prepare DCN Input (Static Features) ---
        try:
            # ★ self.feature_index を渡すように修正 (BaseModelが作成したインデックス)
            linear_sparse_embedding_list, linear_dense_value_list = (
                self.input_from_feature_columns(
                    X_static_tensor,
                    self.static_linear_feature_columns,
                    self.embedding_dict,
                    self.feature_index,  # ★ 追加
                )
            )
            dnn_sparse_embedding_list, dnn_dense_value_list = (
                self.input_from_feature_columns(
                    X_static_tensor,
                    self.static_dnn_feature_columns,
                    self.embedding_dict,
                    self.feature_index,  # ★ 追加
                )
            )
        except Exception as e:
            logging.error(f"Error processing static features: {e}")
            # ★ エラー時も空リストを返すように変更
            linear_sparse_embedding_list, linear_dense_value_list = [], []
            dnn_sparse_embedding_list, dnn_dense_value_list = [], []
            # return torch.zeros((batch_size, 1)).to(self.device) # ★ エラーでも処理を継続

        # CrossNet Input
        if not linear_sparse_embedding_list and not linear_dense_value_list:
            # ★ self.cross_input_dim を参照するように修正
            cross_input = torch.zeros((batch_size, self.cross_input_dim)).to(
                self.device
            )
        else:
            cross_input = combined_dnn_input(
                linear_sparse_embedding_list, linear_dense_value_list
            )

        # DNN Input
        dnn_input_static = torch.zeros((batch_size, 0)).to(
            self.device
        )  # ★ 初期化を追加
        if dnn_sparse_embedding_list or dnn_dense_value_list:
            dnn_input_static = combined_dnn_input(
                dnn_sparse_embedding_list, dnn_dense_value_list
            )

        # ★ LSTM出力がNone or バッチサイズ不一致の場合の処理を改善
        if lstm_output_features is None or lstm_output_features.shape[0] != batch_size:
            zero_lstm_padding = torch.zeros((batch_size, self.lstm.hidden_size)).to(
                self.device
            )
            # ★ 静的特徴量がない場合も考慮
            if dnn_input_static.shape[1] > 0:
                dnn_input_combined = torch.cat(
                    [dnn_input_static, zero_lstm_padding], dim=-1
                )
            else:
                dnn_input_combined = (
                    zero_lstm_padding  # LSTMのみのはずがエラーなのでゼロ埋め
                )
        else:
            # ★ 静的特徴量がない場合も考慮
            if dnn_input_static.shape[1] > 0:
                dnn_input_combined = torch.cat(
                    [dnn_input_static, lstm_output_features], dim=-1
                )
            else:
                dnn_input_combined = lstm_output_features  # LSTM出力のみ

        # Check DNN input dimension
        expected_dnn_dim = self.dnn_input_dim
        actual_dnn_dim = dnn_input_combined.shape[1]
        if actual_dnn_dim != expected_dnn_dim:
            # ★★★ 次元不一致時のエラーハンドリングを追加 ★★★
            logging.error(
                f"DNN input dim mismatch: Expected={expected_dnn_dim}, Actual={actual_dnn_dim}"
            )
            # 形状が違う場合、期待される形状にゼロパディングまたは切り捨てを試みる (暫定対応)
            if actual_dnn_dim < expected_dnn_dim:
                padding = torch.zeros(
                    (batch_size, expected_dnn_dim - actual_dnn_dim)
                ).to(self.device)
                dnn_input_combined = torch.cat([dnn_input_combined, padding], dim=1)
                logging.warning(" -> Padded with zeros.")
            else:
                dnn_input_combined = dnn_input_combined[:, :expected_dnn_dim]
                logging.warning(" -> Truncated.")
            # 再度形状を確認
            actual_dnn_dim = dnn_input_combined.shape[1]
            if actual_dnn_dim != expected_dnn_dim:  # それでもダメならゼロを返す
                dnn_output = torch.zeros((batch_size, self.dnn_hidden_units[-1])).to(
                    self.device
                )
            else:
                dnn_output = self.dnn(dnn_input_combined)  # 修正後の入力で再試行
        else:
            dnn_output = self.dnn(dnn_input_combined)

        dnn_logit = self.dnn_linear(dnn_output)

        # CrossNet Layer
        cross_out = torch.zeros_like(dnn_logit)  # ★ 初期化を追加
        expected_cross_dim = self.cross_input_dim
        if cross_input.shape[1] > 0:  # ★ 入力が空でないかチェック
            actual_cross_dim = cross_input.shape[1]
            if actual_cross_dim == expected_cross_dim:
                cross_features = self.crossnet(cross_input)
                cross_out = self.cross_linear(cross_features)
            else:
                # ★★★ 次元不一致時のエラーハンドリングを追加 ★★★
                logging.error(
                    f"CrossNet input dim mismatch: Expected={expected_cross_dim}, Actual={actual_cross_dim}"
                )
                # (暫定対応: ゼロを返す)

        # Final prediction
        final_logit = cross_out + dnn_logit
        y_pred = self.out(final_logit)
        y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
        return y_pred


# --- (DCN_LSTM_Hybrid クラスここまで) ---


# ▼▼▼ [predict_logic.py L201-L215 からコピー] apply_mappings_pl ▼▼▼
def apply_mappings_pl(df, col, mapping):
    """Polars DataFrameにマッピングを適用"""
    # ★ キーを文字列に変換してから適用
    mapping_filled = {str(k): int(v) for k, v in mapping.items()}
    return df.with_columns(
        pl.col(col)
        .cast(pl.Utf8)  # ★ 文字列にキャスト
        .map_elements(
            lambda x: mapping_filled.get(x, 0),  # 存在しないキーは 0 (unknown)
            return_dtype=pl.Int32,  # ★ IDは整数
        )
        .alias(col)  # 元のカラム名を上書き
    )


# ▼▼▼ [features_engineered.py からインポート想定] ▼▼▼
try:
    # ★ normalize_nkfc もインポート
    from features_engineered import preprocess_data, feature_engineering, normalize_nkfc

    logging.info("features_engineered.py から関数をインポートしました。")
except ImportError as e:
    logging.error(f"features_engineered.py のインポートに失敗: {e}")

    # (フォールバック定義)
    def preprocess_data(race_path, pedigree_path):
        raise ImportError("features_engineered.py が見つかりません。")

    def feature_engineering(df_polars):
        raise ImportError("features_engineered.py が見つかりません。")

    def normalize_nkfc(text):
        raise ImportError("features_engineered.py が見つかりません。")


# ▼▼▼ [predict_logic.py L232-L260 からコピー] apply_career_reset ▼▼▼
def apply_career_reset(df_polars):
    """4年以上(1460日)の休養がある同名馬のキャリアをリセットする。"""
    logging.info("--- 同名馬・長期休養馬のキャリアを補正中... ---")
    gap_threshold_days = 1460
    df_polars = df_polars.sort(["馬名", "日付"])  # ★ ソートキー確認
    df_polars = (
        df_polars.with_columns(
            (pl.col("日付").diff().dt.total_days().over("馬名")).alias(
                "_前走からの日数"
            )
        )
        .with_columns(
            (pl.col("_前走からの日数") >= gap_threshold_days)
            .fill_null(False)  # ★ fill_null追加
            .cast(pl.Int32)
            .alias("_キャリア区切り")
        )
        .with_columns(
            pl.col("_キャリア区切り").cum_sum().over("馬名").alias("_キャリア区間ID")
        )
        .with_columns(
            pl.col("_キャリア区間ID").max().over("馬名").alias("_最新キャリア区間ID")
        )
    )
    # ★ 最新区間のみフィルタ
    df_polars = df_polars.filter(
        pl.col("_キャリア区間ID") == pl.col("_最新キャリア区間ID")
    )
    # ★ 一時列削除
    df_polars = df_polars.drop(
        ["_前走からの日数", "_キャリア区切り", "_キャリア区間ID", "_最新キャリア区間ID"]
    )
    logging.info("--- キャリア補正完了 ---")
    return df_polars


class Backtester:
    # ▼▼▼ 【★★★ __init__ を大幅に書き換え ★★★】 ▼▼▼
    def __init__(
        self,
        historical_data_path,  # ★生のCSVパス
        pedigree_data_path,  # ★生のCSVパス
        preprocessor_base_dir,
        model_base_dir,
    ):
        logging.info("--- バックテスター (Backtester) の初期化を開始 ---")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"使用デバイス: {self.device}")

        self.preprocessor_dir = preprocessor_base_dir
        self.model_dir = model_base_dir

        # --- 1. 設定 (N_STEPS) をロード ---
        self.n_steps = 5
        self.target_col = "rank_score"  # ★ ターゲットカラム名確認

        # --- 2. 履歴データと血統データをロード & ★★★ 3文字切り捨て ★★★ & 特徴量生成 ---
        try:
            logging.info(f"履歴データ {historical_data_path} を読み込み中...")
            df_historical_base = preprocess_data(
                historical_data_path, pedigree_data_path
            )
            df_historical_base = apply_career_reset(df_historical_base)
            logging.info("履歴データの前処理完了。")

            self.df_historical_base_raw = df_historical_base.clone()  # ← 追加

            # ▼▼▼ 【★★★ ここで3文字切り捨てを追加 ★★★】 ▼▼▼
            logging.info("  - (互換性) 騎手・調教師名を3文字に切り捨てます。")
            for col in ["騎手", "調教師"]:
                if col in df_historical_base.columns:
                    df_historical_base = df_historical_base.with_columns(
                        pl.col(col).str.slice(0, 3).alias(col)
                    )
            # ▲▲▲ 【★★★ ここまで追加 ★★★】 ▲▲▲

            logging.info(
                "全履歴データに対して特徴量生成 (feature_engineering) を実行中 (キャッシュ生成)..."
            )
            # ★ここで feature_engineering を実行し、結果をキャッシュとして保持する★
            # (ループ内での再計算を高速化するため)
            # 特徴量生成 (キャッシュ)
            logging.info("全履歴データに対して特徴量生成...")
            self.df_featured_all_cache_pl = feature_engineering(df_historical_base)

            # ★ 基準値カラムを抽出して保存 ★
            self.baseline_columns = [
                col for col in self.df_featured_all_cache_pl.columns if "_基準_" in col
            ]

            # ★ コース×馬場状態ごとの基準値を辞書化 ★
            self.baseline_dict = {}
            for col in self.baseline_columns:
                parts = col.split("_")
                if len(parts) >= 5:
                    metric = parts[0]
                    track_type = parts[2]
                    distance = parts[3]
                    condition = parts[4]
                    key = (track_type, distance, condition)

                    baseline_value = (
                        self.df_featured_all_cache_pl.filter(  # ★ キャッシュから取得
                            (pl.col("芝・ダ") == track_type)
                            & (pl.col("距離") == int(distance))
                            & (pl.col("馬場状態") == condition)
                        )
                        .select(col)
                        .drop_nulls()
                        .unique()
                    )

                    if len(baseline_value) > 0:
                        if key not in self.baseline_dict:
                            self.baseline_dict[key] = {}
                        self.baseline_dict[key][col] = baseline_value[0, 0]

            logging.info(f"基準値辞書を作成: {len(self.baseline_dict)} コース条件")

        except Exception as e:
            logging.error(f"履歴データ読み込みまたは特徴量キャッシュ生成に失敗: {e}")
            raise e

        # --- 3. 前処理オブジェクトをロード (predict_logic.py L355-L436 と同様) ---
        try:
            logging.info(f"{preprocessor_base_dir} から前処理オブジェクトをロード中...")
            self.scaler = joblib.load(
                os.path.join(preprocessor_base_dir, "numerical_scaler.pkl")
            )
            with open(
                os.path.join(preprocessor_base_dir, "categorical_mappings.json"),
                "r",
                encoding="utf-8",
            ) as f:
                self.all_mappings = json.load(f)
            # ★★★ feature_columns.pkl からロード ★★★
            all_feature_columns_pkl = joblib.load(
                os.path.join(model_base_dir, "feature_columns.pkl")
            )
            # ★★★ static/sequence の名前リストも pkl からロード ★★★
            static_name_set = set(
                joblib.load(
                    os.path.join(preprocessor_base_dir, "static_feature_names.pkl")
                )
            )
            sequence_name_set = set(
                joblib.load(
                    os.path.join(preprocessor_base_dir, "sequence_feature_names.pkl")
                )
            )
            # pkl からロードした情報を使ってリストを再構築
            self.static_fcs_sorted = [
                fc for fc in all_feature_columns_pkl if fc.name in static_name_set
            ]
            self.sequence_fcs_sorted = [
                fc for fc in all_feature_columns_pkl if fc.name in sequence_name_set
            ]
            self.static_feature_names_ordered = [
                fc.name for fc in self.static_fcs_sorted
            ]
            self.sequence_feature_names_ordered = [
                fc.name for fc in self.sequence_fcs_sorted
            ]
            self.categorical_features = list(
                self.all_mappings.keys()
            )  # マッピング辞書のキーを使用
            self.numerical_features_static_scaled = (
                self.scaler.feature_names_in_.tolist()  # scaler が知っている列
            )
            # ★ 全数値特徴量リストの再構築 ★
            all_numerical_names_in_pkl = [
                fc.name for fc in all_feature_columns_pkl if isinstance(fc, DenseFeat)
            ]
            self.numerical_features_all = list(
                set(
                    all_numerical_names_in_pkl
                )  # pkl に DenseFeat として定義されているもの全て
            )

            logging.info("前処理オブジェクトのロード完了")
        except Exception as e:
            logging.error(f"前処理オブジェクトのロードに失敗: {e}")
            raise e

        # --- 4. モデルをロード (predict_logic.py L440-L464 と同様) ---
        try:
            logging.info(f"{model_base_dir} からモデルをロード中...")
            model_path = os.path.join(model_base_dir, "lstm_hybrid_model.pth")
            self.model = DCN_LSTM_Hybrid(
                static_linear_feature_columns=self.static_fcs_sorted,
                static_dnn_feature_columns=self.static_fcs_sorted,
                sequence_feature_columns=self.sequence_fcs_sorted,
                static_feature_names=self.static_feature_names_ordered,  # ★ 修正
                lstm_hidden_size=64,  # ★ 値確認
                lstm_layers=1,  # ★ 値確認
                lstm_dropout=0.2,  # ★ 値確認
                cross_num=3,  # ★ 値確認 (学習時と合わせる)
                dnn_hidden_units=(256, 128),  # ★ 値確認
                l2_reg_linear=1e-5,  # ★ 値確認
                l2_reg_embedding=1e-5,  # ★ 値確認
                l2_reg_cross=1e-5,  # ★ 値確認
                l2_reg_dnn=1e-5,  # ★ 値確認 (学習時は 0 だった？) -> 1e-5 に統一
                dnn_dropout=0.3,  # ★ 値確認
                dnn_use_bn=True,  # ★ 値確認
                task="regression",
                device=self.device,
            )
            # ★ map_location を指定
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            logging.info("モデルのロードと評価モードへの設定完了")
        except Exception as e:
            logging.error(f"モデルのロードに失敗: {e}")
            raise e

        # --- 5. ★★★ .npz のロードは行わない ★★★ ---

        logging.info("--- バックテスターの初期化が完了 ---")

    # ▼▼▼ 【★★★ ヘルパー関数を predict_logic.py からコピー ★★★】 ▼▼▼

    def _preprocess_for_prediction(self, df_combined_pl):
        """
        (predict_logic.py L708-L777 からそのままコピー)
        ★ スケーリング実行部分を確認 ★
        """
        # --- 1. 前処理の適用 (Polars) ---
        df_pl = df_combined_pl
        # カテゴリ特徴量をマッピング辞書でIDに変換
        for col in self.categorical_features:
            if col in df_pl.columns:
                mapping = self.all_mappings.get(col)
                if mapping:
                    df_pl = apply_mappings_pl(df_pl, col, mapping)

        # 単勝オッズの対数変換 (スケーリング対象の場合)
        if (
            "単勝オッズ" in df_pl.columns
            and "単勝オッズ" in self.numerical_features_static_scaled
        ):
            df_pl = df_pl.with_columns(
                pl.col("単勝オッズ")
                .fill_nan(0.0)  # ★ Null/NaN 処理
                .fill_null(0.0)
                .log1p()  # ★ 対数変換
                .alias("単勝オッズ")
            )

        # スケーリング対象の静的数値特徴量の Null/NaN 埋め
        static_num_cols_in_df = [
            col for col in self.numerical_features_static_scaled if col in df_pl.columns
        ]
        if static_num_cols_in_df:
            exprs_to_fill = [
                pl.col(col).fill_nan(0.0).fill_null(0.0).alias(col)
                for col in static_num_cols_in_df
            ]
            df_pl = df_pl.with_columns(exprs_to_fill)

            # ★★★ スケーリング実行 ★★★
            # Polars DataFrame から NumPy 配列に変換
            np_data_to_scale = df_pl.select(static_num_cols_in_df).to_numpy()
            # 無限大も0に置換
            np_data_to_scale = np.nan_to_num(
                np_data_to_scale, nan=0.0, posinf=0.0, neginf=0.0
            )
            # scaler を使って変換
            scaled_data = self.scaler.transform(np_data_to_scale)
            # 結果を Polars DataFrame に戻す
            df_scaled = pl.DataFrame(scaled_data, schema=static_num_cols_in_df)
            # 元の列を削除し、スケーリング後の列を結合
            df_pl = df_pl.drop(static_num_cols_in_df).hstack(df_scaled)

        # スケーリング対象外の数値特徴量の Null/NaN 埋め
        other_num_cols = [
            col
            for col in self.numerical_features_all  # ★ 全数値特徴量リストを使用
            if col not in self.numerical_features_static_scaled and col in df_pl.columns
        ]
        if other_num_cols:
            df_pl = df_pl.with_columns(
                [
                    pl.col(col).fill_nan(0.0).fill_null(0.0).alias(col)
                    for col in other_num_cols
                ]
            )
        # ターゲット列の Null/NaN 埋め (存在する場合)
        if self.target_col in df_pl.columns:
            df_pl = df_pl.with_columns(
                pl.col(self.target_col)
                .fill_nan(0.0)  # ★ 0埋め
                .fill_null(0.0)
                .alias(self.target_col)
            )

        # --- 2. シーケンスデータの生成 (Polars Window Functions) ---
        # 馬ごとにレースインデックスを振る (0始まり)
        df_pl = df_pl.with_columns(
            pl.int_range(0, pl.len()).over("馬名").alias("_horse_race_idx")
        )
        lag_exprs = []
        # N_STEPS 分の過去データ (Lag) を作成
        for i in range(1, self.n_steps + 1):
            for col_name in self.sequence_feature_names_ordered:  # ★ 順序リストを使用
                if col_name in df_pl.columns:
                    lag_exprs.append(
                        pl.col(col_name)
                        .shift(i)  # i個前のレースの値を取得
                        .over("馬名")  # 馬ごとに計算
                        .alias(f"{col_name}_lag_{i}")  # 例: "馬体重_lag_1"
                    )
        df_pl = df_pl.with_columns(lag_exprs)
        return df_pl

    def _extract_tensors_from_df(self, df_processed_live):
        """
        (predict_logic.py L779-L865 からそのままコピー)
        ★ 特徴量名の順序と Null 埋め戦略を確認 ★
        """
        n_rows = len(df_processed_live)
        if n_rows == 0:
            logging.warning("_extract_tensors_from_df: 入力DataFrameが空です。")
            return None, None

        # --- 静的テンソル (X_static) の抽出 ---
        # モデルが期待する順序の特徴量名リストを使用
        final_static_cols_to_select = []
        missing_static_cols = []
        for col in self.static_feature_names_ordered:  # ★ 順序リストを使用
            if col in df_processed_live.columns:
                final_static_cols_to_select.append(col)
            else:
                missing_static_cols.append(col)
                final_static_cols_to_select.append(
                    pl.lit(0.0).alias(col)
                )  # ★ 欠損は0.0で埋める

        if missing_static_cols:
            logging.warning(
                f"静的特徴量が不足しています (0.0で補完): {missing_static_cols}"
            )

        try:
            # select で指定した順序で NumPy 配列に変換
            X_static_np = df_processed_live.select(
                final_static_cols_to_select
            ).to_numpy()
        except Exception as e:
            logging.error(f"静的テンソル抽出中にエラー: {e}")
            logging.error(f"対象カラム: {final_static_cols_to_select}")
            logging.error(f"DataFrameスキーマ: {df_processed_live.schema}")
            return None, None

        # --- シーケンス・テンソル (X_seq) の抽出 ---
        sequence_data_np_list = []  # 各特徴量の (n_rows, n_steps) 配列を格納するリスト
        for col_name in self.sequence_feature_names_ordered:  # ★ 順序リストを使用
            # Lag カラム名リスト (Lag N -> Lag 1 の順)
            lag_cols = [f"{col_name}_lag_{i}" for i in range(self.n_steps, 0, -1)]

            # 実際に DataFrame に存在する Lag カラムのみを選択
            select_exprs = []
            num_present_lags = 0
            for lag_col in lag_cols:
                if lag_col in df_processed_live.columns:
                    select_exprs.append(pl.col(lag_col))
                    num_present_lags += 1
                else:
                    # 存在しない Lag カラムは Null として扱う (後で fill_null)
                    select_exprs.append(pl.lit(None).alias(lag_col))

            # Null 埋めの値を決定 (カテゴリ/フラグは 0, 数値は 0.0)
            fill_value = 0.0
            if (
                (col_name in self.all_mappings)
                or (col_name.endswith("フラグ"))
                or (col_name.endswith("_id"))
            ):
                fill_value = 0

            # fill_null を適用した式リストを作成
            filled_exprs = [expr.fill_null(fill_value) for expr in select_exprs]

            try:
                # (n_rows, n_steps) の NumPy 配列に変換
                seq_np_for_col = df_processed_live.select(filled_exprs).to_numpy()
                sequence_data_np_list.append(
                    seq_np_for_col[:, :, np.newaxis]
                )  # (n_rows, n_steps, 1) に変換してリストに追加
            except Exception as e:
                logging.error(f"シーケンス特徴量 '{col_name}' の抽出中にエラー: {e}")
                logging.error(f"対象Lagカラム: {lag_cols}")
                logging.error(f"DataFrameスキーマ: {df_processed_live.schema}")
                # ★ エラー時はゼロ埋め配列を追加して処理を継続 (暫定)
                zero_seq = np.zeros((n_rows, self.n_steps, 1))
                sequence_data_np_list.append(zero_seq)

        if not sequence_data_np_list:
            # シーケンス特徴量が全くない場合
            X_seq_np = np.empty((n_rows, self.n_steps, 0), dtype=np.float32)
        else:
            # (n_rows, n_steps, n_features) の形状に結合
            try:
                X_seq_np = np.concatenate(sequence_data_np_list, axis=2)
            except ValueError as e:
                logging.error(f"シーケンス特徴量の結合中にエラー: {e}")
                logging.error(
                    "各シーケンス特徴量の形状が不揃いか、リストが空の可能性があります。"
                )
                # ★ エラー時は空のテンソルを返す (暫定)
                X_seq_np = np.empty((n_rows, self.n_steps, 0), dtype=np.float32)

        # NumPy 配列を PyTorch テンソルに変換
        try:
            X_seq_tensor = torch.tensor(X_seq_np, dtype=torch.float32).to(self.device)
            X_static_tensor = torch.tensor(X_static_np, dtype=torch.float32).to(
                self.device
            )
        except Exception as e:
            logging.error(f"テンソル変換中にエラー: {e}")
            logging.error(f"X_seq_np shape: {X_seq_np.shape}, dtype: {X_seq_np.dtype}")
            logging.error(
                f"X_static_np shape: {X_static_np.shape}, dtype: {X_static_np.dtype}"
            )
            return None, None

        return X_seq_tensor, X_static_tensor

    # ▼▼▼ 【★★★ run_backtest を大幅に書き換え ★★★】 ▼▼▼
    def run_backtest(
        self,
        output_csv_path="./data/processed/rl_betting_data.csv",  # ★ 出力ファイル名変更 (RL用)
        start_date_str="2024-01-01",  # バックテスト対象の開始日
        debug_mode=False,  # ★ デフォルトを False に変更 ★
        debug_race_id="2025.9.28_中山_5",  # デバッグ時に使用
    ):
        """
        バックテスト実行のメインロジック (★書き換え後: Point-in-Time特徴量計算 & RL用出力★)
        """
        try:
            # 1. バックテスト対象のレースIDと日付、コース情報を取得
            if debug_mode and debug_race_id:
                logging.info(f"\n{'='*60}")
                logging.info(f"★★★ デバッグモード: {debug_race_id} のみ処理 ★★★")
                logging.info(f"{'='*60}\n")
                target_races = (
                    # ★ キャッシュから取得
                    self.df_featured_all_cache_pl.filter(
                        pl.col("race_id") == debug_race_id
                    )
                    .select(["race_id", "日付", "場所", "芝・ダ", "距離", "馬場状態"])
                    .unique()  # レース情報のみ取得
                )
            else:
                logging.info(f"--- バックテスト開始 (対象: {start_date_str} 以降) ---")
                start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
                target_races = (
                    # ★ キャッシュから取得
                    self.df_featured_all_cache_pl.filter(pl.col("日付") >= start_date)
                    .select(["race_id", "日付", "場所", "芝・ダ", "距離", "馬場状態"])
                    .unique()  # レース情報のみ取得
                    .sort("日付")
                )

            logging.info(f"バックテスト対象レース数: {len(target_races)} 件")

            all_predictions_df_list = []  # 全レースの予測結果(DF)をためるリスト

            # 2. ★★★ 対象レースを1件ずつループ ★★★
            # ▼▼▼ 【★★★ tqdm プログレスバーを追加 ★★★】 ▼▼▼
            # デバッグモードでない場合のみ tqdm を適用
            if not debug_mode:
                # tqdm をインポート (関数の外、またはファイルの先頭で行うのが通常)
                from tqdm import tqdm

                race_iterator = tqdm(
                    target_races.rows(named=True),
                    total=len(target_races),
                    desc="バックテスト処理中",
                )
            else:
                race_iterator = target_races.rows(
                    named=True
                )  # デバッグ時は通常のイテレータ

            # enumerate を使わずにループ (tqdm がカウントしてくれる)
            for row in race_iterator:
                # ▲▲▲ 【★★★ tqdm プログレスバーここまで ★★★】 ▲▲▲

                race_id = row["race_id"]
                race_date = row["日付"]  # date型のはず
                # logging.info(f"--- 処理中 {i+1}/{len(target_races)}: {race_id} ({race_date}) ---") # ← tqdm が表示するのでコメントアウト

                try:
                    # 3. ★★★ データを「過去データ」と「当レースデータ」に分割 ★★★
                    # 過去データは特徴量計算済みキャッシュから
                    df_past_pl = self.df_featured_all_cache_pl.filter(
                        pl.col("日付") < race_date
                    )
                    # 当レースデータは「前処理のみ」の生データから (キャッシュ生成前の状態)
                    df_live_pl = self.df_historical_base_raw.filter(
                        pl.col("race_id") == race_id
                    )

                    if len(df_live_pl) == 0:
                        logging.warning(
                            f"  {race_id}: 当レースデータが生データに見つかりません。スキップ。"
                        )
                        continue

                    # 4. 過去データと予測対象レースを結合 (スキーマ合わせ含む)
                    past_schema = df_past_pl.schema
                    live_schema = df_live_pl.schema
                    past_cols_set = set(past_schema.keys())
                    live_cols_set = set(live_schema.keys())
                    all_cols_ordered = list(past_schema.keys()) + [
                        col for col in live_schema.keys() if col not in past_cols_set
                    ]
                    merged_schema = {**live_schema, **past_schema}
                    select_exprs_past = [
                        (
                            pl.col(col).cast(merged_schema[col], strict=False)
                            if col in past_cols_set
                            else pl.lit(None, dtype=merged_schema[col]).alias(col)
                        )
                        for col in all_cols_ordered
                    ]
                    select_exprs_live = [
                        (
                            pl.col(col).cast(merged_schema[col], strict=False)
                            if col in live_cols_set
                            else pl.lit(None, dtype=merged_schema[col]).alias(col)
                        )
                        for col in all_cols_ordered
                    ]
                    df_past_aligned = df_past_pl.select(select_exprs_past)
                    df_live_aligned = df_live_pl.select(select_exprs_live)
                    df_combined_pl = pl.concat(
                        [df_past_aligned, df_live_aligned], how="vertical"
                    ).sort(["馬名", "日付"])

                    # 5. 基準値の補完
                    track_type = str(row.get("芝・ダ", "芝"))
                    distance = str(row.get("距離", 0))
                    condition = str(row.get("馬場状態", "良"))
                    course_key = (track_type, distance, condition)
                    missing_baseline_cols = [
                        col
                        for col in self.baseline_columns
                        if col not in df_combined_pl.columns
                    ]
                    if missing_baseline_cols:
                        if course_key in self.baseline_dict:
                            baseline_values = self.baseline_dict[course_key]
                            for col in missing_baseline_cols:
                                if col in baseline_values:
                                    baseline_value = baseline_values[col]
                                    df_combined_pl = df_combined_pl.with_columns(
                                        pl.lit(baseline_value).alias(col)
                                    )
                                else:
                                    df_combined_pl = df_combined_pl.with_columns(
                                        pl.lit(0.0).alias(col)
                                    )
                                    # logging.warning(f"  基準値補完: {col} が辞書になく0で補完 (コース:{course_key})") # ログが多い場合はコメントアウト
                        else:
                            # logging.warning(f"  基準値補完: コース {course_key} の基準値が辞書になく0で補完") # ログが多い場合はコメントアウト
                            for col in missing_baseline_cols:
                                df_combined_pl = df_combined_pl.with_columns(
                                    pl.lit(0.0).alias(col)
                                )

                    # 6. ★★★ ループ内で特徴量を再計算 ★★★
                    # logging.info("  結合データに対して特徴量生成 (Point-in-Time) を実行中...") # tqdmがあるのでコメントアウト推奨
                    start_fe = datetime.now()
                    df_featured_combined_pl = feature_engineering(df_combined_pl)
                    end_fe = datetime.now()
                    # logging.info(f"    -> 特徴量生成完了 ({(end_fe - start_fe).total_seconds():.2f}秒)") # tqdmがあるのでコメントアウト推奨

                    # 7. ★★★ 当レースの「生の特徴量」を抽出 (CSV保存用) ★★★
                    df_featured_live_pl = df_featured_combined_pl.filter(
                        pl.col("race_id") == race_id
                    )

                    # ▼▼▼ 【★★★ 生の値確認コード (絞り込み) - デバッグ時のみ有効化 ★★★】 ▼▼▼
                    if debug_mode:  # デバッグモードの時だけ表示
                        try:
                            logging.info(
                                f"\n--- [DEBUG] スケーリング前の生の値確認 (Race: {row.get('race_id', '不明')}) ---"
                            )
                            features_to_check = [
                                "斤量体重比",
                                "前走との上り3F差",
                                "前走との平均速度差",
                                "追走指数",
                                "斤量",
                                "馬体重",
                                "上り3F",
                                "平均速度",
                                "キャリア平均_4角順位率",
                            ]
                            if len(df_featured_live_pl) > 0:
                                logging.info(
                                    f"  --- 1頭目 ({df_featured_live_pl[0, '馬名']}) の生の値 ---"
                                )
                                for feat in features_to_check:
                                    if feat in df_featured_live_pl.columns:
                                        raw_value = df_featured_live_pl[0, feat]
                                        logging.info(f"    {feat:<25}: {raw_value}")
                                    else:
                                        logging.warning(f"    {feat:<25}: カラム欠損 ⚠️")
                            else:
                                logging.warning("  df_featured_live_pl が空です。")
                        except Exception as e_raw:
                            logging.error(f"  生の値のデバッグ中にエラー: {e_raw}")
                        logging.info(
                            "--------------------------------------------------\n"
                        )
                    # ▲▲▲ 【★★★ 生の値確認コードここまで ★★★】 ▲▲▲

                    # 8. 前処理 & シーケンス生成 (スケーリング含む)
                    # logging.info("  前処理とシーケンス生成を実行中...") # tqdmがあるのでコメントアウト推奨
                    df_processed_all = self._preprocess_for_prediction(
                        df_featured_combined_pl
                    )

                    # 9. 予測対象行の抽出 (前処理後)
                    df_processed_live = df_processed_all.filter(
                        pl.col("race_id") == race_id
                    )

                    if len(df_processed_live) == 0:
                        logging.warning(f"  {race_id}: 前処理後データなし スキップ")
                        continue

                    # 10. テンソル抽出
                    X_seq_tensor, X_static_tensor = self._extract_tensors_from_df(
                        df_processed_live
                    )
                    if X_seq_tensor is None or X_static_tensor is None:
                        logging.error(f"  {race_id}: テンソル抽出失敗 スキップ")
                        continue

                    # ▼▼▼ 【★★★ テンソル値比較コード - デバッグ時のみ有効化 ★★★】 ▼▼▼
                    if debug_mode:
                        try:
                            logging.info(
                                f"\n--- [DEBUG] モデル入力直前のテンソル値 (Race: {row.get('race_id', '不明')}) ---"
                            )
                            logging.info(
                                f"--- [DEBUG] 静的特徴量のインデックス確認 ---"
                            )
                            if (
                                hasattr(self, "static_feature_names_ordered")
                                and len(self.static_feature_names_ordered) > 14
                            ):
                                logging.info(
                                    f"  インデックス[14] の特徴量名: {self.static_feature_names_ordered[14]}"
                                )
                                logging.info(
                                    f"  末尾3つの特徴量名: {self.static_feature_names_ordered[-3:]}"
                                )
                            else:
                                logging.warning(
                                    "  static_feature_names_ordered が見つからないか、短すぎます。"
                                )
                            logging.info(f"---------------------------------------")
                            if X_static_tensor.shape[0] > 0:
                                static_vals_np = X_static_tensor[0].cpu().numpy()
                                logging.info(
                                    f"  X_static (1頭目, 全次元): {static_vals_np}"
                                )
                                logging.info(
                                    f"  X_static Shape: {X_static_tensor.shape}"
                                )
                            if X_seq_tensor.shape[0] > 0:
                                seq_vals_np = X_seq_tensor[0].cpu().numpy()
                                logging.info(
                                    f"  X_seq (1頭目, 最新ステップ, 全次元): {seq_vals_np[-1, :]}"
                                )
                                logging.info(f"  X_seq Shape: {X_seq_tensor.shape}")
                        except Exception as e_tensor:
                            logging.error(
                                f"  テンソル値のデバッグ中にエラー: {e_tensor}"
                            )
                        logging.info(
                            "--------------------------------------------------\n"
                        )
                        logging.info(f"--- [DEBUG] モデルの状態確認 ---")
                        logging.info(f"  model.training: {self.model.training}")
                        logging.info(f"-----------------------------")
                    # ▲▲▲ 【★★★ テンソル値比較コードここまで ★★★】 ▲▲▲

                    # 11. 予測実行
                    with torch.no_grad():
                        y_pred_tensor = self.model(X_seq_tensor, X_static_tensor)
                    scores = y_pred_tensor.cpu().numpy().squeeze()
                    if scores.ndim == 0:
                        scores = [scores.item()]
                    else:
                        scores = scores.tolist()

                    # 12. ★★★ 結果の整形 (「生の特徴量」と予測スコアを結合) ★★★
                    if len(df_featured_live_pl) != len(scores):
                        logging.error(
                            f"  {race_id}: スコア数({len(scores)}) と 生データ行数({len(df_featured_live_pl)}) が不一致。スキップ。"
                        )
                        continue

                    df_race_result_pl = df_featured_live_pl.with_columns(
                        pl.Series("予測スコア", scores)
                    )
                    all_predictions_df_list.append(df_race_result_pl.to_pandas())

                except Exception as e:
                    logging.error(f"  {race_id} の処理中にエラー: {e}")
                    logging.error(traceback.format_exc())

            # 3. 全ループ終了後、結果を結合して保存
            if not all_predictions_df_list:
                logging.warning(
                    "予測結果が1件もありませんでした。CSVは作成されません。"
                )
                return

            logging.info("全レースの予測完了。結果を結合してCSVに保存中...")
            df_final_results = pd.concat(all_predictions_df_list, ignore_index=True)

            # ▼▼▼ 【★★★ RL (ベット戦略学習用) にカラムを絞り込む処理 ★★★】 ▼▼▼
            cols_for_rl = [
                # --- 識別情報 ---
                "race_id",
                "日付",
                "馬番",
                "馬名",
                # --- 状態 (State) - レース前の情報 ---
                "予測スコア",
                "単勝オッズ",
                # --- ↓ 必要に応じて状態として追加する特徴量 (レース前情報) ↓ ---
                "枠番",
                "斤量",
                "騎手",
                "調教師",
                "距離",
                "芝・ダ",
                "馬場状態",
                "回り",
                "頭数",
                "年齢",
                "性別",
                # --- 報酬 (Reward) 計算用 - レース後の結果 ---
                "着順",
                "単勝オッズ_実",
                "複勝圏内_実",
                "馬連",
                "馬単",
                "３連複",
                "３連単",
            ]
            cols_to_save = [
                col for col in cols_for_rl if col in df_final_results.columns
            ]
            required_cols = [
                "race_id",
                "馬番",
                "予測スコア",
                "単勝オッズ",
                "着順",
                "単勝オッズ_実",
                "複勝圏内_実",
            ]
            missing_required = [col for col in required_cols if col not in cols_to_save]
            if missing_required:
                logging.error(
                    f"致命的エラー: RL学習に必要なカラム {missing_required} がCSVに含まれていません！"
                )
                # return # 必要ならここで処理中断
            df_final_results = df_final_results[cols_to_save]
            logging.info(
                f"RL (ベット戦略学習用) にカラムを {len(cols_to_save)} 列に絞り込みました。"
            )
            # ▲▲▲ 【★★★ カラム絞り込みここまで ★★★】 ▲▲▲

            # ▼▼▼ 【★★★ 配当カラムのクリーンアップ処理を追加 ★★★】 ▼▼▼
            logging.info("配当情報を着順に基づいてクリーンアップ中...")
            try:
                if "着順" in df_final_results.columns:
                    df_final_results["着順"] = (
                        pd.to_numeric(df_final_results["着順"], errors="coerce")
                        .fillna(99)
                        .astype(int)
                    )
                    if "単勝オッズ_実" in df_final_results.columns:
                        df_final_results["単勝オッズ_実"] = df_final_results.apply(
                            lambda row: row["単勝オッズ_実"] if row["着順"] == 1 else 0,
                            axis=1,
                        )
                        logging.info("  - 単勝配当を1着以外は0にしました。")
                    if "複勝圏内_実" in df_final_results.columns:
                        df_final_results["複勝圏内_実"] = df_final_results.apply(
                            lambda row: (
                                row["複勝圏内_実"] if 1 <= row["着順"] <= 3 else 0
                            ),
                            axis=1,
                        )
                        logging.info("  - 複勝配当を3着以内以外は0にしました。")
                    combination_payout_cols = ["馬連", "馬単", "３連複", "３連単"]
                    for col in combination_payout_cols:
                        if col in df_final_results.columns:
                            df_final_results[col] = 0
                            logging.info(f"  - {col} 配当を全ての行で0にしました。")
                else:
                    logging.warning(
                        "着順カラムが見つからないため、配当情報のクリーンアップをスキップしました。"
                    )
            except Exception as e_cleanup:
                logging.error(f"配当情報のクリーンアップ中にエラー: {e_cleanup}")
                logging.error(traceback.format_exc())
            # ▲▲▲ 【★★★ クリーンアップ処理ここまで ★★★】 ▲▲▲

            output_dir = os.path.dirname(output_csv_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            df_final_results.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
            logging.info(
                f"--- バックテスト完了。結果を {output_csv_path} に保存しました ---"
            )
            logging.info(f"  - 保存された行数: {len(df_final_results)}")
            logging.info(f"  - 保存された列数: {len(df_final_results.columns)}")

        except Exception as e:
            logging.error(f"run_backtest で致命的なエラー: {e}")
            logging.error(traceback.format_exc())
            return


if __name__ == "__main__":
    # --- パス設定 ---
    DEBUG_MODE = False
    DEBUG_RACE_ID = "2025.9.28_中山_5"
    START_DATE_STR = "2024-01-01"

    HISTORICAL_DATA_PATH = "./2010_2025_data.csv"
    PEDIGREE_DATA_PATH = "./2005_2025_Pedigree.csv"

    PREPROCESSOR_DIR = "lstm_preprocessor_score"
    MODEL_DIR = "lstm_models_score"

    OUTPUT_CSV_PATH = "./data/processed/backtest_predictions_raw_features_pit.csv"  # ★ Point-in-Time を示すファイル名

    logging.info(f"--- バックテスト ({os.path.basename(__file__)}) 開始 ---")
    if DEBUG_MODE:
        logging.info(f"★★★ デバッグモード: {DEBUG_RACE_ID} のみ処理 ★★★")
    else:
        logging.info(f"★★★ フルバックテストモード: {START_DATE_STR} 以降を処理 ★★★")
        logging.warning("フルバックテストは非常に時間がかかります...")

    try:
        backtester = Backtester(
            historical_data_path=HISTORICAL_DATA_PATH,
            pedigree_data_path=PEDIGREE_DATA_PATH,
            preprocessor_base_dir=PREPROCESSOR_DIR,
            model_base_dir=MODEL_DIR,
        )

        backtester.run_backtest(
            output_csv_path=OUTPUT_CSV_PATH,
            start_date_str=START_DATE_STR,
            debug_mode=DEBUG_MODE,
            debug_race_id=DEBUG_RACE_ID,
        )

    except FileNotFoundError as e:
        logging.error(f"ファイルが見つかりません: {e}")
    except Exception as e:
        logging.error(f"予期せぬエラーが発生しました: {e}")
        logging.error(traceback.format_exc())

    logging.info(f"--- バックテスト ({os.path.basename(__file__)}) 終了 ---")
