# predict_logic.py
import polars as pl
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

# --- 必要なモジュールとクラス定義をインポートまたはコピー ---
# (本来は keiba_analysis.py, features_engineered.py, scraper.py からインポート)


# ▼▼▼ [keiba_analysis.py からコピー] DCN_LSTM_Hybrid クラス定義 ▼▼▼
class DCN_LSTM_Hybrid(BaseModel):
    def __init__(
        self,
        static_linear_feature_columns,
        static_dnn_feature_columns,
        sequence_feature_columns,
        static_feature_names,
        lstm_hidden_size=64,
        lstm_layers=1,
        lstm_dropout=0.2,
        cross_num=2,
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

        super(DCN_LSTM_Hybrid, self).__init__(
            linear_feature_columns=static_linear_feature_columns,
            dnn_feature_columns=static_dnn_feature_columns,
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
                        ids = X_seq_tensor[:, :, current_seq_col_idx].long()
                        emb = embedding_layer(ids)
                        seq_sparse_embedding_list.append(emb)
                    except Exception:
                        pass
                current_seq_col_idx += 1
            elif isinstance(fc, DenseFeat):
                try:
                    val = X_seq_tensor[:, :, current_seq_col_idx].unsqueeze(-1)
                    val = torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
                    seq_dense_value_list.append(val)
                except Exception:
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
                calculated_input_size = sum(t.shape[-1] for t in lstm_input_tensors)
                logging.error(
                    f"LSTM Error: Expected {self.lstm.input_size}, Got {calculated_input_size}, Err: {e}"
                )
                lstm_output_features = torch.zeros(
                    (batch_size, self.lstm.hidden_size)
                ).to(self.device)

        # --- Prepare DCN Input (Static Features) ---
        try:
            linear_sparse_embedding_list, linear_dense_value_list = (
                self.input_from_feature_columns(
                    X_static_tensor,
                    self.static_linear_feature_columns,
                    self.embedding_dict,
                    self.feature_index,
                )
            )
            dnn_sparse_embedding_list, dnn_dense_value_list = (
                self.input_from_feature_columns(
                    X_static_tensor,
                    self.static_dnn_feature_columns,
                    self.embedding_dict,
                    self.feature_index,
                )
            )
        except Exception as e:
            logging.error(f"Error processing static features: {e}")
            return torch.zeros((batch_size, 1)).to(self.device)

        # CrossNet Input
        if not linear_sparse_embedding_list and not linear_dense_value_list:
            cross_input = torch.zeros((batch_size, self.cross_input_dim)).to(
                self.device
            )
        else:
            cross_input = combined_dnn_input(
                linear_sparse_embedding_list, linear_dense_value_list
            )

        # DNN Input
        if not dnn_sparse_embedding_list and not dnn_dense_value_list:
            dnn_input_combined = lstm_output_features
        else:
            dnn_input_static = combined_dnn_input(
                dnn_sparse_embedding_list, dnn_dense_value_list
            )
            if (
                lstm_output_features is None
                or lstm_output_features.shape[0] != batch_size
            ):
                zero_lstm_padding = torch.zeros((batch_size, self.lstm.hidden_size)).to(
                    self.device
                )
                dnn_input_combined = torch.cat(
                    [dnn_input_static, zero_lstm_padding], dim=-1
                )
            else:
                dnn_input_combined = torch.cat(
                    [dnn_input_static, lstm_output_features], dim=-1
                )

        # Check DNN input dimension
        expected_dnn_dim = self.dnn_input_dim
        actual_dnn_dim = dnn_input_combined.shape[1]
        if actual_dnn_dim != expected_dnn_dim:
            logging.error(
                f"Error: DNN input dim mismatch: Expected={expected_dnn_dim}, Actual={actual_dnn_dim}"
            )
            dnn_output = torch.zeros((batch_size, self.dnn_hidden_units[-1])).to(
                self.device
            )
        else:
            dnn_output = self.dnn(dnn_input_combined)

        dnn_logit = self.dnn_linear(dnn_output)

        # CrossNet Layer
        expected_cross_dim = self.cross_input_dim
        actual_cross_dim = cross_input.shape[1]
        if actual_cross_dim > 0 and actual_cross_dim == expected_cross_dim:
            cross_features = self.crossnet(cross_input)
            cross_out = self.cross_linear(cross_features)
        else:
            if actual_cross_dim > 0 and actual_cross_dim != expected_cross_dim:
                logging.error(
                    f"Error: CrossNet input dim mismatch: Expected={expected_cross_dim}, Actual={actual_cross_dim}"
                )
            cross_out = torch.zeros_like(dnn_logit)

        # Final prediction
        final_logit = cross_out + dnn_logit
        y_pred = self.out(final_logit)
        y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
        return y_pred


# ▲▲▲ [keiba_analysis.py からコピー] ここまで ▲▲▲


# ▼▼▼ [preprocess_sequences.py からコピー] apply_mappings_pl ヘルパー関数 ▼▼▼
def apply_mappings_pl(df, col, mapping):
    """Polars DataFrameにマッピングを適用"""
    mapping_filled = {str(k): int(v) for k, v in mapping.items()}
    return df.with_columns(
        pl.col(col)
        .cast(pl.Utf8)
        .map_elements(
            lambda x: mapping_filled.get(x, 0),
            return_dtype=pl.Int32,
        )
        .alias(col)
    )


# ▲▲▲ [preprocess_sequences.py からコピー] ここまで ▲▲▲


# ▼▼▼ [features_engineered.py からインポート想定] ▼▼▼
try:
    from features_engineered import preprocess_data, feature_engineering, normalize_nkfc

    logging.info("features_engineered.py から関数をインポートしました。")
except ImportError as e:
    logging.error(f"features_engineered.py のインポートに失敗: {e}")

    def preprocess_data(race_path, pedigree_path):
        raise ImportError("features_engineered.py が見つかりません。")

    def feature_engineering(df_polars):
        raise ImportError("features_engineered.py が見つかりません。")

    def normalize_nkfc(text):
        raise ImportError("features_engineered.py が見つかりません。")


# ▲▲▲ [features_engineered.py からインポート想定] ここまで ▲▲▲


# ▼▼▼【★修正★】キャリア補正関数を (features_engineered.py L977-L990 より) コピー ▼▼▼
def apply_career_reset(df_polars):
    """4年以上(1460日)の休養がある同名馬のキャリアをリセットする。"""
    logging.info("--- 同名馬・長期休養馬のキャリアを補正中... ---")
    gap_threshold_days = 1460
    df_polars = df_polars.sort(["馬名", "日付"])
    df_polars = (
        df_polars.with_columns(
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
            pl.col("_キャリア区切り").cum_sum().over("馬名").alias("_キャリア区間ID")
        )
        .with_columns(
            pl.col("_キャリア区間ID").max().over("馬名").alias("_最新キャリア区間ID")
        )
    )
    df_polars = df_polars.filter(
        pl.col("_キャリア区間ID") == pl.col("_最新キャリア区間ID")
    )
    df_polars = df_polars.drop(
        ["_前走からの日数", "_キャリア区切り", "_キャリア区間ID", "_最新キャリア区間ID"]
    )
    logging.info("--- キャリア補正完了 ---")
    return df_polars


# ▲▲▲【★修正★】ここまで ▲▲▲

# ▼▼▼ [scraper.py からインポート想定] ▼▼▼
try:
    from scraper import extract_race_info_from_url, parse_race_info

    logging.info("scraper.py から関数をインポートしました。")
except ImportError as e:
    logging.error(f"scraper.py のインポートに失敗: {e}")
# ▲▲▲ [scraper.py からインポート想定] ここまで ▲▲▲

# (カテゴリカル変数（文字列）を含まず、率（数値）のみが含まれているリスト)
FEATURE_COLS_LGBM = [
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


class Predictor:
    def __init__(
        self,
        historical_data_path,
        pedigree_data_path,
        preprocessor_base_dir,
        model_base_dir,
    ):
        logging.info("--- 予測器 (Predictor) の初期化を開始 ---")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"使用デバイス: {self.device}")

        self.preprocessor_dir = preprocessor_base_dir
        self.model_dir = model_base_dir
        self.lgbm_model_dir = "./lgbm_models"

        # --- 1. 設定 (N_STEPS) をロード ---
        self.n_steps = 5
        self.target_col = "rank_score"

        # --- 2. 履歴データと血統データをロード & 特徴量生成 (★修正★) ---
        try:
            logging.info(f"履歴データ {historical_data_path} を読み込み中...")
            # preprocess_data を実行
            df_historical_base = preprocess_data(
                historical_data_path, pedigree_data_path
            )
            df_historical_base = apply_career_reset(df_historical_base)
            logging.info("履歴データの前処理完了。")

            logging.info("  - (互換性) 騎手・調教師名を3文字に切り捨てます。")
            for col in ["騎手", "調教師"]:
                if col in df_historical_base.columns:
                    df_historical_base = df_historical_base.with_columns(
                        pl.col(col).str.slice(0, 3).alias(col)
                    )

            # ▼▼▼【ここから修正】▼▼▼
            logging.info(
                "全履歴データに対して特徴量生成 (feature_engineering) を実行中..."
            )
            # ★ここで feature_engineering を実行し、結果を保持する★
            self.df_featured_all_pl = feature_engineering(df_historical_base)
            logging.info(f"特徴量生成完了。Shape: {self.df_featured_all_pl.shape}")

            # ★★★ 追加: 基準値カラムを抽出して保存 ★★★
            self.baseline_columns = [
                col for col in self.df_featured_all_pl.columns if "_基準_" in col
            ]
            logging.info(f"保存する基準値カラム数: {len(self.baseline_columns)}")

            # ★★★ 追加: コース×馬場状態ごとの基準値を辞書化 ★★★
            self.baseline_dict = {}
            for col in self.baseline_columns:
                # カラム名の例: "上り3F_基準_芝_1600_良"
                parts = col.split("_")
                if len(parts) >= 5:
                    metric = parts[0]  # 例: "上り3F"
                    track_type = parts[2]  # 例: "芝"
                    distance = parts[3]  # 例: "1600"
                    condition = parts[4]  # 例: "良"

                    key = (track_type, distance, condition)

                    # 該当コースの基準値を取得(最初の非null値)
                    baseline_value = (
                        self.df_featured_all_pl.filter(
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
            # ▲▲▲【ここまで修正】▲▲▲

            # (血統データの個別保持は継続)
            self.df_ped_pl = pl.read_csv(
                pedigree_data_path, encoding="cp932", infer_schema_length=10000
            )
            for col in ["馬名", "種牡馬", "母父馬"]:
                if col in self.df_ped_pl.columns:
                    self.df_ped_pl = self.df_ped_pl.with_columns(
                        pl.col(col)
                        .cast(pl.Utf8)
                        .map_elements(normalize_nkfc, return_dtype=pl.Utf8)
                        .str.strip_chars()
                        .str.replace_all(r"\s+", "")
                        .str.replace_all(r"[▲▽△☆]", "")
                        .alias(col)
                    )
            if "母父名" in self.df_ped_pl.columns:
                self.df_ped_pl = self.df_ped_pl.rename({"母父名": "母父馬"})
            self.ped_cols_to_merge = ["馬名", "種牡馬", "母父馬"]
            self.df_ped_pl = self.df_ped_pl.unique(subset=["馬名"], keep="last")

        except Exception as e:
            logging.error(f"履歴データ読み込みまたは特徴量生成に失敗: {e}")
            raise e

        # --- 3. 前処理オブジェクトをロード ---
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
            all_feature_columns_pkl = joblib.load(
                os.path.join(model_base_dir, "feature_columns.pkl")
            )
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
            self.categorical_features = list(self.all_mappings.keys())
            self.numerical_features_static_scaled = (
                self.scaler.feature_names_in_.tolist()
            )
            sequence_numerical_names = [
                name
                for name in sequence_name_set
                if name not in self.all_mappings and name != self.target_col
            ]
            self.numerical_features_all = list(
                set(self.numerical_features_static_scaled + sequence_numerical_names)
            )
            logging.info("前処理オブジェクトのロード完了")
        except Exception as e:
            logging.error(f"前処理オブジェクトのロードに失敗: {e}")
            raise e

        # --- 4A. モデルをロード ---
        try:
            logging.info(f"{model_base_dir} からモデルをロード中...")
            model_path = os.path.join(model_base_dir, "lstm_hybrid_model.pth")
            self.model = DCN_LSTM_Hybrid(
                static_linear_feature_columns=self.static_fcs_sorted,
                static_dnn_feature_columns=self.static_fcs_sorted,
                sequence_feature_columns=self.sequence_fcs_sorted,
                static_feature_names=self.static_feature_names_ordered,
                lstm_hidden_size=64,
                lstm_layers=1,
                lstm_dropout=0.2,
                cross_num=3,
                dnn_hidden_units=(256, 128),
                l2_reg_linear=1e-5,
                l2_reg_embedding=1e-5,
                l2_reg_cross=1e-5,
                l2_reg_dnn=1e-5,
                dnn_dropout=0.3,
                dnn_use_bn=True,
                task="regression",
                device=self.device,
            )
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            logging.info("モデルのロードと評価モードへの設定完了")
        except Exception as e:
            logging.error(f"モデルのロードに失敗: {e}")
            raise e

        # ▼▼▼【ここから追加】▼▼▼
        # --- 4B. モデルをロード (LightGBM) ---
        try:
            logging.info(f"{self.lgbm_model_dir} からLGBMモデルをロード中...")
            ranker_path = os.path.join(self.lgbm_model_dir, "lgbm_ranker.pkl")
            classifier_path = os.path.join(self.lgbm_model_dir, "lgbm_classifier.pkl")

            self.lgbm_ranker = joblib.load(ranker_path)
            self.lgbm_classifier = joblib.load(classifier_path)

            logging.info("LGBM二重モデル（ランカー、分類器）のロード完了")

            # LGBMが使う特徴量リストをグローバルから取得
            self.feature_cols_lgbm = FEATURE_COLS_LGBM
            # (カテゴリカル変数は使わない前提)
            self.categorical_cols_lgbm = []

        except Exception as e:
            logging.error(f"LGBMモデルのロードに失敗: {e}")
            raise e
        # ▲▲▲【ここまで追加】▲▲▲

        # --- 5. (オプション) SHAP重要度をロード ---
        try:
            shap_path = os.path.join(model_base_dir, "shap_importance.json")
            if os.path.exists(shap_path):
                with open(shap_path, "r", encoding="utf-8") as f:
                    self.shap_importance = json.load(f)
                logging.info("SHAP重要度データをロードしました。")
            else:
                self.shap_importance = None
                logging.warning("SHAP重要度データが見つかりません。")
        except Exception as e:
            logging.warning(f"SHAP重要度のロードに失敗: {e}")
            self.shap_importance = None

        logging.info("--- 予測器の初期化が完了 ---")

    def _prepare_live_data(self, shutuba_csv_path, race_info, form_race_date):
        """
        スクレイピングした出馬表とレース情報から、
        features_engineered.py が処理できる形式の Polars DataFrame を作成する。
        """
        try:
            df_shutuba_pd = pd.read_csv(shutuba_csv_path)
            if "厩舎" in df_shutuba_pd.columns:
                df_shutuba_pd["所属"] = (
                    df_shutuba_pd["厩舎"]
                    .astype(str)
                    .str.extract(r"(美浦|栗東)")
                    .fillna("他")
                )
                df_shutuba_pd["調教師"] = (
                    df_shutuba_pd["厩舎"]
                    .astype(str)
                    .str.replace(r"^(美浦|栗東)・?", "", regex=True)
                )
            else:
                df_shutuba_pd["所属"] = "他"
            df_shutuba_pd = df_shutuba_pd.rename(
                columns={
                    "馬 番": "馬番",
                    "枠 番": "枠番",
                    "枠": "枠番",
                    "性齢": "性齢",
                    "斤 量": "斤量",
                    "騎 手": "騎手",
                    "馬 体重": "馬体重",
                    "オッズ": "単勝オッズ",
                }
            )
            for col in ["騎手", "馬名", "調教師"]:
                if col in df_shutuba_pd.columns:
                    df_shutuba_pd[col] = (
                        df_shutuba_pd[col]
                        .astype(str)
                        .map(normalize_nkfc)
                        .str.strip()
                        .str.replace(r"\s+", "", regex=True)
                        .str.replace(r"[▲▽△☆]", "", regex=True)
                    )

            for col in ["騎手", "調教師"]:
                if col in df_shutuba_pd.columns:
                    df_shutuba_pd[col] = df_shutuba_pd[col].str[:3]

            df_shutuba_pl = pl.from_pandas(df_shutuba_pd)
            if "性齢" in df_shutuba_pl.columns:
                df_shutuba_pl = df_shutuba_pl.with_columns(
                    [
                        pl.col("性齢").str.slice(0, 1).alias("性別"),
                        pl.col("性齢")
                        .str.slice(1, None)
                        .cast(pl.Float64, strict=False)
                        .alias("年齢"),
                    ]
                )
            else:
                df_shutuba_pl = df_shutuba_pl.with_columns(
                    [pl.lit(None).alias("性別"), pl.lit(None).alias("年齢")]
                )
            if "馬体重" not in df_shutuba_pl.columns:
                df_shutuba_pl = df_shutuba_pl.with_columns(
                    pl.lit(None).cast(pl.Float64).alias("馬体重")
                )
            race_date = pd.to_datetime(form_race_date)
            race_location = race_info.get("race_location", "不明")
            race_number = race_info.get("race_number", 0)
            date_s = race_date.strftime("%Y.%m.%d")
            race_id = f"{date_s}_{race_location}_{race_number}"
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
            race_class_str = race_info.get("race_class", "オープン")
            if "G1" in race_class_str or "GI" in race_class_str:
                race_class_cat = "G1"
            elif "G2" in race_class_str or "GII" in race_class_str:
                race_class_cat = "G2"
            elif "G3" in race_class_str or "GIII" in race_class_str:
                race_class_cat = "G3"
            elif "L" in race_class_str or "リステッド" in race_class_str:
                race_class_cat = "リステッド"
            elif "3勝" in race_class_str or "3勝" in race_class_str:
                race_class_cat = "3勝クラス"
            elif "2勝" in race_class_str or "2勝" in race_class_str:
                race_class_cat = "2勝クラス"
            elif "1勝" in race_class_str or "1勝" in race_class_str:
                race_class_cat = "1勝クラス"
            elif "未勝利" in race_class_str:
                race_class_cat = "未勝利"
            elif "新馬" in race_class_str:
                race_class_cat = "新馬"
            else:
                race_class_cat = "オープン"
            race_level = class_map.get(race_class_cat, 5)
            track_type_map = {"芝": "芝", "ダ": "ダ", "障": "障"}
            track_type = track_type_map.get(race_info.get("race_track_type"), "芝")
            distance = race_info.get("race_distance", 0)
            df_shutuba_pl = df_shutuba_pl.with_columns(
                [
                    pl.lit(race_date).alias("日付"),
                    pl.lit(date_s).alias("日付S"),
                    pl.lit(race_id).alias("race_id"),
                    pl.lit(race_info.get("race_name", "不明")).alias("レース名"),
                    pl.lit(race_location).alias("場所"),
                    pl.lit(race_number).alias("R"),
                    pl.lit(race_info.get("race_track_condition", "良")).alias(
                        "馬場状態"
                    ),
                    pl.lit(distance).alias("距離"),
                    pl.lit(track_type).alias("芝・ダ"),
                    pl.lit(mawari_map.get(race_location, "右")).alias("回り"),
                    pl.lit(race_class_cat).alias("クラスカテゴリ"),
                    pl.lit(race_level).cast(pl.Int64).alias("クラスレベル"),
                    (pl.lit(track_type) + "_" + pl.lit(distance).cast(pl.Utf8)).alias(
                        "コース距離"
                    ),
                ]
            )
            df_shutuba_pl = df_shutuba_pl.with_columns(
                pl.col("馬名").n_unique().over("race_id").alias("頭数")
            )
            df_shutuba_pl = df_shutuba_pl.join(
                self.df_ped_pl.select(self.ped_cols_to_merge), on="馬名", how="left"
            )

            df_shutuba_pl = df_shutuba_pl.with_columns(
                pl.when(
                    # 馬体重が Null でなく、0より大きく、斤量も Null でない場合
                    pl.col("馬体重").is_not_null()
                    & (pl.col("馬体重") > 0)
                    & pl.col("斤量").is_not_null()
                )
                .then(pl.col("斤量") / pl.col("馬体重") * 100)  # パーセントで計算
                .otherwise(0.0)  # 計算できない場合は 0.0 とする
                .alias("斤量体重比")
            )
            logging.info("  - 斤量体重比 を計算して追加/上書きしました。")

            dummy_cols_float = [
                "着順",
                "人気",
                "PCI",
                "上り3F",
                "Ave-3F",
                "走破タイム",
                "平均速度",
                "着差",
                "4角",
                "3角",
                "単勝オッズ_実",
                "複勝圏内_実",
                "rank_score",
                "単勝",
                "複勝圏内",
            ]
            for col in dummy_cols_float:
                if col not in df_shutuba_pl.columns:
                    df_shutuba_pl = df_shutuba_pl.with_columns(
                        pl.lit(None).cast(pl.Float64).alias(col)
                    )
            if "脚質" not in df_shutuba_pl.columns:
                df_shutuba_pl = df_shutuba_pl.with_columns(pl.lit("不明").alias("脚質"))
            if (
                "単勝オッズ" not in df_shutuba_pl.columns
                and "オッズ" in df_shutuba_pl.columns
            ):
                df_shutuba_pl = df_shutuba_pl.rename({"オッズ": "単勝オッズ"})
            elif "単勝オッズ" not in df_shutuba_pl.columns:
                df_shutuba_pl = df_shutuba_pl.with_columns(
                    pl.lit(None).cast(pl.Float64).alias("単勝オッズ")
                )
            cols_to_process = ["単勝オッズ", "斤量", "馬体重", "枠番"]
            for col in cols_to_process:
                if col in df_shutuba_pl.columns:
                    df_shutuba_pl = df_shutuba_pl.with_columns(
                        pl.col(col)
                        .cast(pl.Utf8)
                        .map_elements(normalize_nkfc, return_dtype=pl.Utf8)
                        .str.extract(r"(\d+\.?\d*)", 0)
                        .cast(pl.Float64, strict=False)
                        .alias(col)
                    )
            return df_shutuba_pl
        except Exception as e:
            logging.error(f"_prepare_live_data でエラー: {e}")
            logging.error(traceback.format_exc())
            return None

    def _get_past_performances(self, horse_names, race_date):
        """
        馬名リストとレース日付に基づき、保持している特徴量生成済みデータ
        (self.df_featured_all_pl) から過去戦績を取得する。(★修正★)
        """
        if not horse_names:
            return pl.DataFrame()

        # Polars でフィルタリング (対象を self.df_featured_all_pl に変更)
        df_past = self.df_featured_all_pl.filter(
            (pl.col("馬名").is_in(horse_names)) & (pl.col("日付") < race_date)
        )

        # ★★★ デバッグ: 各馬の過去データ件数を確認 ★★★
        logging.info("========== 過去データ取得結果 ==========")
        for horse in horse_names:
            count = df_past.filter(pl.col("馬名") == horse).shape[0]
            if count == 0:
                logging.warning(
                    f"  ⚠️ {horse}: 過去データなし (新馬または履歴データに未登録)"
                )
            else:
                logging.info(f"  ✓ {horse}: {count}件の過去データ")
        logging.info("=" * 40)

        return df_past

    def _preprocess_for_prediction(self, df_combined_pl):
        """
        結合されたDataFrame (過去データ + 予測対象レース) に
        preprocess_sequences.py と同じ前処理とシーケンス生成を適用する。
        """
        # --- 1. 前処理の適用 (Polars) ---
        df_pl = df_combined_pl
        for col in self.categorical_features:
            if col in df_pl.columns:
                mapping = self.all_mappings.get(col)
                if mapping:
                    df_pl = apply_mappings_pl(df_pl, col, mapping)
        if (
            "単勝オッズ" in df_pl.columns
            and "単勝オッズ" in self.numerical_features_static_scaled
        ):
            df_pl = df_pl.with_columns(
                pl.col("単勝オッズ")
                .fill_nan(0.0)
                .fill_null(0.0)
                .log1p()
                .alias("単勝オッズ")
            )
        static_num_cols_in_df = [
            col for col in self.numerical_features_static_scaled if col in df_pl.columns
        ]
        if static_num_cols_in_df:
            exprs_to_fill = [
                pl.col(col).fill_nan(0.0).fill_null(0.0).alias(col)
                for col in static_num_cols_in_df
            ]
            df_pl = df_pl.with_columns(exprs_to_fill)
            np_data_to_scale = df_pl.select(static_num_cols_in_df).to_numpy()
            np_data_to_scale = np.nan_to_num(
                np_data_to_scale, nan=0.0, posinf=0.0, neginf=0.0
            )
            scaled_data = self.scaler.transform(np_data_to_scale)
            df_scaled = pl.DataFrame(scaled_data, schema=static_num_cols_in_df)
            df_pl = df_pl.drop(static_num_cols_in_df).hstack(df_scaled)
        other_num_cols = [
            col
            for col in self.numerical_features_all
            if col not in self.numerical_features_static_scaled and col in df_pl.columns
        ]
        if other_num_cols:
            df_pl = df_pl.with_columns(
                [
                    pl.col(col).fill_nan(0.0).fill_null(0.0).alias(col)
                    for col in other_num_cols
                ]
            )
        if self.target_col in df_pl.columns:
            df_pl = df_pl.with_columns(
                pl.col(self.target_col)
                .fill_nan(0.0)
                .fill_null(0.0)
                .alias(self.target_col)
            )
        # --- 2. シーケンスデータの生成 (Polars Window Functions) ---
        df_pl = df_pl.with_columns(
            pl.int_range(0, pl.len()).over("馬名").alias("_horse_race_idx")
        )
        lag_exprs = []
        for i in range(1, self.n_steps + 1):
            for col_name in self.sequence_feature_names_ordered:
                if col_name in df_pl.columns:
                    lag_exprs.append(
                        pl.col(col_name)
                        .shift(i)
                        .over("馬名")
                        .alias(f"{col_name}_lag_{i}")
                    )
        df_pl = df_pl.with_columns(lag_exprs)
        return df_pl

    def _extract_tensors_from_df(self, df_processed_live):
        """
        前処理済みの DataFrame (予測対象レースのみ) から
        X_static と X_seq のテンソルを抽出する。
        """
        n_rows = len(df_processed_live)
        if n_rows == 0:
            return None, None
        final_static_cols_to_select = [
            col
            for col in self.static_feature_names_ordered
            if col in df_processed_live.columns
        ]
        if len(final_static_cols_to_select) != len(self.static_feature_names_ordered):
            logging.warning("モデルが期待する静的特徴量の一部がDFに存在しません。")
            missing_cols = set(self.static_feature_names_ordered) - set(
                final_static_cols_to_select
            )
            logging.warning(f"不足している静的特徴量 (0で補完): {missing_cols}")
            dummy_exprs = [pl.lit(0.0).alias(col) for col in missing_cols]
            df_processed_live = df_processed_live.with_columns(dummy_exprs)
            final_static_cols_to_select = self.static_feature_names_ordered
        X_static_np = df_processed_live.select(final_static_cols_to_select).to_numpy()
        sequence_data_np = []
        for col_name in self.sequence_feature_names_ordered:
            lag_cols = [f"{col_name}_lag_{i}" for i in range(self.n_steps, 0, -1)]
            valid_lag_cols = [
                col for col in lag_cols if col in df_processed_live.columns
            ]
            fill_value_expr = None
            if (
                (col_name in self.all_mappings)
                or (col_name.endswith("フラグ"))
                or (col_name.endswith("_id"))
            ):
                fill_value_expr = pl.lit(0)
            else:
                if col_name in df_processed_live.columns:
                    fill_value_expr = pl.col(col_name)
                else:
                    fill_value_expr = pl.lit(0.0)
            select_exprs = [
                pl.col(c).fill_null(fill_value_expr) for c in valid_lag_cols
            ]
            if not valid_lag_cols:
                seq_np = np.zeros((n_rows, self.n_steps))
            else:
                seq_np = df_processed_live.select(select_exprs).to_numpy()
                if len(valid_lag_cols) < self.n_steps:
                    num_missing = self.n_steps - len(valid_lag_cols)
                    padding_values_np = df_processed_live.select(
                        fill_value_expr.alias("fill_val")
                    ).to_numpy()
                    padding_values_np = np.tile(padding_values_np, (1, num_missing))
                    seq_np = np.hstack([padding_values_np, seq_np])
            sequence_data_np.append(seq_np[:, :, np.newaxis])
        if not sequence_data_np:
            X_seq_np = np.empty((n_rows, self.n_steps, 0))
        else:
            X_seq_np = np.concatenate(sequence_data_np, axis=2)
        X_seq_tensor = torch.tensor(X_seq_np, dtype=torch.float32).to(self.device)
        X_static_tensor = torch.tensor(X_static_np, dtype=torch.float32).to(self.device)
        return X_seq_tensor, X_static_tensor

    def _get_factors_and_history(self, df_processed_live_pd, df_past_pd):
        """
        予測後のDFと過去データDF(Pandas)から、
        詳細表示用のファクター(★実際の生成特徴量★)と過去戦績を抽出する。
        """
        factors_dict = {}
        top_feature_names = []
        if self.shap_importance:
            try:
                sorted_importance = sorted(
                    self.shap_importance.items(), key=lambda item: item[1], reverse=True
                )
                top_feature_names = [name for name, imp in sorted_importance[:15]]
            except Exception as e:
                logging.warning(f"SHAP重要度リストのソートに失敗: {e}")
        for idx, row in df_processed_live_pd.iterrows():
            horse_name = row["馬名"]
            horse_factors = []
            if top_feature_names:
                for col_name in top_feature_names:
                    if col_name in row:
                        actual_value = row[col_name]
                        if isinstance(actual_value, (float, np.floating)):
                            actual_value_formatted = f"{actual_value:.3f}"
                        else:
                            actual_value_formatted = str(actual_value)
                        horse_factors.append((col_name, actual_value_formatted))
                    else:
                        horse_factors.append((col_name, "N/A"))
            else:
                horse_factors.append(("Error", "SHAP.json が見つかりません"))
            factors_dict[horse_name] = horse_factors
        history_dict = {}
        past_grouped = df_past_pd.groupby("馬名")
        cols_to_show = [
            "日付",
            "レース名",
            "場所",
            "距離",
            "着順",
            "上り3F",
            "単勝オッズ",
        ]
        for horse_name in df_processed_live_pd["馬名"].unique():
            if horse_name in past_grouped.groups:
                df_horse_past = past_grouped.get_group(horse_name).sort_values(
                    "日付", ascending=False
                )
                df_horse_history = df_horse_past[cols_to_show].head(self.n_steps)
                history_dict[horse_name] = df_horse_history.to_dict("records")
            else:
                history_dict[horse_name] = []
        return factors_dict, history_dict

    def _calculate_kelly_bet(self, df_live_results_pd):
        """
        予測結果のPandas DFを受け取り、ケリー基準で賭け金を計算して返す。
        (103.99% を達成した train_lightgbm_combined.py のロジックを移植)
        """

        # --- ケリー戦略のパラメータ (103.99% 達成時のもの) ---
        KELLY_FRACTION = 0.05
        BASE_UNIT = 100000.0
        MAX_BET_PER_RACE = 500.0  # 103.99% 達成時の上限値

        # 戦略フィルター
        MIN_ODDS = 1.0  # 最小オッズ
        MAX_ODDS = 10.0  # 最大オッズ (B < 10.0)
        MIN_EXPECTED_VALUE = 0.0  # 期待値 (E > 0)

        rank_col = "predict_rank_score"
        proba_col = "predict_win_proba"
        odds_col = "単勝オッズ"

        bet_amounts = {}  # 馬名: 賭け金 の辞書

        # --- ロジック開始 ---

        # 1. ランキング上位3位の馬を抽出
        top_n_horses = df_live_results_pd.nlargest(3, rank_col)

        # 2. 上位3頭を1頭ずつチェック
        for target_horse_idx, target_horse in top_n_horses.iterrows():
            horse_name = target_horse["馬名"]
            P = target_horse[proba_col]  # 予測勝率
            B = target_horse[odds_col]  # 単勝オッズ

            if pd.isna(P) or pd.isna(B) or B < MIN_ODDS:
                bet_amounts[horse_name] = 0
                continue

            # 3. 戦略フィルター (オッズ)
            if B >= MAX_ODDS:
                bet_amounts[horse_name] = 0
                continue

            # 4. 期待値 E の計算
            E = (P * B) - 1

            # 5. 戦略フィルター (期待値)
            if E <= MIN_EXPECTED_VALUE:
                bet_amounts[horse_name] = 0
                continue

            # 6. ケリー分数 F の計算
            B_minus_1 = B - 1
            if B_minus_1 <= 0.001:
                bet_amounts[horse_name] = 0
                continue

            F = (P * B_minus_1 - (1 - P)) / B_minus_1

            if F <= 0:
                bet_amounts[horse_name] = 0
                continue

            # 7. 賭け金（Bet Size）の決定
            calculated_bet = F * KELLY_FRACTION * BASE_UNIT
            purchase_amount = np.floor(calculated_bet / 100.0) * 100
            purchase_amount = min(purchase_amount, MAX_BET_PER_RACE)

            if purchase_amount < 100:
                bet_amounts[horse_name] = 0
                continue

            bet_amounts[horse_name] = int(purchase_amount)

        # ランク外の馬の賭け金を0に設定
        all_horse_names = df_live_results_pd["馬名"].unique()
        for horse in all_horse_names:
            if horse not in bet_amounts:
                bet_amounts[horse] = 0

        return bet_amounts

    # ▲▲▲【ここまで追加】▲▲▲

    def run_prediction(self, url, shutuba_csv_path, form_race_date, race_info):
        """
        予測実行のメインロジック (★修正★)
        """

        print("=" * 80)
        print("【予測開始】run_prediction が呼び出されました")
        print(f"  - URL: {url}")
        print(f"  - 日付: {form_race_date}")
        print(f"  - レース情報: {race_info}")
        print("=" * 80)

        try:
            # --- 1. レース情報取得 ---
            logging.info(f"レース情報 (app.py から受信): {race_info}")

            # --- 2. 予測用DataFrameの準備 ---
            logging.info(f"予測用DataFrameを準備中 (CSV: {shutuba_csv_path})")
            df_live_pl = self._prepare_live_data(
                shutuba_csv_path, race_info, form_race_date
            )
            if df_live_pl is None or len(df_live_pl) == 0:
                return {"error": "出馬表データ (live data) の準備に失敗しました。"}

            live_horse_names = df_live_pl["馬名"].to_list()
            live_race_date = pd.to_datetime(form_race_date)

            # --- 3. 過去データの取得 (★修正: 日付だけでフィルタリング) ---
            logging.info(
                f"全過去戦績 (特徴量計算済み、{live_race_date} より前) を取得中..."
            )
            # ▼▼▼【変更点】馬名 (horse_names) でフィルタしない ▼▼▼
            df_past_pl = self.df_featured_all_pl.filter(pl.col("日付") < live_race_date)
            # ▲▲▲【変更点ここまで】▲▲▲
            logging.info(f"  取得件数: {len(df_past_pl)}")

            # (過去データがない馬への警告はそのまま or 削除しても良い)
            horses_with_past_in_full_history = set(
                df_past_pl.filter(pl.col("馬名").is_in(live_horse_names))["馬名"]
                .unique()
                .to_list()
            )
            horses_without_past = (
                set(live_horse_names) - horses_with_past_in_full_history
            )
            if horses_without_past:
                logging.warning(
                    f"過去データがない馬が {len(horses_without_past)} 頭います: {horses_without_past}"
                )
                logging.warning(
                    "これらの馬は特徴量が計算できないため、予測精度が低下します。"
                )

            # --- 4. 全過去データと予測対象レースを結合 ---
            # (スキーマ合わせのロジックは変更なし)
            past_schema = df_past_pl.schema
            live_schema = df_live_pl.schema
            past_cols_set = set(past_schema.keys())
            live_cols_set = set(live_schema.keys())
            all_cols_ordered = list(past_schema.keys()) + [
                col for col in live_schema.keys() if col not in past_cols_set
            ]
            merged_schema = {**live_schema, **past_schema}
            select_exprs_past = []
            for col in all_cols_ordered:
                if col in past_cols_set:
                    select_exprs_past.append(
                        pl.col(col).cast(merged_schema[col], strict=False)
                    )
                else:
                    select_exprs_past.append(
                        pl.lit(None, dtype=merged_schema[col]).alias(col)
                    )
            select_exprs_live = []
            for col in all_cols_ordered:
                if col in live_cols_set:
                    select_exprs_live.append(
                        pl.col(col).cast(merged_schema[col], strict=False)
                    )
                else:
                    select_exprs_live.append(
                        pl.lit(None, dtype=merged_schema[col]).alias(col)
                    )
            df_past_aligned = df_past_pl.select(select_exprs_past)
            df_live_aligned = df_live_pl.select(select_exprs_live)
            df_combined_pl = pl.concat(
                [df_past_aligned, df_live_aligned], how="vertical"
            )
            df_combined_pl = df_combined_pl.sort(
                ["馬名", "日付"]
            )  # ソートは馬名→日付のまま

            # ▼▼▼【★★★ 追加: 基準値の補完 ★★★】▼▼▼
            logging.info("結合データに基準値カラムを補完中...")

            # 予測対象レースのコース情報を取得
            race_location = race_info.get("race_location", "不明")
            track_type = race_info.get("race_track_type", "芝")
            distance = str(race_info.get("race_distance", 0))
            condition = race_info.get("race_track_condition", "良")
            course_key = (track_type, distance, condition)

            # 欠けている基準値カラムをチェック
            missing_baseline_cols = [
                col
                for col in self.baseline_columns
                if col not in df_combined_pl.columns
            ]

            if missing_baseline_cols:
                logging.info(f"欠けている基準値カラム数: {len(missing_baseline_cols)}")

                # 基準値辞書から該当コースの値を取得して補完
                if course_key in self.baseline_dict:
                    baseline_values = self.baseline_dict[course_key]
                    for col in missing_baseline_cols:
                        if col in baseline_values:
                            baseline_value = baseline_values[col]
                            df_combined_pl = df_combined_pl.with_columns(
                                pl.lit(baseline_value).alias(col)
                            )
                            logging.info(f"  補完: {col} = {baseline_value:.4f}")
                        else:
                            # フォールバック: 0で補完
                            df_combined_pl = df_combined_pl.with_columns(
                                pl.lit(0.0).alias(col)
                            )
                            logging.warning(f"  {col}: 基準値が見つからず0で補完")
                else:
                    # コース条件が辞書にない場合は0で補完
                    logging.warning(
                        f"コース条件 {course_key} の基準値が見つかりません。0で補完します。"
                    )
                    for col in missing_baseline_cols:
                        df_combined_pl = df_combined_pl.with_columns(
                            pl.lit(0.0).alias(col)
                        )
            else:
                logging.info("全ての基準値カラムが既に存在します。")
            # ▲▲▲【★★★ 基準値補完ここまで ★★★】▲▲▲

            # ▼▼▼【★★★ 修正: 結合後に特徴量を再計算 ★★★】▼▼▼
            logging.info("結合データに対して特徴量生成を実行中...")
            # 結合後のデータに対して feature_engineering を実行
            # これにより、予測対象レースの行にも正しく特徴量が計算される
            df_featured_combined_pl = feature_engineering(df_combined_pl)
            logging.info(f"特徴量生成完了。Shape: {df_featured_combined_pl.shape}")
            # ▲▲▲【★★★ 修正ここまで ★★★】▲▲▲

            # ▼▼▼ 【★★★ 生の値確認コード ★★★】 ▼▼▼
            try:
                logging.info(
                    f"\n--- [DEBUG] スケーリング前の生の値確認 (Race: {race_info.get('race_name', '不明')}) ---"
                )
                # 生データから当レース分を抽出
                df_featured_live_pl = df_featured_combined_pl.filter(
                    (pl.col("日付") == live_race_date)
                    & (pl.col("馬名").is_in(live_horse_names))
                )

                # 比較対象の特徴量リスト
                features_to_check = [
                    "斤量体重比",
                    "前走との上り3F差",
                    "前走との平均速度差",
                    "追走指数",
                    # (参考: 計算元の値も確認)
                    "斤量",
                    "馬体重",
                    "上り3F",
                    "平均速度",
                    "キャリア平均_4角順位率",  # 追走指数の計算元
                ]

                if len(df_featured_live_pl) > 0:
                    logging.info(
                        f"  --- 1頭目 ({df_featured_live_pl[0, '馬名']}) の生の値 ---"
                    )
                    for feat in features_to_check:
                        if feat in df_featured_live_pl.columns:
                            # .item() を使って Polars の値を Python の値に変換
                            raw_value = df_featured_live_pl[0, feat]
                            logging.info(f"    {feat:<20}: {raw_value}")
                        else:
                            logging.warning(f"    {feat:<20}: カラム欠損 ⚠️")
                else:
                    logging.warning("  df_featured_live_pl が空です。")

            except Exception as e_raw:
                logging.error(f"  生の値のデバッグ中にエラー: {e_raw}")
            logging.info("--------------------------------------------------\n")
            # ▲▲▲ 【★★★ 生の値確認コード ★★★】 ▲▲▲

            # ▼▼▼ 【★★★ デバッグプリント (スケーリング前) ★★★】 ▼▼▼
            try:
                logging.info(
                    f"\n--- [DEBUG] スケーリング「前」の生の特徴量 (Race: {race_info.get('race_name', '不明')}) ---"
                )

                # スケーリング前の生データ (df_featured_combined_pl) から当レース分を抽出
                df_featured_live_pl = df_featured_combined_pl.filter(
                    (pl.col("日付") == live_race_date)
                    & (pl.col("馬名").is_in(live_horse_names))
                )

                # モデルが使用する全特徴量リスト
                all_model_features = sorted(
                    list(
                        set(
                            self.static_feature_names_ordered
                            + self.sequence_feature_names_ordered
                        )
                    )
                )

                # 表示対象カラム (メタ情報 + 全特徴量)
                meta_cols = ["馬名", "馬番"]

                # ▼▼▼ 【変更前】 ▼▼▼
                # cols_to_print = meta_cols + [col for col in all_model_features if col in df_featured_live_pl.columns]

                # ▼▼▼ 【変更後】(重複を除外する) ▼▼▼
                feature_cols_to_print = [
                    col
                    for col in all_model_features
                    if col in df_featured_live_pl.columns
                ]
                seen = set()
                cols_to_print = []
                for col in meta_cols + feature_cols_to_print:
                    if col not in seen:
                        seen.add(col)
                        cols_to_print.append(col)
                # ▲▲▲ 【変更ここまで】 ▲▲▲

                df_print_pd = df_featured_live_pl.select(cols_to_print).to_pandas()

                for _, row in df_print_pd.iterrows():
                    print(f"\n  🐴 {row['馬名']} (馬番: {row['馬番']})")
                    for col_name in all_model_features:
                        if col_name in row:
                            val = row[col_name]
                            if isinstance(val, (float, np.floating)):
                                print(f"      {col_name:<30}: {val:.4f}")
                            else:
                                print(f"      {col_name:<30}: {val}")
                        else:
                            # (リストにはあるがDFにない特徴量)
                            pass

            except Exception as e:
                logging.error(f"  スケーリング前 生特徴量のprint中にエラー: {e}")
                logging.error(traceback.format_exc())
            logging.info("--------------------------------------------------\n")
            # ▲▲▲ 【★★★ デバッグプリントここまで ★★★】 ▲▲▲

            # ▼▼▼ 【デバッグ追加】▼▼▼
            try:
                logging.info(
                    "\n--- [DEBUG] feature_engineering 直前の df_combined_pl (武豊の過去データ) ---"
                )
                df_take_past = df_combined_pl.filter(
                    (pl.col("騎手") == "武豊")
                    & (pl.col("日付") < live_race_date)  # 3文字キーでフィルタ
                ).sort("日付", descending=True)

                if len(df_take_past) > 0:
                    logging.info(
                        f"  武豊の過去データ件数 (結合後): {len(df_take_past)}"
                    )
                    # 複勝圏内カラムの存在とデータ型、Null率を確認
                    if "複勝圏内" in df_take_past.columns:
                        fukusho_col = df_take_past.select(pl.col("複勝圏内"))
                        logging.info(
                            f"  '複勝圏内' カラムのデータ型: {fukusho_col.dtypes[0]}"
                        )
                        logging.info(
                            f"  '複勝圏内' カラムの Null 数: {fukusho_col.null_count().row(0)[0]}"
                        )
                        logging.info(
                            f"  直近5走の '複勝圏内' 値: {fukusho_col.head(5).to_series().to_list()}"
                        )
                    else:
                        logging.warning("  '複勝圏内' カラムが見つかりません！")
                else:
                    logging.warning(
                        "  結合後のデータに '武豊' の過去データが見つかりません！"
                    )

            except Exception as e:
                logging.error(f"  df_combined_pl のデバッグ中にエラー: {e}")
            logging.info("--------------------------------------------------\n")
            # ▲▲▲ 【デバッグ追加ここまで】▲▲▲

            logging.info(f"\n{'='*60}")
            logging.info(f"[DEBUG] 過去データ取得結果")
            logging.info(f"  - 取得日付範囲: ~{live_race_date}")
            logging.info(f"  - 全馬名数: {df_past_pl['馬名'].n_unique()}")
            logging.info(f"  - 全行数: {len(df_past_pl)}")
            logging.info(
                f"  - 出走馬の過去データ行数: {df_past_pl.filter(pl.col('馬名').is_in(live_horse_names)).shape[0]}"
            )
            logging.info(f"{'='*60}\n")

            # --- 6. 前処理 & シーケンス生成 ---
            logging.info("前処理とシーケンス生成 (Polars Shift) を実行中...")
            df_processed_all = self._preprocess_for_prediction(df_featured_combined_pl)

            # --- 7. 予測対象行の抽出 ---
            df_processed_live = df_processed_all.filter(
                (pl.col("日付") == live_race_date)
                & (pl.col("馬名").is_in(live_horse_names))
            )
            if len(df_processed_live) == 0:
                return {
                    "error": "前処理・シーケンス生成後、予測対象の行が見つかりません。"
                }
            processed_horses = df_processed_live["馬名"].to_list()
            logging.info(f"予測対象 (処理後): {processed_horses}")

            # --- 8. テンソル抽出 ---
            logging.info("予測用テンソルを抽出中...")
            X_seq_tensor, X_static_tensor = self._extract_tensors_from_df(
                df_processed_live
            )
            if X_seq_tensor is None:
                return {"error": "テンソルの抽出に失敗しました。"}

            # ▼▼▼ 【デバッグ追加: テンソル値比較】 ▼▼▼
            try:
                logging.info(
                    f"\n--- [DEBUG] モデル入力直前のテンソル値 (Race: {race_info.get('race_name', '不明')}) ---"
                )

                # ▼▼▼ 【★★★ 特徴量名確認コード ★★★】 ▼▼▼
                logging.info(f"--- [DEBUG] 静的特徴量のインデックス確認 ---")
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
                # ▲▲▲ 【★★★ 特徴量名確認コード ★★★】 ▲▲▲

                # 最初の馬 (ルクスフロンティアのはず) のテンソル値を出力
                if X_static_tensor.shape[0] > 0:
                    static_vals_np = X_static_tensor[0].cpu().numpy()
                    # ▼▼▼【変更】[:10] を削除 ▼▼▼
                    logging.info(f"  X_static (1頭目, 全次元): {static_vals_np}")
                    # ▲▲▲【変更ここまで】▲▲▲
                    logging.info(f"  X_static Shape: {X_static_tensor.shape}")
                if X_seq_tensor.shape[0] > 0:
                    seq_vals_np = X_seq_tensor[0].cpu().numpy()
                    # 最新ステップ (Lag 1 相当) の値を出力
                    # ▼▼▼【変更】[:10] を削除 ▼▼▼
                    logging.info(
                        f"  X_seq (1頭目, 最新ステップ, 全次元): {seq_vals_np[-1, :]}"
                    )
                    # ▲▲▲【変更ここまで】▲▲▲
                    logging.info(f"  X_seq Shape: {X_seq_tensor.shape}")
            except Exception as e_tensor:
                logging.error(f"  テンソル値のデバッグ中にエラー: {e_tensor}")
            logging.info("--------------------------------------------------\n")
            # ▲▲▲ 【デバッグ追加ここまで】 ▲▲▲

            logging.info(f"--- [DEBUG] モデルの状態確認 ---")  # (これは残す)
            logging.info(f"  model.training: {self.model.training}")
            logging.info(f"-----------------------------")

            # --- 9. 予測実行 (Deep Learning) ---
            logging.info(
                f"DLモデル予測を実行中 (Batch Size: {X_static_tensor.shape[0]})..."
            )
            with torch.no_grad():
                y_pred_tensor = self.model(X_seq_tensor, X_static_tensor)
            dl_scores = y_pred_tensor.cpu().numpy().squeeze()
            if dl_scores.ndim == 0:
                dl_scores = [dl_scores.item()]
            else:
                dl_scores = dl_scores.tolist()
            logging.info(f"DLモデル予測完了。スコア: {dl_scores}")

            # --- 10. (Pandas) DFにDLスコアをマージ ---
            df_processed_live_pd = df_processed_live.to_pandas()
            if len(df_processed_live_pd) == len(dl_scores):
                df_processed_live_pd["予測スコア"] = (
                    dl_scores  # これが LGBM の特徴量になる
                )
            else:
                logging.error("DLスコアと処理済みDFの行数が一致しません！")
                return {"error": "DLスコアとDFの行数が不一致です。"}

            # ▼▼▼【ここから修正】▼▼▼
            # --- 11. 予測実行 (LightGBM 二重モデル) ---
            logging.info("LGBM二重モデル予測を実行中...")
            try:
                # LGBM が必要とする特徴量セットを抽出
                # (self.feature_cols_lgbm に "予測スコア" が含まれている必要がある)

                # (デバッグ用) スクレピングしたてのDF (df_live_pl) からオッズをマージし直す
                # → feature_engineering が '単勝オッズ' を 0 で埋めている可能性があるため
                df_live_pd_simple = df_live_pl.to_pandas()[["馬名", "単勝オッズ"]]
                df_processed_live_pd = pd.merge(
                    df_processed_live_pd.drop(columns=["単勝オッズ"], errors="ignore"),
                    df_live_pd_simple,
                    on="馬名",
                    how="left",
                )

                missing_lgbm_cols = [
                    col
                    for col in self.feature_cols_lgbm
                    if col not in df_processed_live_pd.columns
                ]
                if missing_lgbm_cols:
                    logging.error(
                        f"LGBMの予測に必要な特徴量が不足しています: {missing_lgbm_cols}"
                    )
                    for col in missing_lgbm_cols:
                        df_processed_live_pd[col] = 0  # 不足分は 0 で埋める

                # Null/Inf が残っているとLGBMがエラーを起こすため、最終チェック
                X_test_lgbm = (
                    df_processed_live_pd[self.feature_cols_lgbm]
                    .fillna(0)
                    .replace([np.inf, -np.inf], 0)
                )

                # 予測実行
                y_pred_rank = self.lgbm_ranker.predict(X_test_lgbm)
                y_pred_proba = self.lgbm_classifier.predict_proba(X_test_lgbm)[:, 1]

                # DFにLGBMの予測結果を追加
                df_processed_live_pd["predict_rank_score"] = y_pred_rank
                df_processed_live_pd["predict_win_proba"] = y_pred_proba

                logging.info("LGBM二重モデル予測完了。")

            except Exception as e:
                logging.error(f"LGBM予測ステップでエラー: {e}")
                logging.error(traceback.format_exc())
                return {"error": f"LGBM予測ステップでエラー: {e}"}

            # --- 12. ケリー基準による賭け金計算 ---
            logging.info("ケリー基準による賭け金計算を実行中...")

            # (X_test_lgbm に '単勝オッズ' が含まれているので、df_processed_live_pd にも存在するはず)
            if "単勝オッズ" not in df_processed_live_pd.columns:
                logging.error("賭け金計算に必要な '単勝オッズ' がDFにありません！")
                # (上記のマージ処理でカバーされているはずだが念のため)
                return {"error": "賭け金計算に必要な '単勝オッズ' がDFにありません。"}

            bet_amounts_dict = self._calculate_kelly_bet(df_processed_live_pd)
            df_processed_live_pd["推奨賭け金"] = (
                df_processed_live_pd["馬名"].map(bet_amounts_dict).fillna(0)
            )

            logging.info(f"賭け金計算完了: {bet_amounts_dict}")

            # --- 13. 結果の整形 ---
            logging.info("予測結果を整形中...")
            df_live_pd = df_live_pl.to_pandas()

            cols_to_merge = ["馬名", "馬番", "枠番", "騎手", "単勝オッズ"]
            lgbm_results_cols = [
                "馬名",
                "予測スコア",
                "predict_rank_score",
                "predict_win_proba",
                "推奨賭け金",
            ]

            df_results = pd.merge(
                df_live_pd[cols_to_merge],
                df_processed_live_pd[lgbm_results_cols],
                on="馬名",
                how="left",
            )

            # ★ ソート基準を LGBMランカー(predict_rank_score) に変更
            df_results = df_results.sort_values(
                "predict_rank_score", ascending=False
            ).reset_index(drop=True)

            # --- 14. 詳細情報 (ファクター, 過去戦績) の取得 ---
            df_past_featured_pd = df_featured_combined_pl.filter(
                pl.col("日付") < live_race_date
            ).to_pandas()
            df_featured_live_pd = df_featured_combined_pl.filter(
                (pl.col("日付") == live_race_date)
                & (pl.col("馬名").is_in(live_horse_names))
            ).to_pandas()

            factors_dict, history_dict = self._get_factors_and_history(
                df_featured_live_pd, df_past_featured_pd
            )

            predictions_list = []
            for _, row in df_results.iterrows():
                horse_name = row["馬名"]
                predictions_list.append(
                    {
                        "馬番": row["馬番"],
                        "枠番": row["枠番"],
                        "馬名": horse_name,
                        "騎手": row["騎手"],
                        "単勝オッズ": row["単勝オッズ"],
                        # ▼▼▼【ここを修正】▼▼▼
                        "予測スコア": row["予測スコア"],  # DLモデルのスコア
                        "LGBM_rank_score": row[
                            "predict_rank_score"
                        ],  # ★ LGBMランカーのスコア
                        "推奨賭け金": row["推奨賭け金"],  # ★ ケリー基準の賭け金
                        "factors": factors_dict.get(horse_name, []),
                        "history": history_dict.get(horse_name, []),
                    }
                )
            # ▲▲▲【ここまで修正】▲▲▲

            final_race_info = {
                "name": race_info.get("race_name", "不明"),
                "location": race_info.get("race_location", "不明"),
                "number": race_info.get("race_number", "R"),
                "distance": race_info.get("race_distance", 0),
                "track": race_info.get("race_track_type", "芝"),
                "condition": race_info.get("race_track_condition", "良"),
                "date": form_race_date,
            }

            return {
                "success": True,
                "race_info": final_race_info,
                "predictions": predictions_list,
            }
        except Exception as e:
            logging.error(f"run_prediction で致命的なエラー: {e}")
            logging.error(traceback.format_exc())
            return {"error": f"予測処理中にエラーが発生しました: {e}"}


# --- main ブロック (テスト用) ---
if __name__ == "__main__":
    # --- パス設定 ---
    HISTORICAL_DATA_PATH = "2010_2025_data.csv"
    PEDIGREE_DATA_PATH = "2005_2025_Pedigree.csv"
    PREPROCESSOR_DIR = "lstm_preprocessor_score"
    MODEL_DIR = "lstm_models_score"

    # --- テスト用のダミーデータ ---
    TEST_URL = "dummy_url"  # 実際には使われない
    TEST_SHUTUBA_CSV = "shutuba_temp.csv"  # scraper.py が生成する想定
    TEST_DATE = "2025-09-27"  # 予測したいレースの日付
    TEST_RACE_INFO = {
        "race_name": "ポートアイランドS",
        "race_location": "阪神",
        "race_number": 11,
        "race_distance": 1600,
        "race_track_type": "芝",
        "race_track_condition": "良",
        "race_class": "L(リステッド)",
        "datetime": "2025年9月27日 15:35",
    }
    dummy_shutuba_data = {
        "枠 番": [1, 2, 3],
        "馬 番": [1, 2, 3],
        "馬名": ["テスト馬A", "テスト馬B", "テスト馬C"],
        "性齢": ["牡3", "牝4", "牡5"],
        "斤 量": [55.0, 54.0, 57.0],
        "騎 手": ["騎手X", "騎手Y", "騎手Z"],
        "厩舎": ["栗東・厩舎A", "美浦・厩舎B", "栗東・厩舎C"],
        "馬 体重": ["480(+2)", "450(-4)", "500(0)"],
        "オッズ": [3.5, 8.2, 15.0],
    }
    pd.DataFrame(dummy_shutuba_data).to_csv(TEST_SHUTUBA_CSV, index=False)
    logging.basicConfig(level=logging.INFO)  # ログ表示設定

    try:
        predictor = Predictor(
            historical_data_path=HISTORICAL_DATA_PATH,
            pedigree_data_path=PEDIGREE_DATA_PATH,
            preprocessor_base_dir=PREPROCESSOR_DIR,
            model_base_dir=MODEL_DIR,
        )

        result = predictor.run_prediction(
            url=TEST_URL,
            shutuba_csv_path=TEST_SHUTUBA_CSV,
            form_race_date=TEST_DATE,
            race_info=TEST_RACE_INFO,
        )

        if "error" in result:
            print(f"\n--- 予測エラー ---")
            print(result["error"])
        else:
            print("\n--- 予測成功 ---")
            print("レース情報:", result.get("race_info"))
            print("予測結果 (上位3件):")
            for pred in result.get("predictions", [])[:3]:
                print(
                    f"  馬番:{pred['馬番']}, 馬名:{pred['馬名']}, スコア:{pred['予測スコア']:.4f}"
                )

    except FileNotFoundError as e:
        print(f"ファイルが見つかりません: {e}")
        print(
            "必要なデータファイル (CSV) や、学習済みモデル/前処理ファイルが存在するか確認してください。"
        )
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        print(traceback.format_exc())
    finally:
        if os.path.exists(TEST_SHUTUBA_CSV):
            os.remove(TEST_SHUTUBA_CSV)
