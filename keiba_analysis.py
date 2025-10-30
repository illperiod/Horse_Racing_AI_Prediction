import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, TensorDataset
import json
import joblib
import os
import re
from tqdm import tqdm
from deepctr_torch.inputs import (
    SparseFeat,
    DenseFeat,
    get_feature_names,
    build_input_features,
    combined_dnn_input,
)
import shap  # <-- ▼▼▼ 追加 ▼▼▼
import matplotlib.pyplot as plt  # <-- ▼▼▼ 追加 ▼▼▼
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.layers import CrossNet, DNN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# from sklearn.cluster import KMeans # <-- 削除
import gc
import matplotlib

matplotlib.rcParams["font.sans-serif"] = [
    "Yu Gothic",
    "MS Gothic",
    "Meiryo",
]  # Windows用
matplotlib.rcParams["axes.unicode_minus"] = False
import types
import polars as pl
import torch.nn.functional as F
from collections import defaultdict

# --- グローバル設定 ---
INPUT_DATA_PATH = "features_engineered.csv"
PREPROCESSOR_BASE_DIR = "lstm_preprocessor_score"
MODEL_OUTPUT_DIR = "lstm_models_score"
SUMMARY_JSON_FILENAME = "summary_stepA_lstm.json"

# --- 学習ハイパーパラメータ ---
EMBEDDING_DIM = 8
CROSS_NUM = 3
DNN_HIDDEN_UNITS = (256, 128)
L2_REG = 1e-5
DROPOUT_RATE = 0.3
N_STEPS = 5
LSTM_HIDDEN_SIZE = 64
LSTM_LAYERS = 1
LSTM_DROPOUT = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 256
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
LOG_INTERVAL = 1000

# True にすると学習をスキップし、既存の .pth ファイルを読み込んでSHAP分析のみ実行
RUN_SHAP_ONLY = False
MIN_DATA_SIZE = N_STEPS + 1


def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def safe_mse_score(y_true, y_pred):
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    if (
        y_true_np.shape[0] == 0
        or y_pred_np.shape[0] == 0
        or not np.isfinite(y_true_np).all()
        or not np.isfinite(y_pred_np).all()
    ):
        return np.inf
    return mean_squared_error(y_true_np, y_pred_np)


def safe_mae_score(y_true, y_pred):
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    if (
        y_true_np.shape[0] == 0
        or y_pred_np.shape[0] == 0
        or not np.isfinite(y_true_np).all()
        or not np.isfinite(y_pred_np).all()
    ):
        return np.inf
    return mean_absolute_error(y_true_np, y_pred_np)


def create_recency_based_mapping(df, column_name):
    print(f"--- {column_name} の日付順マッピングを作成中 ---")
    df_sorted = df.sort_values("日付", ascending=False)
    unique_items = df_sorted[column_name].dropna().unique()
    item_to_id = {name: i + 1 for i, name in enumerate(unique_items)}
    id_to_item = {i + 1: name for i, name in enumerate(unique_items)}
    print(f"--- {column_name} のマッピング完了 ({len(unique_items)}件) ---")
    return item_to_id, id_to_item


# --- カスタムデータセットクラス ---
class HorseRaceDataset(Dataset):
    # keiba_analysis.py の HorseRaceDataset.__init__

    def __init__(
        self,
        df,
        all_feature_names,
        sequence_feature_names,
        static_feature_names,
        categorical_features_set,
        numerical_features_set,
        target_col,
        n_steps=5,
    ):
        self.df = df  # This should be the original Pandas DF (train_df_full etc.)
        self.all_feature_names = all_feature_names
        self.sequence_feature_names = sequence_feature_names
        self.static_feature_names = static_feature_names
        self.target_col = target_col
        self.n_steps = n_steps

        self.categorical_features = categorical_features_set
        self.numerical_features = numerical_features_set

        print("--- データセットのインデックスを作成中 ---")
        self.indices = []

        # Use a temporary copy ONLY for groupby operations to get indices
        df_pd_temp = self.df.copy()
        df_pd_temp["_horse_race_idx"] = df_pd_temp.groupby("馬名").cumcount()

        valid_current_indices_labels = df_pd_temp[
            df_pd_temp["_horse_race_idx"] >= n_steps
        ].index  # These are LABELS
        horse_group_indices = df_pd_temp.groupby("馬名").groups  # Groups using LABELS

        pbar = tqdm(valid_current_indices_labels, desc="馬ごとにシーケンスを準備")
        for current_idx_label in pbar:  # Iterate through LABELS
            horse_name = df_pd_temp.loc[
                current_idx_label, "馬名"
            ]  # Access temp df by LABEL
            all_horse_indices_labels = horse_group_indices[
                horse_name
            ]  # Get all LABELS for the horse
            try:
                # Find the POSITION of the current LABEL within the horse's group LABELS
                current_pos_in_group = all_horse_indices_labels.get_loc(
                    current_idx_label
                )
            except KeyError:
                continue  # Should not happen if index logic is correct

            if current_pos_in_group >= n_steps:
                # Get the LABELS of the past n steps using POSITION slicing
                past_indices_labels = all_horse_indices_labels[
                    current_pos_in_group - n_steps : current_pos_in_group
                ]

                # ▼▼▼【ここを修正】▼▼▼
                # Store the LABELS directly. __getitem__ uses .loc which works with labels.
                original_current_idx_label = current_idx_label
                original_past_indices_labels = past_indices_labels.tolist()
                self.indices.append(
                    (original_current_idx_label, original_past_indices_labels)
                )
                # ▲▲▲【ここまで修正】▲▲▲

        print(f"--- インデックス作成完了 ({len(self.indices)} サンプル) ---")

    # __len__ and __getitem__ methods remain the same as they use .loc with labels
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if idx >= len(self.indices):
            raise IndexError("Index out of bounds")
        # These are the LABELS stored in __init__
        current_idx_label, past_indices_labels = self.indices[idx]

        # --- シーケンスデータ取得 & テンソル化 ---
        seq_tensors = []
        try:
            # Slice the past indices labels list
            actual_past_indices_labels = past_indices_labels[-self.n_steps :]
            num_actual_past = len(actual_past_indices_labels)
            num_padding = self.n_steps - num_actual_past

            if num_actual_past > 0:
                # Use .loc with the list of LABELS on the original DataFrame
                past_races_df = self.df.loc[
                    actual_past_indices_labels, self.sequence_feature_names
                ]
            else:
                past_races_df = pd.DataFrame(columns=self.sequence_feature_names)

            # (Rest of the sequence tensor creation is the same)
            for name in self.sequence_feature_names:
                values = (
                    past_races_df[name].values if num_actual_past > 0 else np.array([])
                )
                # ... (padding and tensor creation) ...
                if name in self.categorical_features:
                    safe_values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                    padded_values = np.pad(
                        safe_values, (num_padding, 0), "constant", constant_values=0
                    )
                    seq_tensors.append(
                        torch.tensor(padded_values, dtype=torch.float32).unsqueeze(-1)
                    )
                elif name in self.numerical_features:
                    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                    padded_values = np.pad(
                        values, (num_padding, 0), "constant", constant_values=0.0
                    )
                    seq_tensors.append(
                        torch.tensor(padded_values, dtype=torch.float32).unsqueeze(-1)
                    )

            if seq_tensors:
                X_seq_tensor = torch.cat(seq_tensors, dim=-1)
            else:
                X_seq_tensor = torch.empty((self.n_steps, 0), dtype=torch.float32)

        except Exception as e:
            print(
                f"Error getting sequence data for idx {idx} (Label: {current_idx_label}): {e}"
            )
            raise e

        # --- 静的データ取得 & テンソル化 ---
        static_tensors = []
        try:
            # Use .loc with the single LABEL on the original DataFrame
            current_race_series = self.df.loc[
                current_idx_label, self.static_feature_names
            ]
            # (Rest of the static tensor creation is the same)
            for name in self.static_feature_names:
                value = current_race_series[name]
                # ... (tensor creation) ...
                if name in self.categorical_features:
                    val_int = 0 if pd.isna(value) else int(value)
                    tensor_val = torch.tensor(val_int, dtype=torch.float32)
                    static_tensors.append(tensor_val)
                elif name in self.numerical_features:
                    value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
                    tensor_val = torch.tensor(value, dtype=torch.float32)
                    static_tensors.append(tensor_val)

            if static_tensors:
                X_static_tensor = torch.stack(static_tensors)
            else:
                X_static_tensor = torch.empty((0,), dtype=torch.float32)
        except Exception as e:
            print(
                f"Error getting static data for idx {idx} (Label: {current_idx_label}): {e}"
            )
            raise e

        # --- 目的変数 ---
        try:
            # Use .loc with the single LABEL on the original DataFrame
            label_val = self.df.loc[current_idx_label, self.target_col]
            label = torch.tensor(np.nan_to_num(label_val, nan=0.0), dtype=torch.float32)
        except Exception as e:
            print(
                f"Error getting label for idx {idx} (Label: {current_idx_label}): {e}"
            )
            raise e

        return X_seq_tensor, X_static_tensor, label


# --- ハイブリッドモデルクラス ---
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

        # Pass all sparse features to BaseModel for embedding dict creation
        all_sparse_fcs = [
            fc
            for fc in static_linear_feature_columns + sequence_feature_columns
            if isinstance(fc, SparseFeat)
        ]
        all_dense_fcs = [
            fc for fc in static_linear_feature_columns if isinstance(fc, DenseFeat)
        ]  # Only static dense for linear part

        super(DCN_LSTM_Hybrid, self).__init__(
            linear_feature_columns=static_linear_feature_columns,
            dnn_feature_columns=static_dnn_feature_columns,  # <-- ★ + sequence_feature_columns を削除
            l2_reg_linear=l2_reg_linear,
            l2_reg_embedding=l2_reg_embedding,
            init_std=init_std,
            seed=seed,
            task=task,
            device=device,
            gpus=gpus,
        )

        self.static_feature_names = static_feature_names
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

        # ▼▼▼【 デバッグプリント追加 1/2 】▼▼▼
        print("\n--- [DEBUG] Calculating LSTM Input Size ---")
        lstm_input_size = 0
        print("  Sparse Features for LSTM:")
        for fc in seq_sparse_feature_columns:
            if fc.embedding_name in self.embedding_dict:
                dim = self.embedding_dict[fc.embedding_name].embedding_dim
                print(f"    - {fc.name}: Embedding Dim = {dim}")
                lstm_input_size += dim
            else:
                print(f"    - {fc.name}: [Warning] Embedding key not found!")
        print(f"  Subtotal from Sparse Embeddings: {lstm_input_size}")

        print("  Dense Features for LSTM:")
        for fc in seq_dense_feature_columns:
            print(f"    - {fc.name}: Dim = 1")
            lstm_input_size += 1
        print(f"  Count of Dense Features: {len(seq_dense_feature_columns)}")
        print(f"  >>> Final Calculated LSTM Input Size: {lstm_input_size}")
        # ▲▲▲【 デバッグプリント追加ここまで 】▲▲▲

        # Use the embedding_dict created by BaseModel
        lstm_input_size = sum(
            self.embedding_dict[fc.embedding_name].embedding_dim
            for fc in seq_sparse_feature_columns
            if fc.embedding_name in self.embedding_dict  # Check existence
        ) + len(seq_dense_feature_columns)

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
        )

        # --- DCN Part ---
        # Calculate input dim for DNN (static features + LSTM output)
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

        # Calculate input dim for CrossNet (static linear features only)
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
                    except IndexError as e:
                        # Handle potential index error if tensor shape is wrong
                        pass
                    except Exception as e:
                        # Handle other embedding errors
                        pass
                current_seq_col_idx += 1  # Increment index even if embedding not found
            elif isinstance(fc, DenseFeat):
                try:
                    val = X_seq_tensor[:, :, current_seq_col_idx].unsqueeze(-1)
                    val = torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
                    seq_dense_value_list.append(val)
                except IndexError as e:
                    # Handle potential index error
                    pass
                except Exception as e:
                    # Handle other dense feature errors
                    pass
                current_seq_col_idx += 1  # Increment index for dense features too

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
                # Handle LSTM processing error, maybe due to input size mismatch
                calculated_input_size = sum(t.shape[-1] for t in lstm_input_tensors)
                print(
                    f"LSTM Error: Expected {self.lstm.input_size}, Got {calculated_input_size}, Err: {e}"
                )
                lstm_output_features = torch.zeros(
                    (batch_size, self.lstm.hidden_size)
                ).to(self.device)

        # --- Prepare DCN Input (Static Features) ---
        # Use BaseModel's method with the correct feature_index
        try:
            linear_sparse_embedding_list, linear_dense_value_list = (
                self.input_from_feature_columns(
                    X_static_tensor,
                    self.static_linear_feature_columns,
                    self.embedding_dict,
                    self.feature_index,  # Use the index built by BaseModel
                )
            )
            dnn_sparse_embedding_list, dnn_dense_value_list = (
                self.input_from_feature_columns(
                    X_static_tensor,
                    self.static_dnn_feature_columns,
                    self.embedding_dict,
                    self.feature_index,  # Use the index built by BaseModel
                )
            )
        except Exception as e:
            # Handle errors during static feature processing
            print(f"Error processing static features: {e}")
            linear_sparse_embedding_list, linear_dense_value_list = [], []
            dnn_sparse_embedding_list, dnn_dense_value_list = [], []

        # CrossNet Input
        if not linear_sparse_embedding_list and not linear_dense_value_list:
            # Handle case where no linear features are present
            cross_input = torch.zeros((batch_size, self.cross_input_dim)).to(
                self.device
            )
        else:
            cross_input = combined_dnn_input(
                linear_sparse_embedding_list, linear_dense_value_list
            )

        # DNN Input
        if not dnn_sparse_embedding_list and not dnn_dense_value_list:
            # Only LSTM output goes to DNN if no static DNN features
            if (
                lstm_output_features is not None
                and lstm_output_features.shape[0] == batch_size
            ):
                dnn_input_combined = lstm_output_features
            else:  # Handle case where LSTM also had issues
                dnn_input_combined = torch.zeros((batch_size, self.dnn_input_dim)).to(
                    self.device
                )
        else:
            dnn_input_static = combined_dnn_input(
                dnn_sparse_embedding_list, dnn_dense_value_list
            )
            # Combine static DNN features and LSTM output
            if (
                lstm_output_features is None
                or lstm_output_features.shape[0] != batch_size
            ):
                # If LSTM output is missing or has wrong batch size, use zeros
                if (
                    dnn_input_static.shape[0] == batch_size
                ):  # Check if static part is ok
                    zero_lstm_padding = torch.zeros(
                        (batch_size, self.lstm.hidden_size)
                    ).to(self.device)
                    dnn_input_combined = torch.cat(
                        [dnn_input_static, zero_lstm_padding], dim=-1
                    )
                else:  # If static part also has wrong batch size (shouldn't happen with valid input)
                    dnn_input_combined = torch.zeros(
                        (batch_size, self.dnn_input_dim)
                    ).to(self.device)
            else:
                dnn_input_combined = torch.cat(
                    [dnn_input_static, lstm_output_features], dim=-1
                )

        # Check DNN input dimension
        expected_dnn_dim = self.dnn_input_dim
        actual_dnn_dim = dnn_input_combined.shape[1]
        if actual_dnn_dim != expected_dnn_dim:
            print(
                f"Error: DNN input dim mismatch: Expected={expected_dnn_dim}, Actual={actual_dnn_dim}"
            )
            # Handle mismatch, e.g., return zeros or raise error
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
                print(
                    f"Error: CrossNet input dim mismatch: Expected={expected_cross_dim}, Actual={actual_cross_dim}"
                )
            # Handle mismatch or zero input for CrossNet
            cross_out = torch.zeros_like(dnn_logit)

        # Final prediction
        final_logit = cross_out + dnn_logit
        y_pred = self.out(final_logit)

        # Handle potential NaN/Inf in output
        y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

        return y_pred


class SHAPCompatibleWrapper(nn.Module):
    """
    SHAP用のラッパーモデル
    inplace操作を回避し、viewの問題を解決
    """

    # ▼▼▼【ここを修正】▼▼▼
    def __init__(self, original_model):  # <-- `self` を引数の最初に追加
        # ▲▲▲【修正ここまで】▲▲▲
        super().__init__()
        self.model = original_model

    def forward(self, X_seq_tensor, X_static_tensor):
        # 入力をcloneしてviewの問題を回避
        X_seq = X_seq_tensor.clone()
        X_static = X_static_tensor.clone()

        # モデルの予測を取得
        with torch.no_grad():
            output = self.model(X_seq, X_static)

        # 勾配計算用に再度forward (cloneされた入力で)
        output_for_grad = self.model(X_seq_tensor, X_static_tensor)

        # 出力もcloneして返す
        return output_for_grad.clone()


def prepare_tensor_dataset(dataset, device):
    """
    HorseRaceDatasetを一度だけループし、
    全データをメモリ上のテンソルに展開する。
    """
    print(f"--- データセットをメモリに一括展開中（{len(dataset)}件） ---")

    # BATCH_SIZE=1, num_workers=0 のローダーで1件ずつ安全に処理
    temp_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=8
    )  # Increased workers

    all_seq_tensors = []
    all_static_tensors = []
    all_labels = []

    # このループは時間がかかる（が、実行は最初の一度だけ）
    for seq_tensor, static_tensor, label in tqdm(
        temp_loader, desc="データをテンソルに変換中"
    ):
        all_seq_tensors.append(seq_tensor.squeeze(0))
        all_static_tensors.append(static_tensor.squeeze(0))
        all_labels.append(label.squeeze(0))

    print("--- テンソルのスタック処理中 ---")
    try:
        X_seq = torch.stack(all_seq_tensors)
        X_static = torch.stack(all_static_tensors)
        y = torch.stack(all_labels)

        print(f"シーケンス形状: {X_seq.shape}")
        print(f"静的データ形状: {X_static.shape}")
        print(f"ラベル形状: {y.shape}")

        del all_seq_tensors, all_static_tensors, all_labels
        gc.collect()

        # GPUが使えるなら、最初からテンソルをGPUに置いてしまう (Optional)
        # if device == "cuda":
        #     print("--- 全テンソルをVRAMに移動中 ---")
        #     X_seq = X_seq.to(device)
        #     X_static = X_static.to(device)
        #     y = y.to(device)

        return TensorDataset(X_seq, X_static, y)

    except Exception as e:
        print(f"テンソルのスタック処理中にエラー: {e}")
        print("データが空か、形状が不揃いの可能性があります。")
        return None


# --- 学習ループ関数 ---
def train_hybrid_model(
    train_loader, val_loader, model, device, output_model_dir, current_batch_size
):
    print(f"\n--- LSTM Hybridモデルの学習処理を開始 ---")
    if train_loader is None or val_loader is None:
        print(f"--- 学習または検証データ不足 ---")
        return False, None, None

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG)
    loss_func = nn.MSELoss()
    best_val_loss, patience_counter, best_model_state = np.inf, 0, None

    # ▼▼▼【AMP 追加 1/3】GradScalerを初期化 (GPU使用時のみ) ▼▼▼
    scaler = GradScaler(enabled=(device == "cuda"))
    # ▲▲▲【AMP 追加ここまで】▲▲▲

    try:
        train_loader_len = len(train_loader)
    except TypeError:
        train_loader_len = None  # Iterable dataset
    try:
        val_loader_len = len(val_loader)
    except TypeError:
        val_loader_len = None  # Iterable dataset

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        train_preds = []
        train_true = []
        train_iterator = (
            tqdm(train_loader, desc=f"Epoch {epoch+1} Train", total=train_loader_len)
            if train_loader_len is not None
            else tqdm(train_loader, desc=f"Epoch {epoch+1} Train")
        )

        for seq_tensor_batch, static_tensor_batch, label_batch in train_iterator:
            try:
                seq_tensor_batch = seq_tensor_batch.to(device)
                static_tensor_batch = static_tensor_batch.to(device)
                label_batch = label_batch.to(device)
            except Exception as e:
                print(f"Error moving batch to device: {e}")
                continue
            try:
                y_pred = model(seq_tensor_batch, static_tensor_batch).squeeze(-1)

                if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                    # print("NaN/Inf detected in prediction, skipping batch")
                    continue

                expected_label_shape = label_batch.squeeze().shape
                if y_pred.shape == expected_label_shape:
                    loss = loss_func(y_pred, label_batch.squeeze().float())
                    if torch.isnan(loss):
                        # print("NaN detected in loss, skipping batch")
                        continue

                    optimizer.zero_grad()  # zero_grad の位置を autocast の前に移動

                    # ▼▼▼【AMP 追加 2/3】autocastコンテキスト内で順伝播と損失計算 ▼▼▼
                    with autocast(enabled=(device == "cuda")):
                        try:
                            y_pred = model(
                                seq_tensor_batch, static_tensor_batch
                            ).squeeze(-1)

                            if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                                loss = None
                                continue  # continue は try ブロック内でOK

                            expected_label_shape = label_batch.squeeze().shape
                            if y_pred.shape == expected_label_shape:
                                # ★ 損失計算も autocast 内に入れる ★
                                loss = loss_func(y_pred, label_batch.squeeze().float())
                                if torch.isnan(loss):
                                    loss = None
                            else:
                                loss = None

                        except Exception as e:
                            print(f"Error during training forward/loss pass: {e}")
                            loss = None
                    # ▲▲▲【修正ここまで】▲▲▲

                    if loss is not None and not torch.isnan(loss):
                        # ▼▼▼【AMP 追加 3/3】scalerを使って逆伝播とoptimizerステップ ▼▼▼
                        scaler.scale(loss).backward()
                        # Unscale gradients before clipping
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        # scaler.step() は内部で unscale された勾配を使う
                        scaler.step(optimizer)
                        # 次のイテレーションのために scaler を更新
                        scaler.update()
                        # ▲▲▲【AMP 追加ここまで】▲▲▲

                        total_loss += loss.item()
                        train_preds.extend(y_pred.cpu().detach().numpy())
                        train_true.extend(label_batch.squeeze().cpu().detach().numpy())
                    # else: # Skip batch if loss is None or NaN
                    #    pass

            except Exception as e:
                print(f"Error during training step: {e}")
                import traceback

                traceback.print_exc()
                continue  # Skip batch on error

        if not train_true or not train_preds or train_loader_len == 0:
            print(f"Epoch {epoch+1}: No training data processed.")
            avg_train_loss = np.inf
            train_mae = np.inf
        else:
            avg_train_loss = (
                total_loss / train_loader_len if train_loader_len else total_loss
            )  # Avoid division by zero
            train_mae = safe_mae_score(train_true, train_preds)

        # --- Validation ---
        model.eval()
        val_preds = []
        val_true = []
        val_total_loss = 0
        val_iterator = (
            tqdm(val_loader, desc=f"Epoch {epoch+1} Val", total=val_loader_len)
            if val_loader_len is not None
            else tqdm(val_loader, desc=f"Epoch {epoch+1} Val")
        )
        with torch.no_grad():
            for seq_tensor_batch, static_tensor_batch, label_batch in val_iterator:
                try:
                    seq_tensor_batch = seq_tensor_batch.to(device)
                    static_tensor_batch = static_tensor_batch.to(device)
                    label_batch = label_batch.to(device)

                    y_pred = model(seq_tensor_batch, static_tensor_batch).squeeze(-1)
                    y_pred = torch.nan_to_num(y_pred, nan=0.0)

                    expected_label_shape = label_batch.squeeze().shape
                    if y_pred.shape == expected_label_shape:
                        val_loss = loss_func(y_pred, label_batch.squeeze().float())
                        if not torch.isnan(val_loss):
                            val_total_loss += val_loss.item()
                        val_preds.extend(y_pred.cpu().detach().numpy())
                        val_true.extend(label_batch.squeeze().cpu().detach().numpy())
                    # else:
                    # print(f"Shape mismatch (Val): y_pred:{y_pred.shape}, label:{expected_label_shape}")

                except Exception as e:
                    print(f"Error during validation step: {e}")
                    import traceback

                    traceback.print_exc()
                    continue  # Skip batch on error

        if not val_true or not val_preds or val_loader_len == 0:
            print(f"Epoch {epoch+1}: No validation data processed.")
            current_val_loss = np.inf
            val_mae = np.inf
        else:
            current_val_loss = (
                val_total_loss / val_loader_len if val_loader_len else val_total_loss
            )
            val_mae = safe_mae_score(val_true, val_preds)

        log_msg = f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Train MAE: {train_mae:.4f} | Val Loss: {current_val_loss:.4f} | Val MAE: {val_mae:.4f}"
        # Log less frequently if LOG_INTERVAL is large
        log_freq = max(1, train_loader_len // LOG_INTERVAL if train_loader_len else 10)
        if (epoch + 1) % log_freq == 0 or epoch + 1 == EPOCHS:
            print(log_msg)

        # Update tqdm postfix
        postfix_str = f"Tr L: {avg_train_loss:.3f}, V L: {current_val_loss:.3f}, V MAE: {val_mae:.3f}"
        if hasattr(train_iterator, "set_postfix_str"):  # Check if method exists
            train_iterator.set_postfix_str(postfix_str)

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            print(f"    -> Val Loss improved! Saving model state ({best_val_loss:.4f})")
        else:
            patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(
                f"--- Early stopping at epoch {epoch+1}. Best Val Loss: {best_val_loss:.4f} ---"
            )
            break

    if best_model_state:
        # Test evaluation is currently omitted for simplicity
        test_mae = np.inf  # Placeholder
        print(
            f"    -> [Finished] Best Val Loss: {best_val_loss:.4f}, Best Val MAE: {val_mae:.4f}"
        )
        # Use the MAE corresponding to the best validation loss epoch
        # (Technically, we should save the MAE when best_val_loss is updated)
        # For now, just report the final val_mae
        metrics = {"val_loss": best_val_loss, "val_mae": val_mae, "test_mae": test_mae}
        return True, metrics, best_model_state
    else:
        print("--- Training finished but no best model state was saved. ---")
        return False, None, None


# --- メイン関数 ---
# --- メイン関数 ---
def main():
    print("--- DCN-LSTM Hybrid モデル 学習プログラムを開始 ---")
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning, module="shap")
    warnings.filterwarnings("ignore", message="Glyph.*missing from font")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")

    try:
        # Use Polars for faster CSV reading
        df = pl.read_csv(INPUT_DATA_PATH).to_pandas()
        # Convert date column after reading
        df["日付"] = pd.to_datetime(df["日付"])
        df = df.sort_values(["馬名", "日付"]).reset_index(drop=True)
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return

    # --- Preprocessing ---
    print("--- 前処理を実行中 ---")
    # Shorten Jockey/Trainer names
    for col in ["騎手", "調教師"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.replace(r"[▲△☆\s]", "", regex=True).str[:3]
            )
    # Convert Class Level to numeric, create Class Category
    df["クラスレベル"] = (
        pd.to_numeric(df["クラスレベル"], errors="coerce").fillna(-1).astype(int)
    )
    conditions = [
        df["クラスレベル"] == 0,
        df["クラスレベル"].between(1, 4),
        df["クラスレベル"].between(5, 9),
    ]
    choices = ["新馬", "平場", "オープン"]
    df["クラスカテゴリ"] = np.select(conditions, choices, default="不明")

    # --- Feature Definition ---
    target_col = "rank_score"
    meta_cols = [
        "日付S",
        "日付",
        "レース名",
        "Ｒ",
        "race_id",
        "馬名",
        "分類クラス",
        "着順",
        "単勝オッズ",
        "単勝",
        "複勝配当",
        "複勝圏内",
        "着差",
        "馬連",
        "馬単",
        "３連複",
        "３連単",
        "単勝オッズ_実",  # Added
        "複勝圏内_実",  # Added (replaced 複勝配当)
        # "複勝配当", # Removed
    ]

    # ▼▼▼【ここから修正】▼▼▼
    categorical_features = [
        "回り",
        "馬場状態",
        "性別",
        "枠番",
        "騎手",
        "調教師",
        "種牡馬",
        "母父馬",
        "クラスカテゴリ",
        "展開予測",
        "馬番",
        # "running_style_id", # <-- 削除
    ]
    # ▲▲▲【ここまで修正】▲▲▲

    categorical_features = [col for col in categorical_features if col in df.columns]
    LEAKAGE_COLS = [
        # List of columns to exclude as leakage
        "PCI",
        "上り3F",
        "RPCI",
        "平均速度",
        "Ave-3F",
        "平均1Fタイム",
        "走破タイム",
        "1角",
        "2角",
        "3角",
        "4角",
        "3角順位率",
        "4角順位率",
        "PCI3",
        "上り3F_vs_基準",
        "平均速度_vs_基準",
        "PCIvs自己平均",
        "上り3Fvs自己平均",
        "4角順位率vs自己平均",
        "クラスレベルvs自己平均",
        "平均速度vs自己平均",
        "Ave-3Fvs自己平均",
        "上り3F_vs_基準vs自己平均",
        "平均速度_vs_基準vs自己平均",
        "馬体重vs自己平均",
        "世代",
        "前走との上り3F差",
        "前走との平均速度差",
        "騎手_先行割合",
        "騎手_差し割合",
        "騎手_追込割合",
        "騎手_追込割合",
    ]
    numerical_features = [
        col
        for col in df.columns
        if col not in categorical_features
        and col not in meta_cols
        and col not in LEAKAGE_COLS
        and col != target_col  # Exclude target from numerical features
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    all_feature_names = categorical_features + numerical_features

    # --- Feature Classification (Sequence vs Static) ---
    sequence_feature_names = [  # Redefined for LSTM
        # "rank_score", # Target is not an input feature
        "斤量負担率",
        "馬体重増減",
        "前走クラスレベル",
        "クラス昇級フラグ",
        "クラス降級フラグ",
        "キャリア平均_上り3F_vs_基準",
        "キャリア平均_平均速度_vs_基準",
        "近5走平均_上り3F_vs_基準",
        "近5走平均_平均速度_vs_基準",
        "レース間隔",
        "長期休養明けフラグ",
        "連闘フラグ",
        "叩き良化型フラグ",
        "騎手乗り替わりフラグ",
        "騎手",
        "調教師",
        "コンビ歴",
        "キャリア平均_4角順位率",
        "近5走平均_4角順位率",
        "馬体重",
        "キャリア",
    ]
    # Filter sequence features to only those available and not leakage/meta/target
    sequence_feature_names = [
        f
        for f in sequence_feature_names
        if f in df.columns and f in all_feature_names and f != target_col
    ]
    static_feature_names = [
        f
        for f in all_feature_names
        if f not in sequence_feature_names and f != target_col
    ]

    # ▼▼▼【ここから修正】▼▼▼
    # running_style_id が削除されたため、以下のブロックは不要
    # if (
    #     "running_style_id" in df.columns
    #     and "running_style_id" not in static_feature_names
    # ):
    #     static_feature_names.append("running_style_id")
    #     if "running_style_id" in sequence_feature_names:
    #         sequence_feature_names.remove("running_style_id")  # Should not be in both
    # ▲▲▲【ここまで修正】▲▲▲

    # Ensure required static features are included
    required_static = [
        "枠番",
        "斤量",
        "馬場状態",
        "距離",
        "芝・ダ",
        "場所",
        "騎手",
        "調教師",
        "単勝オッズ",
    ]
    for req in required_static:
        if (
            req in df.columns
            and req in all_feature_names
            and req not in static_feature_names
        ):
            static_feature_names.append(req)
            if req in sequence_feature_names:
                sequence_feature_names.remove(req)  # Should not be in both

    print("\n--- 特徴量の最終分類 ---")
    seq_cat_count = len(
        [f for f in sequence_feature_names if f in categorical_features]
    )
    seq_num_count = len([f for f in sequence_feature_names if f in numerical_features])
    static_cat_count = len(
        [f for f in static_feature_names if f in categorical_features]
    )
    static_num_count = len([f for f in static_feature_names if f in numerical_features])
    print(
        f"Sequence Features: {len(sequence_feature_names)} ({seq_cat_count} cat, {seq_num_count} num)"
    )
    print(
        f"Static Features: {len(static_feature_names)} ({static_cat_count} cat, {static_num_count} num)"
    )

    # Define all features needed for the datasets
    all_needed_features = list(
        set(
            sequence_feature_names + static_feature_names + [target_col, "馬名", "日付"]
        )
    )

    # --- Data Splitting ---
    print("\n--- データ分割 (Train/Val/Test) ---")
    # Make sure to select only the necessary columns BEFORE splitting
    df_filtered = df[all_needed_features].copy()
    train_df_full = df_filtered[df_filtered["日付"].dt.year <= 2022].copy()
    val_df_full = df_filtered[df_filtered["日付"].dt.year == 2023].copy()
    test_df_full = df_filtered[df_filtered["日付"].dt.year >= 2024].copy()
    del df_filtered  # Free memory
    print(
        f"Train: {len(train_df_full)}, Val: {len(val_df_full)}, Test: {len(test_df_full)}"
    )

    # --- Scaling and Mapping ---
    print("\n--- スケーリングとマッピングを実行 ---")
    all_mappings = {}
    # Recency-based mapping for Jockey/Trainer
    jockey_to_id_map, id_to_jockey_map = create_recency_based_mapping(df, "騎手")
    trainer_to_id_map, id_to_trainer_map = create_recency_based_mapping(df, "調教師")
    all_mappings["騎手"] = jockey_to_id_map
    all_mappings["調教師"] = trainer_to_id_map

    # Standard mapping for other categoricals
    other_categorical_features = [
        f for f in categorical_features if f not in ["騎手", "調教師"]
    ]
    for col in tqdm(other_categorical_features, desc="  - Other Categorical Encoding"):
        # Create map based on training data only
        categories = (
            train_df_full[col].astype(str).fillna("__UNKNOWN__").unique()
            if col in train_df_full.columns
            else []
        )
        cat_map = {cat: i + 1 for i, cat in enumerate(categories)}  # 1-based index
        all_mappings[col] = cat_map
        # Apply map to all splits
        for d in [train_df_full, val_df_full, test_df_full]:
            if col in d.columns:
                d[col] = (
                    d[col]
                    .astype(str)
                    .fillna("__UNKNOWN__")
                    .map(cat_map)
                    .fillna(0)
                    .astype(int)
                )

    # ▼▼▼【ここから修正】▼▼▼
    # Special handling for running_style_id if exists
    # (running_style_id を削除したため、このブロック全体を削除 L1078-L1089)
    # ▲▲▲【ここまで修正】▲▲▲

    # Apply Jockey/Trainer mappings
    for col in ["騎手", "調教師"]:
        cat_map = all_mappings.get(col)
        if cat_map:
            for d in [train_df_full, val_df_full, test_df_full]:
                if col in d.columns:
                    d[col] = d[col].map(cat_map).fillna(0).astype(int)

    # Scaling static numerical features
    scaler = StandardScaler()
    numerical_features_static_scaled = [
        f for f in static_feature_names if f in numerical_features
    ]
    print(
        f"  - Scaling {len(numerical_features_static_scaled)} static numerical features..."
    )

    # Log transform '単勝オッズ' if it's scaled
    if "単勝オッズ" in numerical_features_static_scaled:
        print("  - Applying log1p transform to 単勝オッズ...")
        for d in [train_df_full, val_df_full, test_df_full]:
            if "単勝オッズ" in d.columns:
                d["単勝オッズ"] = (
                    d["単勝オッズ"].fillna(0).replace([np.inf, -np.inf], 0)
                )
                d["単勝オッズ"] = np.log1p(d["単勝オッズ"])  # Use log1p

    # Fill NaN/Inf before scaling
    for d in [train_df_full, val_df_full, test_df_full]:
        if numerical_features_static_scaled:
            cols_to_fill = [
                c for c in numerical_features_static_scaled if c in d.columns
            ]
            if cols_to_fill:
                d[cols_to_fill] = (
                    d[cols_to_fill].fillna(0).replace([np.inf, -np.inf], 0)
                )

    # Fit scaler ONLY on training data
    if numerical_features_static_scaled and not train_df_full.empty:
        cols_in_train = [
            c for c in numerical_features_static_scaled if c in train_df_full.columns
        ]
        if cols_in_train:
            scaler.fit(train_df_full[cols_in_train])
            # Transform all splits
            for d in [train_df_full, val_df_full, test_df_full]:
                cols_in_d = [
                    c for c in numerical_features_static_scaled if c in d.columns
                ]
                if cols_in_d:
                    d[cols_in_d] = scaler.transform(d[cols_in_d])
            print("  - Static numerical features scaled.")
        else:
            print(
                "  - Warning: No static numerical features found in training data to fit scaler."
            )
    else:
        print(
            "  - Warning: No static numerical features to scale or training data empty."
        )

    # Save preprocessors
    os.makedirs(PREPROCESSOR_BASE_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(PREPROCESSOR_BASE_DIR, "numerical_scaler.pkl"))
    with open(
        os.path.join(PREPROCESSOR_BASE_DIR, "categorical_mappings.json"),
        "w",
        encoding="utf-8",
    ) as f:
        # Convert keys/values to JSON serializable types
        serializable_mappings = {
            k: {str(sub_k): int(sub_v) for sub_k, sub_v in v.items()}
            for k, v in all_mappings.items()
        }
        json.dump(serializable_mappings, f, ensure_ascii=False, indent=4)
    # Save id_to_jockey/trainer maps (optional, useful for interpretation)
    with open(
        os.path.join(PREPROCESSOR_BASE_DIR, "id_to_jockey.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(
            {int(k): v for k, v in id_to_jockey_map.items()},
            f,
            ensure_ascii=False,
            indent=4,
        )
    with open(
        os.path.join(PREPROCESSOR_BASE_DIR, "id_to_trainer.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(
            {int(k): v for k, v in id_to_trainer_map.items()},
            f,
            ensure_ascii=False,
            indent=4,
        )
    # Save feature name lists
    joblib.dump(
        sequence_feature_names,
        os.path.join(PREPROCESSOR_BASE_DIR, "sequence_feature_names.pkl"),
    )
    joblib.dump(
        static_feature_names,
        os.path.join(PREPROCESSOR_BASE_DIR, "static_feature_names.pkl"),
    )
    print(f"  - 前処理オブジェクトを '{PREPROCESSOR_BASE_DIR}' に保存しました。")

    # --- Define Feature Columns for DeepCTR ---
    # (This section defines all_feature_columns based on mappings and numerical_features)
    all_feature_columns = []
    processed_sparse_names = set()

    for feat_name, cat_map in all_mappings.items():
        # Make sure the feature is actually used AND not already added
        if feat_name in all_feature_names and feat_name not in processed_sparse_names:
            vocab_size = len(cat_map) + 1  # +1 for unknown (index 0)

            # ▼▼▼【ここから修正】▼▼▼
            # (running_style_id の特別処理ブロックを削除 L1182-L1187)
            # ▲▲▲【ここまで修正】▲▲▲

            all_feature_columns.append(
                SparseFeat(
                    feat_name, vocabulary_size=vocab_size, embedding_dim=EMBEDDING_DIM
                )
            )
            processed_sparse_names.add(feat_name)

    # Add DenseFeat for all relevant numerical features
    for feat_name in numerical_features:
        if feat_name in all_feature_names:  # Check if it's in the final list
            all_feature_columns.append(DenseFeat(feat_name, 1))

    print("\n--- 生成された全特徴量定義 (一部) ---")
    # Debug print
    if len(all_feature_columns) > 10:
        for fc in all_feature_columns[:5]:
            print(f"{fc.name}: {type(fc)}")
        print("...")
        for fc in all_feature_columns[-5:]:
            print(f"{fc.name}: {type(fc)}")
    else:
        for fc in all_feature_columns:
            print(f"{fc.name}: {type(fc)}")

    # --- Save Feature Columns Definition ---
    # (Save it here after defining it)
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    joblib.dump(
        all_feature_columns, os.path.join(MODEL_OUTPUT_DIR, "feature_columns.pkl")
    )
    print(
        f"--- 特徴量カラム定義を {MODEL_OUTPUT_DIR}/feature_columns.pkl に保存しました ---"
    )

    # --- Separate Feature Columns for Model Instantiation ---
    # (Define static_fcs, sequence_fcs etc. directly from all_feature_columns)
    try:
        if not all_feature_columns:  # Check if list is empty
            raise ValueError("all_feature_columns is empty after definition.")

        # Use the feature name lists loaded/defined earlier
        static_fcs = [
            fc for fc in all_feature_columns if fc.name in static_feature_names
        ]
        static_linear_fcs = static_fcs  # Assign to required variable
        static_dnn_fcs = static_fcs  # Assign to required variable
        print(f"--- Static features ({len(static_fcs)}件) を特定しました ---")

        # Reconstruct sequence_fcs (Sparse + Dense)
        # all_feature_columns (正しい順序) から sequence_feature_names (セット) に
        # 含まれるものだけを、元の順序を維持して抽出する
        # (predict_logic.py, preprocess_sequences.py とロジックを統一)

        # L1017あたりでロードされた sequence_feature_names (リスト) をセットに変換
        if "sequence_feature_names" not in locals() or not isinstance(
            sequence_feature_names, list
        ):
            # 万が一ロードされていない場合はロードする
            print(
                "[警告] sequence_feature_names が見つからないため、pklから再ロードします。"
            )
            sequence_feature_names_set = set(
                joblib.load(
                    os.path.join(PREPROCESSOR_BASE_DIR, "sequence_feature_names.pkl")
                )
            )
        else:
            sequence_feature_names_set = set(sequence_feature_names)

        sequence_fcs = [
            fc for fc in all_feature_columns if fc.name in sequence_feature_names_set
        ]

        # ターゲット列が誤って含まれていないか確認 (念のため)
        sequence_fcs = [fc for fc in sequence_fcs if fc.name != target_col]

        print(
            f"--- Sequence features ({len(sequence_fcs)}件) を特定しました (順序補正済) ---"
        )

        # ▼▼▼【 デバッグプリント追加 2/2 】▼▼▼
        print("\n--- [DEBUG] Features used for CURRENT model definition ---")
        print("  Sequence Features (used for LSTM input):")
        if not sequence_fcs:
            print("    (None)")
        else:
            for i, fc in enumerate(sequence_fcs):
                print(f"    {i+1:2d}. {fc.name} ({type(fc).__name__})")
        # ▲▲▲【 デバッグプリント追加ここまで 】▲▲▲

        # Ensure lists are not empty if they shouldn't be
        if not static_fcs:
            print("警告: static_fcs が空です。")
        if not sequence_fcs:
            print("警告: sequence_fcs が空です。")

        # Order static features (for dataset, though less critical now with NPZ)
        static_feature_names_ordered = [fc.name for fc in static_fcs]

    except NameError as ne:
        print(f"エラー: 特徴量リストの分離中に未定義の変数があります: {ne}")
        print(
            "sequence_feature_names や static_feature_names が正しくロード/定義されているか確認してください。"
        )
        return
    except Exception as e:
        print(f"エラー: 特徴量リストの分離中に予期せぬエラーが発生: {e}")
        import traceback

        traceback.print_exc()
        return

    # --- データセットとデータローダー (NPZからロード) ---
    print("\n--- 事前処理済みデータ (NPZ) をロード中 ---")
    NPZ_PATH = "preprocessed_lstm_data.npz"

    train_loader = None
    val_loader = None
    current_batch_size = BATCH_SIZE

    # ▼▼▼【SHAP用に変数を外で定義】▼▼▼
    npz_static_names_ordered = []
    npz_seq_names_ordered = []
    # ▲▲▲【ここまで】▲▲▲

    try:
        data = np.load(NPZ_PATH, allow_pickle=True)
        X_seq_all = data["X_seq"]
        X_static_all = data["X_static"]
        y_all = data["y"]

        if "日付" not in data:
            raise KeyError(
                "NPZファイルに '日付' メタデータが見つかりません。preprocess_sequences.py を確認してください。"
            )

        # ▼▼▼【 ここから修正 】▼▼▼
        try:
            dates_all_np = data["日付"]
            parsed_dates = None
            if dates_all_np.dtype == "O":
                # オブジェクト型の場合、バイト列かもしれないのでデコード試行
                try:
                    # Check if the first element is bytes
                    if isinstance(dates_all_np[0], bytes):
                        dates_all_str = np.vectorize(lambda x: x.decode("utf-8"))(
                            dates_all_np
                        )
                        parsed_dates = pd.to_datetime(dates_all_str)
                    else:  # Assume it's already strings or compatible objects
                        parsed_dates = pd.to_datetime(dates_all_np)
                except Exception:  # Fallback if decoding/parsing object array fails
                    raise ValueError(
                        "NPZ内の日付データ(オブジェクト型)の形式が不正です。"
                    )
            else:
                # オブジェクト型以外なら直接変換
                parsed_dates = pd.to_datetime(dates_all_np)

            # --- ★ pd.Series でラップする ---
            dates_all = pd.Series(parsed_dates)
            if dates_all.isnull().any():
                print("[警告] 日付データのパース結果にNullが含まれています。")

        except Exception as date_e:
            print(f"日付データのパースエラー: {date_e}")
            raise ValueError("NPZ内の日付データの形式が不正です。")
        # ▲▲▲【 ここまで修正 】▲▲▲

        print(
            f"  ロード成功: X_seq={X_seq_all.shape}, X_static={X_static_all.shape}, y={y_all.shape}"
        )

        # --- NPZデータの日付に基づいて Train/Validation 分割 ---
        train_year_end = 2022
        val_year = 2023

        # ここで dates_all.dt.year が使えるようになるはず
        train_mask = dates_all.dt.year <= train_year_end
        val_mask = dates_all.dt.year == val_year

        train_X_seq = X_seq_all[train_mask]
        train_X_static = X_static_all[train_mask]
        train_y = y_all[train_mask]

        val_X_seq = X_seq_all[val_mask]
        val_X_static = X_static_all[val_mask]
        val_y = y_all[val_mask]

        print(f"  Train データ数: {len(train_y)}")
        print(f"  Validation データ数: {len(val_y)}")

        # ☆☆☆ 分割後のデータ存在チェック ☆☆☆
        if len(train_y) == 0 or len(val_y) == 0:
            print("エラー: 学習データまたは検証データの数が0です。")
            print(
                f"NPZファイルに含まれるデータの期間と、設定された分割年({train_year_end}年以前、{val_year}年)を確認してください。"
            )
            # ☆☆☆ loader を None のままにする ☆☆☆

        # ☆☆☆ データが存在する場合のみ TensorDataset と DataLoader を作成 ☆☆☆
        else:
            print("\n--- 高速DataLoaderを作成中 (NPZデータから) ---")
            train_X_seq_tensor = torch.tensor(train_X_seq, dtype=torch.float32)
            train_X_static_tensor = torch.tensor(train_X_static, dtype=torch.float32)
            train_y_tensor = torch.tensor(train_y, dtype=torch.float32)

            val_X_seq_tensor = torch.tensor(val_X_seq, dtype=torch.float32)
            val_X_static_tensor = torch.tensor(val_X_static, dtype=torch.float32)
            val_y_tensor = torch.tensor(val_y, dtype=torch.float32)

            train_tensor_dataset = TensorDataset(
                train_X_seq_tensor, train_X_static_tensor, train_y_tensor
            )
            val_tensor_dataset = TensorDataset(
                val_X_seq_tensor, val_X_static_tensor, val_y_tensor
            )

            # current_batch_size は try の前で定義済み
            num_workers = 4
            train_loader = DataLoader(
                train_tensor_dataset,
                batch_size=current_batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True if device == "cuda" else False,
            )
            val_loader = DataLoader(
                val_tensor_dataset,
                batch_size=current_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True if device == "cuda" else False,
            )

        # ▼▼▼【 ここから検証コード（修正版）を追加 】▼▼▼
        print("\n--- データ内のカテゴリIDと語彙サイズの整合性をチェック ---")
        try:
            # ▼▼▼【SHAP用にグローバルに代入】▼▼▼
            npz_static_names_ordered = data["static_feature_names_ordered"].tolist()
            npz_seq_names_ordered = data["sequence_feature_names_ordered"].tolist()
            # ▲▲▲【ここまで】▲▲▲

            # モデル定義で使用する SparseFeat のリストを取得
            sparse_features_static_list = [
                fc for fc in static_fcs if isinstance(fc, SparseFeat)
            ]
            sparse_features_seq_list = [
                fc for fc in sequence_fcs if isinstance(fc, SparseFeat)
            ]

            valid_static_train = True
            valid_seq_train = True

            # --- 静的データ (Train) のチェック ---
            print("  静的データ(Train)の最大ID vs 語彙サイズ:")
            if train_X_static_tensor.shape[1] == 0:
                print("    (静的テンソルが空です)")
            else:
                for fc in sparse_features_static_list:
                    feature_name = fc.name
                    if feature_name in npz_static_names_ordered:
                        try:
                            # npzの順序リストから正しい列インデックスを取得
                            tensor_col_idx = npz_static_names_ordered.index(
                                feature_name
                            )

                            if tensor_col_idx < train_X_static_tensor.shape[1]:
                                # 該当列の最大IDを取得
                                max_id = torch.max(
                                    train_X_static_tensor[:, tensor_col_idx]
                                ).item()
                                vocab_size = fc.vocabulary_size
                                if max_id >= vocab_size:
                                    print(
                                        f"    [エラー] {feature_name}: 最大ID={int(max_id)}, 語彙サイズ={vocab_size} (不正!)"
                                    )
                                    valid_static_train = False
                                # else:
                                #     print(f"    OK: {feature_name}: 最大ID={int(max_id)}, 語彙サイズ={vocab_size}")
                            else:
                                print(
                                    f"    [警告] {feature_name}: 列インデックス({tensor_col_idx})がテンソル形状外({train_X_static_tensor.shape[1]})"
                                )
                                valid_static_train = False  # Treat as error
                        except ValueError:
                            # This case means the feature name exists in static_fcs but not in npz_static_names_ordered
                            print(
                                f"    [エラー] {feature_name}: モデルは期待しているが、NPZの順序リストに存在しません。"
                            )
                            valid_static_train = False
                        except Exception as e_inner:
                            print(
                                f"    [エラー] {feature_name} のチェック中に予期せぬエラー: {e_inner}"
                            )
                            valid_static_train = False
                    else:
                        # This case means the feature name exists in static_fcs but not in npz_static_names_ordered (same as ValueError case)
                        print(
                            f"    [エラー] {feature_name}: モデルは期待しているが、NPZの順序リストに存在しません。"
                        )
                        valid_static_train = False

            # --- シーケンスデータ (Train) のチェック ---
            print("  シーケンスデータ(Train)の最大ID vs 語彙サイズ:")
            if train_X_seq_tensor.shape[2] == 0:
                print("    (シーケンステンソルの特徴量次元が0です)")
            else:
                for fc in sparse_features_seq_list:
                    feature_name = fc.name
                    if feature_name in npz_seq_names_ordered:
                        try:
                            # npzの順序リストから正しい特徴量インデックスを取得
                            tensor_feat_idx = npz_seq_names_ordered.index(feature_name)

                            if tensor_feat_idx < train_X_seq_tensor.shape[2]:
                                # 該当次元の最大IDを取得
                                max_id = torch.max(
                                    train_X_seq_tensor[:, :, tensor_feat_idx]
                                ).item()
                                vocab_size = fc.vocabulary_size
                                if max_id >= vocab_size:
                                    print(
                                        f"    [エラー] {feature_name}: 最大ID={int(max_id)}, 語彙サイズ={vocab_size} (不正!)"
                                    )
                                    valid_seq_train = False
                                # else:
                                #     print(f"    OK: {feature_name}: 最大ID={int(max_id)}, 語彙サイズ={vocab_size}")
                            else:
                                print(
                                    f"    [警告] {feature_name}: 特徴量インデックス({tensor_feat_idx})がテンソル形状外({train_X_seq_tensor.shape[2]})"
                                )
                                valid_seq_train = False  # Treat as error
                        except ValueError:
                            print(
                                f"    [エラー] {feature_name}: モデルは期待しているが、NPZの順序リストに存在しません。"
                            )
                            valid_seq_train = False
                        except Exception as e_inner:
                            print(
                                f"    [エラー] {feature_name} のチェック中に予期せぬエラー: {e_inner}"
                            )
                            valid_seq_train = False
                    else:
                        print(
                            f"    [エラー] {feature_name}: モデルは期待しているが、NPZの順序リストに存在しません。"
                        )
                        valid_seq_train = False

            # 同様に検証データ (val_) もチェック (省略可)
            # ...

            if not valid_static_train or not valid_seq_train:
                print(
                    "\nエラー: データ内に不正なカテゴリID、またはモデル定義とNPZデータの列/特徴量順序に不整合があります。学習を中止します。"
                )
                print(
                    "1. preprocess_sequences.py が最新の *_feature_names.pkl を使って NPZ を生成したか確認してください。"
                )
                print(
                    "2. keiba_analysis.py が最新の feature_columns.pkl と *_feature_names.pkl を使ってモデルを定義しているか確認してください。"
                )
                print(
                    "   (特に all_mappings, sequence_feature_names, static_feature_names のロード)"
                )
                print(
                    "3. 上記を確認後、前処理ファイルとNPZファイルを再生成してください (クリーンアップ -> keiba前半実行 -> preprocess実行 -> keiba再実行)。"
                )
                return  # エラーが見つかったら終了
            else:
                print("  カテゴリIDと列/特徴量順序は正常です。")

        except KeyError as e:
            print(
                f"カテゴリIDの検証エラー: NPZファイルに必要なキー '{e}' (列名リスト) がありません。"
            )
            print(
                "preprocess_sequences.py が正しく *_feature_names_ordered を保存しているか確認してください。"
            )
            return
        except Exception as e:
            print(f"カテゴリIDの検証中に予期せぬエラーが発生: {e}")
            import traceback

            traceback.print_exc()
            # return # Keep going or stop? Stop for now.
            return
        # ▲▲▲【 ここまで検証コード（修正版）を追加 】▲▲▲

        # メモリ解放 (検証用に追加)
        # del train_X_seq_tensor, train_X_static_tensor # Keep them for DataLoader
        # del val_X_seq_tensor, val_X_static_tensor
        gc.collect()

    except FileNotFoundError:
        print(f"エラー: {NPZ_PATH} が見つかりません。")
        print(
            "まず preprocess_sequences.py を実行して NPZ ファイルを作成してください。"
        )
        return  # NPZがなければ終了
    except KeyError as e:
        print(f"エラー: NPZファイルに必要なキー '{e}' が見つかりません。")
        return  # 必要なキーがなければ終了
    except Exception as e:
        print(f"NPZファイルの読み込みまたはデータローダー作成エラー: {e}")
        import traceback

        traceback.print_exc()
        return  # その他のエラーでも終了

    # ☆☆☆ DataLoader が作成されたかチェック ☆☆☆
    if train_loader is None or val_loader is None:
        print("--- DataLoader の作成に失敗したため、学習を実行できません。---")
        return
    # ☆☆☆ ここまで ☆☆☆

    # --- モデル定義 ---
    print("\n--- LSTM Hybrid モデルを定義 ---")
    try:
        model = DCN_LSTM_Hybrid(
            static_linear_feature_columns=static_linear_fcs,
            static_dnn_feature_columns=static_dnn_fcs,
            sequence_feature_columns=sequence_fcs,
            static_feature_names=static_feature_names,
            lstm_hidden_size=LSTM_HIDDEN_SIZE,
            lstm_layers=LSTM_LAYERS,
            lstm_dropout=LSTM_DROPOUT,
            cross_num=CROSS_NUM,
            dnn_hidden_units=DNN_HIDDEN_UNITS,
            l2_reg_linear=L2_REG,
            l2_reg_embedding=L2_REG,
            l2_reg_cross=L2_REG,
            l2_reg_dnn=L2_REG,
            dnn_dropout=DROPOUT_RATE,
            dnn_use_bn=True,  # <-- ★★★【前回の修正】inplaceエラー回避のため True に変更 ★★★
            init_std=0.0001,
            seed=1024,
            task="regression",
            device=device,
        ).to(device)
    except NameError as ne:
        print(f"モデル定義エラー: 必要な特徴量リストが定義されていません: {ne}")
        print("特徴量リストの分離ステップでエラーが発生していないか確認してください。")
        return
    except Exception as e:
        print(f"モデル定義エラー: {e}")
        import traceback

        traceback.print_exc()
        return

    # ▼▼▼【ここから RUN_SHAP_ONLY ロジックを適用】▼▼▼

    # --- 学習実行 or スキップ ---
    # (RUN_SHAP_ONLY はグローバル設定で定義されている前提)
    try:
        if RUN_SHAP_ONLY:
            print("\n--- [SHAP ONLY MODE] 学習をスキップします ---")
            model_path = os.path.join(MODEL_OUTPUT_DIR, "lstm_hybrid_model.pth")
            if not os.path.exists(model_path):
                print(f"エラー: {model_path} が見つかりません。")
                print("RUN_SHAP_ONLY=False にして、先に学習を実行してください。")
                return

            print(f"--- {model_path} から学習済み重みをロード中 ---")
            try:
                print("--- [DEBUG] Attempting to load state_dict... ---")
                # SHAPに必要な `best_model_state` を .pth ファイルからロード
                best_model_state = torch.load(
                    model_path, map_location=device, weights_only=True
                )
                print(
                    "--- [DEBUG] state_dict loaded successfully (before applying to model). ---"
                )
                success = True  # SHAP実行フラグを立てる
            except Exception as e:
                print(f"モデルのロードに失敗: {e}")
                return

        else:
            # 既存の学習ロジック
            print("\n--- [学習モード] モデルの学習処理を開始 ---")
            success, metrics, best_model_state = train_hybrid_model(
                train_loader,
                val_loader,
                model,
                device,
                MODEL_OUTPUT_DIR,
                current_batch_size,
            )
    except NameError:
        print("エラー: グローバル設定に 'RUN_SHAP_ONLY' フラグが定義されていません。")
        print(
            "ファイルの先頭に RUN_SHAP_ONLY = True (または False) を追加してください。"
        )
        return

    # --- DataLoaderの破棄 (SHAP実行前のメモリ確保) ---
    print("\n--- (SHAP実行準備) 学習/予測用 DataLoader を破棄 ---")
    try:
        del train_loader
        del val_loader
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    except NameError:
        print("  - DataLoader は (SHAP ONLY MODE のため) 定義されていません。")
    except Exception as e:
        print(f"DataLoader の破棄中にエラー: {e}")

    # --- 結果保存 & SHAP実行 ---
    if success and best_model_state:

        # ▼▼▼【修正】学習実行時のみモデルを保存する▼▼▼
        if not RUN_SHAP_ONLY:
            model_path = os.path.join(MODEL_OUTPUT_DIR, "lstm_hybrid_model.pth")
            try:
                torch.save(best_model_state, model_path)
                print(f"学習済みモデルを {model_path} に保存しました。")
            except Exception as e:
                print(f"モデルの保存に失敗: {e}")
        else:
            print("\n--- [SHAP ONLY MODE] モデルの保存をスキップしました ---")
        # ▲▲▲【修正ここまで】▲▲▲

        # ★★★【注意】SHAPCompatibleWrapper の定義が main() の中にあります ★★★
        # ★★★ これを main() の外 (DCN_LSTM_Hybridクラス定義の後) に移動する必要があります ★★★
        # class SHAPCompatibleWrapper(nn.Module): ... (L1585-L1605)
        # (↑ このコードブロックからは削除されている前提で進めます)

        # ▼▼▼【ここからSHAP実装ブロック (全面的に修正)】▼▼▼
        try:
            print("\n--- SHAPによる特徴量重要度の計算を開始 ---")

            # 1. SHAP実行用のモデルに重みをロード
            model.load_state_dict(
                best_model_state
            )  # (学習直後 or .pth からロードした重み)

            # 2. SHAP用のデータをNPZから再ロード
            print("  - SHAP用データをNPZから再ロード中 (num_workers=0)...")
            data = np.load(NPZ_PATH, allow_pickle=True)

            # (中略: NPZの日付パース、マスク作成、データサンプリング ... L1623～L1658)
            try:
                dates_all_np = data["日付"]
                parsed_dates = None
                if dates_all_np.dtype == "O":
                    if isinstance(dates_all_np[0], bytes):
                        dates_all_str = np.vectorize(lambda x: x.decode("utf-8"))(
                            dates_all_np
                        )
                        parsed_dates = pd.to_datetime(dates_all_str)
                    else:
                        parsed_dates = pd.to_datetime(dates_all_np)
                else:
                    parsed_dates = pd.to_datetime(dates_all_np)
                dates_all = pd.Series(parsed_dates)
            except Exception as date_e:
                print(f"SHAP用データの日付パースエラー: {date_e}, 処理を中断します。")
                raise date_e

            train_year_end = 2022
            val_year = 2023
            train_mask = dates_all.dt.year <= train_year_end
            val_mask = dates_all.dt.year == val_year

            # 訓練データから50サンプル (背景データ)
            train_indices = np.where(train_mask)[0]
            if len(train_indices) == 0:
                print("エラー: SHAPの背景データ（訓練データ）が0件です。")
                raise ValueError("SHAP background data is empty.")
            np.random.seed(42)
            bg_sample_indices = np.random.choice(
                train_indices, size=min(50, len(train_indices)), replace=False
            )

            X_seq_bg = torch.tensor(
                data["X_seq"][bg_sample_indices], dtype=torch.float32
            ).to(device)
            X_static_bg = torch.tensor(
                data["X_static"][bg_sample_indices], dtype=torch.float32
            ).to(device)

            # 検証データから30サンプル (SHAP計算対象)
            val_indices = np.where(val_mask)[0]
            if len(val_indices) == 0:
                print("エラー: SHAPのテストデータ（検証データ）が0件です。")
                raise ValueError("SHAP test data is empty.")
            test_sample_indices = np.random.choice(
                val_indices, size=min(30, len(val_indices)), replace=False
            )

            X_seq_test = torch.tensor(
                data["X_seq"][test_sample_indices], dtype=torch.float32
            ).to(device)
            X_static_test = torch.tensor(
                data["X_static"][test_sample_indices], dtype=torch.float32
            ).to(device)

            # メモリ解放
            del data
            gc.collect()

            print(f"  - 背景データ: {X_seq_bg.shape[0]}サンプル")
            print(f"  - テストデータ: {X_seq_test.shape[0]}サンプル")

            # 3. SHAPラッパーモデルを作成 (クラス定義はトップレベルに移動済みと仮定)
            shap_model = SHAPCompatibleWrapper(model)

            # 4. GradientExplainerを使用
            print("  - GradientExplainer初期化中...")
            explainer = shap.GradientExplainer(shap_model, [X_seq_bg, X_static_bg])

            # 5. shap_values() の直前に train() モードを設定
            print("  - モデルを SHAP 用の train() モードに設定中...")
            shap_model.train()
            # ただし、Dropout は無効 (eval) にする
            for module in shap_model.modules():
                if isinstance(module, nn.Dropout):
                    module.eval()
                if isinstance(module, nn.ReLU):
                    module.inplace = False  # (inplaceエラー回避)

            # 6. SHAP値を計算
            print("  - SHAP値を計算中...")
            shap_values = explainer.shap_values([X_seq_test, X_static_test])

            shap_values_seq = shap_values[0]  # Shape (samples, steps, feats, 1)
            shap_values_static = shap_values[1]  # Shape (samples, feats, 1)

            print(f"  - 静的SHAP値 Shape: {shap_values_static.shape}")
            print(f"  - シーケンスSHAP値 Shape: {shap_values_seq.shape}")

            # 7. SHAP値の集計 (Mean Absolute SHAP)
            print("  - 特徴量重要度を集計中 (Mean Absolute SHAP)...")

            # (samples, feats, 1) -> (samples, feats) -> (feats)
            # .squeeze() で末尾の '1' の次元を削除
            mean_abs_shap_static = np.mean(
                np.abs(np.squeeze(shap_values_static, axis=-1)), axis=0
            )

            # (samples, steps, feats, 1) -> (samples, steps, feats) -> (feats)
            # 時間(steps)とサンプル(samples)の両方で平均
            mean_abs_shap_seq = np.mean(
                np.abs(np.squeeze(shap_values_seq, axis=-1)), axis=(0, 1)
            )

            # 特徴量名と辞書にまとめる
            # (npz_static_names_ordered, npz_seq_names_ordered は main の前半でロード済み)
            static_importance = dict(
                zip(npz_static_names_ordered, mean_abs_shap_static)
            )
            seq_importance = dict(zip(npz_seq_names_ordered, mean_abs_shap_seq))

            # 総合的な重要度
            all_importance = {**static_importance, **seq_importance}

            # 8. 降順で表示
            print("\n" + "=" * 50)
            print("     SHAP 特徴量重要度 (Mean Absolute Value)     ")
            print("=" * 50)

            sorted_importance = sorted(
                all_importance.items(), key=lambda item: item[1], reverse=True
            )

            for feature, importance in sorted_importance:
                # :<30 で特徴量名を左揃え30文字幅に
                print(f"  {feature:<30}: {importance:.6f}")

            print("=" * 50)

            # 9. (オプション) データをファイルに保存
            try:
                # NumPy の float32 を Python の float に変換
                importance_dict_serializable = {
                    k: float(v) for k, v in all_importance.items()
                }
                output_json_path = os.path.join(
                    MODEL_OUTPUT_DIR, "shap_importance.json"
                )
                with open(output_json_path, "w", encoding="utf-8") as f:
                    json.dump(
                        importance_dict_serializable, f, ensure_ascii=False, indent=4
                    )
                print(f"  - SHAP重要度を {output_json_path} に保存しました。")
            except Exception as e:
                print(f"  - SHAP重要度JSONの保存に失敗: {e}")

            # 10. プロット関連のコードは削除

            # メモリ解放
            del explainer, shap_values, X_seq_bg, X_static_bg, X_seq_test, X_static_test
            del all_importance, static_importance, seq_importance, sorted_importance
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            print("--- SHAPによる特徴量重要度の計算が完了 ---")

        except ImportError:
            print(
                "\n--- [警告] SHAP がインストールされていません。`pip install shap` を実行してください。 ---"
            )
        except Exception as e:
            print(f"\n--- [エラー] SHAPの実行中にエラーが発生しました: {e} ---")
            import traceback

            traceback.print_exc()
        # ▲▲▲【SHAP実装ここまで】▲▲▲

        # ▼▼▼【修正】学習実行時のみサマリーを保存する▼▼▼
        summary_json_path = SUMMARY_JSON_FILENAME
        if not RUN_SHAP_ONLY:
            try:
                # Load existing summary if it exists
                if os.path.exists(summary_json_path):
                    with open(summary_json_path, "r", encoding="utf-8") as f:
                        try:
                            summary_data = json.load(f)
                            if not isinstance(summary_data, list):
                                summary_data = []
                        except json.JSONDecodeError:
                            summary_data = []
                else:
                    summary_data = []

                # Append new results
                if "metrics" in locals() and metrics:  # Check if metrics exists
                    summary_data.append(
                        {
                            "condition": "LSTM_Hybrid_Model",
                            "best_blend_ratio": 1.0,
                            **metrics,
                        }
                    )
                    # Save updated summary
                    with open(summary_json_path, "w", encoding="utf-8") as f:
                        json.dump(summary_data, f, ensure_ascii=False, indent=4)
                    print(
                        f"学習結果のサマリーを {summary_json_path} に追記・保存しました。"
                    )
                else:
                    print(
                        "警告: metrics が None のためサマリーを保存できませんでした。"
                    )

            except Exception as e:
                print(f"サマリーJSONの保存に失敗: {e}")
        else:
            print("\n--- [SHAP ONLY MODE] サマリーの保存をスキップしました ---")
        # ▲▲▲【修正ここまで】▲▲▲

    elif not RUN_SHAP_ONLY:
        # RUN_SHAP_ONLY でない (学習モードだった) が失敗した場合
        print("学習が成功しなかったため、モデルとサマリーは保存されません。")

    print(
        "\n\n"
        + "=" * 50
        + "\n🎉 LSTM Hybrid モデルの処理が完了しました 🎉\n"
        + "=" * 50
    )
    # ▼▼▼【修正】metrics変数が存在するかチェック▼▼▼
    if success and "metrics" in locals() and metrics:
        print(f"最終結果: {metrics}")
    elif success:
        print("処理は完了しましたが、メトリクスがありません (SHAP ONLY MODE)。")
    else:
        print("学習に失敗しました。")


if __name__ == "__main__":
    main()
