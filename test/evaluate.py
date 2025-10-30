import pandas as pd
import numpy as np
import torch
import torch.nn as nn
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
    # build_input_features, # 不要
    combined_dnn_input,
)
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.layers import CrossNet, DNN

# from sklearn.preprocessing import StandardScaler # 不要
# from sklearn.cluster import KMeans # 不要
import gc

# import types # 不要
# import polars as pl # 不要
import torch.nn.functional as F

# from collections import defaultdict # 不要
import glob

# --- グローバル設定 ---
PREPROCESSED_DATA_PATH = "preprocessed_lstm_data.npz"  # preprocess_sequences.py の出力
PREPROCESSOR_BASE_DIR = "lstm_preprocessor_score"  # マッピング情報ロード用
MODEL_OUTPUT_DIR = "lstm_models_score"  # モデルロード用
OUTPUT_DIR = "evaluation_results_lstm"  # 評価結果の出力先

# --- モデル構造パラメータ (ロードするモデルと合わせる) ---
EMBEDDING_DIM = 8
CROSS_NUM = 3
DNN_HIDDEN_UNITS = (256, 128)
N_STEPS = 5  # preprocessed_lstm_data.npz の N_STEPS と一致させる
LSTM_HIDDEN_SIZE = 64
LSTM_LAYERS = 1
LSTM_DROPOUT = 0.2
# CLUSTER_N = 6 # 不要
# RUN_CLUSTERING = True # 不要

# --- 評価用パラメータ ---
BATCH_SIZE = 256
NUM_WORKERS = 0  # .npz からのロードは高速なので 0 で良い
TARGET_COL = "rank_score"  # npz ファイル内の 'y' に対応

# ▼▼▼【修正】回収率計算に使う列を「複勝配当」に統一 ▼▼▼
ODDS_COL = "複勝圏内_実"  # 複勝の配当金そのものを使う
WIN_FLAG_COL = "複勝圏内_実"  # 複勝の配当金 (0より大きければ的中)
# ▲▲▲【修正ここまで】▲▲▲

RACE_ID_COL = "race_id"
HORSE_NAME_COL = "馬名"

# ▼▼▼【追加】NPZからロードするメタデータ列のリスト ▼▼▼
META_COLS_FOR_EVAL = [
    RACE_ID_COL,
    HORSE_NAME_COL,
    ODDS_COL,
    WIN_FLAG_COL,
    "日付",  # 評価スクリプトでのフィルタリング用
    "着順",  # ランキング精度計算用
    "馬番",  # ランキング精度計算用
    "単勝",  # 戦略計算用
    "単勝オッズ",  # 戦略計算用
    # 注意: preprocess_sequences.py の META_COLS_FOR_EVAL にもこれらが含まれている必要があります
]
# ▲▲▲【追加ここまで】▲▲▲


# === LSTM Hybrid モデルクラス定義 (ここから) ===
# (keiba_analysis.py から DCN_LSTM_Hybrid クラス定義をそのままコピー)
# (変更なし、内容は省略)
class DCN_LSTM_Hybrid(BaseModel):
    def __init__(
        self,
        static_linear_feature_columns,
        static_dnn_feature_columns,
        sequence_feature_columns,
        # static_feature_names, # __init__ では不要
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

        # self.static_feature_names = static_feature_names # 不要
        self.sequence_feature_columns = sequence_feature_columns
        self.static_linear_feature_columns = static_linear_feature_columns
        self.static_dnn_feature_columns = static_dnn_feature_columns
        # self.dnn_hidden_units = dnn_hidden_units # DNN内で使うので不要

        # --- LSTM部分 ---
        seq_sparse_feature_columns = [
            fc for fc in sequence_feature_columns if isinstance(fc, SparseFeat)
        ]
        seq_dense_feature_columns = [
            fc for fc in sequence_feature_columns if isinstance(fc, DenseFeat)
        ]

        # embedding_dict が super().__init__ で作成されるのを待つ
        lstm_input_size = sum(
            self.embedding_dict[fc.embedding_name].embedding_dim
            for fc in seq_sparse_feature_columns
            if fc.embedding_name in self.embedding_dict
        ) + len(seq_dense_feature_columns)

        if lstm_input_size == 0:
            print(
                "警告: LSTM input_size が 0 です。シーケンス特徴量が正しく定義されていません。"
            )
            # Fallback to a small dummy size to avoid LSTM error if no seq features
            lstm_input_size = 1  # Or handle appropriately

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
        )

        # --- DCN部分 ---
        static_dnn_input_dim = self.compute_input_dim(static_dnn_feature_columns)
        # Handle case where static_dnn_input_dim might be 0
        self.dnn_input_dim = (
            static_dnn_input_dim + lstm_hidden_size
            if static_dnn_input_dim > 0
            else lstm_hidden_size
        )

        if self.dnn_input_dim == 0:
            print(
                "警告: DNN input_dim が 0 です。静的特徴量もシーケンス特徴量もありません。"
            )
            # Add dummy layers if input is zero? Or raise error?
            # For now, create DNN but it might fail in forward if input is truly zero
            self.dnn = nn.Identity()  # Placeholder
            self.dnn_linear = nn.Linear(
                lstm_hidden_size if lstm_hidden_size > 0 else 1, 1, bias=False
            ).to(device)
        else:
            self.dnn = DNN(
                self.dnn_input_dim,
                dnn_hidden_units,
                activation=dnn_activation,
                l2_reg=l2_reg_dnn,
                dropout_rate=dnn_dropout,
                use_bn=dnn_use_bn,
                init_std=init_std,
                device=device,
            )
            dnn_output_dim = (
                dnn_hidden_units[-1] if dnn_hidden_units else self.dnn_input_dim
            )
            self.dnn_linear = nn.Linear(dnn_output_dim, 1, bias=False).to(device)

        self.cross_input_dim = self.compute_input_dim(static_linear_feature_columns)
        if self.cross_input_dim > 0:
            self.crossnet = CrossNet(self.cross_input_dim, cross_num, device=device)
            self.cross_linear = nn.Linear(self.cross_input_dim, 1, bias=False).to(
                device
            )
        else:
            # Handle case where there are no linear features for CrossNet
            print("情報: CrossNet の入力次元が 0 のため、CrossNet は無効化されます。")
            self.crossnet = None
            self.cross_linear = None

        # self.add_regularization_weight(self.embedding_dict, l2_reg_embedding) # BaseModelがやるはず
        # self.add_regularization_weight(self.dnn.linears, l2_reg_dnn) # BaseModelがやるはず
        # self.add_regularization_weight(self.crossnet.kernels, l2_reg_cross) # BaseModelがやるはず? 要確認
        # self.add_regularization_weight(self.linear.weight, l2_reg_linear) # BaseModelがやるはず

        self.to(device)

    # (forwardメソッドは keiba_analysis.py からコピーしたまま、変更なし)
    # (内容は省略)
    def forward(self, X_seq_tensor, X_static_tensor):
        batch_size = (
            X_static_tensor.shape[0]
            if X_static_tensor is not None and X_static_tensor.nelement() > 0
            else (
                X_seq_tensor.shape[0]
                if X_seq_tensor is not None and X_seq_tensor.nelement() > 0
                else 0
            )
        )
        if batch_size == 0:
            # Handle empty batch case
            return torch.zeros((0, 1)).to(self.device)

        # --- LSTM入力の準備 ---
        seq_sparse_embedding_list = []
        seq_dense_value_list = []
        current_seq_col_idx = 0
        # sequence_feature_columns の順序でループし、テンソル列インデックスを取得
        seq_feat_map = {
            fc.name: i for i, fc in enumerate(self.sequence_feature_columns)
        }

        if X_seq_tensor is not None and X_seq_tensor.nelement() > 0:
            for fc in self.sequence_feature_columns:
                try:
                    current_seq_col_idx = seq_feat_map[fc.name]
                    if current_seq_col_idx >= X_seq_tensor.shape[2]:  # Check bounds
                        # print(f"警告: Seq tensor index out of bounds for {fc.name} (idx {current_seq_col_idx}). Skipping.")
                        continue
                except KeyError:
                    # print(f"警告: シーケンステンソル内に特徴量 {fc.name} のインデックスが見つかりません。スキップします。")
                    continue

                if isinstance(fc, SparseFeat):
                    if fc.embedding_name in self.embedding_dict:
                        embedding_layer = self.embedding_dict[fc.embedding_name]
                        try:
                            ids = X_seq_tensor[:, :, current_seq_col_idx].long()
                            emb = embedding_layer(ids)
                            seq_sparse_embedding_list.append(emb)
                        except IndexError as e:
                            print(
                                f"IndexError seq sparse forward: {fc.name}, idx:{current_seq_col_idx}, shape:{X_seq_tensor.shape}"
                            )
                            raise e
                        except Exception as e:
                            print(f"Error seq sparse forward: {fc.name}")
                            raise e
                    # else: print(f"警告: Seq Embが見つかりません: {fc.embedding_name}")
                elif isinstance(fc, DenseFeat):
                    try:
                        val = X_seq_tensor[:, :, current_seq_col_idx].unsqueeze(-1)
                        # Assume data is already cleaned in pre-processing
                        # val = torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
                        seq_dense_value_list.append(val)
                    except IndexError as e:
                        print(
                            f"IndexError seq dense forward: {fc.name}, idx:{current_seq_col_idx}, shape:{X_seq_tensor.shape}"
                        )
                        raise e
                    except Exception as e:
                        print(f"Error seq dense forward: {fc.name}")
                        raise e
        # else: print("情報: X_seq_tensor is empty or None.")

        # --- LSTM処理 ---
        if not seq_sparse_embedding_list and not seq_dense_value_list:
            # print("情報: No valid sequence features found for LSTM input.")
            lstm_output_features = torch.zeros((batch_size, self.lstm.hidden_size)).to(
                self.device
            )
        else:
            try:
                lstm_input = torch.cat(
                    seq_sparse_embedding_list + seq_dense_value_list, dim=-1
                )
                # Check if lstm_input size matches expected size
                if lstm_input.shape[-1] != self.lstm.input_size:
                    print(
                        f"致命的エラー: LSTM 入力次元不一致。期待={self.lstm.input_size}, 実際={lstm_input.shape[-1]}"
                    )
                    # Decide how to handle: raise error or return zeros?
                    raise ValueError(
                        f"LSTM input size mismatch. Expected {self.lstm.input_size}, got {lstm_input.shape[-1]}"
                    )
                    # lstm_output_features = torch.zeros((batch_size, self.lstm.hidden_size)).to(self.device)
                else:
                    lstm_out, (hn, cn) = self.lstm(lstm_input)
                    lstm_output_features = lstm_out[
                        :, -1, :
                    ]  # Use output of last time step
            except Exception as e:
                print(f"Error during LSTM processing forward: {e}")
                # Fallback to zeros
                lstm_output_features = torch.zeros(
                    (batch_size, self.lstm.hidden_size)
                ).to(self.device)

        # --- DCN入力の準備 (静的特徴量) ---
        linear_sparse_embedding_list, linear_dense_value_list = [], []
        dnn_sparse_embedding_list, dnn_dense_value_list = [], []

        if X_static_tensor is not None and X_static_tensor.nelement() > 0:
            try:
                # Use the feature_index generated by BaseModel
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
                print(f"Error in input_from_feature_columns forward: {e}")
                # Fallback lists are already empty
        # else: print("情報: X_static_tensor is empty or None.")

        # CrossNet入力
        cross_input = None
        if self.cross_input_dim > 0:
            if not linear_sparse_embedding_list and not linear_dense_value_list:
                # print("情報: No valid linear features found for CrossNet input.")
                cross_input = torch.zeros((batch_size, self.cross_input_dim)).to(
                    self.device
                )
            else:
                cross_input = combined_dnn_input(
                    linear_sparse_embedding_list, linear_dense_value_list
                )
                # Dimension check for CrossNet input
                if cross_input.shape[1] != self.cross_input_dim:
                    print(
                        f"致命的エラー: CrossNet 入力次元不一致。期待={self.cross_input_dim}, 実際={cross_input.shape[1]}"
                    )
                    raise ValueError(
                        f"CrossNet input size mismatch. Expected {self.cross_input_dim}, got {cross_input.shape[1]}"
                    )
                    # cross_input = torch.zeros((batch_size, self.cross_input_dim)).to(self.device) # Fallback

        # DNN入力
        dnn_input_combined = None
        if self.dnn_input_dim > 0:
            if not dnn_sparse_embedding_list and not dnn_dense_value_list:
                # Case where static DNN features are empty, use only LSTM output
                if (
                    lstm_output_features is not None
                    and lstm_output_features.shape[0] == batch_size
                    and lstm_output_features.shape[1] == self.lstm.hidden_size
                ):
                    dnn_input_combined = lstm_output_features
                    # Check if dnn_input_dim was ONLY lstm_hidden_size
                    if self.dnn_input_dim != self.lstm.hidden_size:
                        # This implies static features were expected but missing
                        # print(f"警告: 静的DNN特徴量が空ですが、DNN入力次元 ({self.dnn_input_dim}) はLSTM ({self.lstm.hidden_size}) より大きいです。")
                        # Pad static part? Or maybe dnn_input_dim calculation was wrong initially?
                        # Assuming the input should just be LSTM if static is missing here.
                        # If this case is valid, dnn_input_dim should equal lstm_hidden_size.
                        # For now, we proceed, but dimension check later will catch mismatch if dnn_input_dim expects static features.
                        pass
                else:  # No static DNN features AND invalid LSTM output
                    # print("警告: 有効な静的DNN特徴量もLSTM出力もありません。DNN入力はゼロになります。")
                    dnn_input_combined = torch.zeros(
                        (batch_size, self.dnn_input_dim)
                    ).to(self.device)
            else:  # Static DNN features exist
                dnn_input_static = combined_dnn_input(
                    dnn_sparse_embedding_list, dnn_dense_value_list
                )
                if (
                    lstm_output_features is None
                    or lstm_output_features.shape[0] != batch_size
                    or lstm_output_features.shape[1] != self.lstm.hidden_size
                ):
                    # print("警告: LSTM出力が無効または形状不一致。静的DNN入力のみで構成（またはゼロパディング）。")
                    # Check if dnn_input_dim expects LSTM output
                    expected_static_dim = self.dnn_input_dim - self.lstm.hidden_size
                    if dnn_input_static.shape[1] == expected_static_dim:
                        # Pad LSTM part with zeros
                        zero_lstm_padding = torch.zeros(
                            (batch_size, self.lstm.hidden_size)
                        ).to(self.device)
                        dnn_input_combined = torch.cat(
                            [dnn_input_static, zero_lstm_padding], dim=-1
                        )
                    elif dnn_input_static.shape[1] == self.dnn_input_dim:
                        # Maybe dnn_input_dim didn't include LSTM? Use static only.
                        # print("情報: DNN入力次元が静的特徴量のみと一致するため、LSTM出力を無視します。")
                        dnn_input_combined = dnn_input_static
                    else:
                        print(
                            f"致命的エラー: DNN静的入力次元 ({dnn_input_static.shape[1]}) と期待値 ({expected_static_dim} or {self.dnn_input_dim}) が不一致。"
                        )
                        raise ValueError("Static DNN input dimension mismatch.")
                        # dnn_input_combined = torch.zeros((batch_size, self.dnn_input_dim)).to(self.device) # Fallback
                else:  # Both static DNN and LSTM are valid
                    dnn_input_combined = torch.cat(
                        [dnn_input_static, lstm_output_features], dim=-1
                    )

            # Final check for DNN input dimension
            if dnn_input_combined.shape[1] != self.dnn_input_dim:
                print(
                    f"致命的エラー: 最終的なDNN入力次元 ({dnn_input_combined.shape[1]}) が期待値 ({self.dnn_input_dim}) と不一致。"
                )
                raise ValueError(
                    f"Final DNN input dimension mismatch. Expected {self.dnn_input_dim}, got {dnn_input_combined.shape[1]}"
                )
                # dnn_input_combined = torch.zeros((batch_size, self.dnn_input_dim)).to(self.device) # Fallback

        # --- DNN層 ---
        dnn_logit = None
        if self.dnn_input_dim > 0 and dnn_input_combined is not None:
            try:
                dnn_output = self.dnn(dnn_input_combined)
                dnn_logit = self.dnn_linear(dnn_output)
            except Exception as e:
                print(f"Error during DNN forward pass: {e}")
                dnn_logit = torch.zeros((batch_size, 1)).to(self.device)  # Fallback
        else:
            # print("情報: DNN input is zero or None, DNN part skipped.")
            dnn_logit = torch.zeros((batch_size, 1)).to(
                self.device
            )  # Ensure shape [B, 1]

        # --- CrossNet層 ---
        cross_out = None
        if (
            self.crossnet is not None
            and self.cross_linear is not None
            and cross_input is not None
        ):
            try:
                cross_features = self.crossnet(cross_input)
                cross_out = self.cross_linear(cross_features)
            except Exception as e:
                print(f"Error during CrossNet forward pass: {e}")
                cross_out = torch.zeros_like(dnn_logit)  # Fallback, shape [B, 1]
        else:
            # print("情報: CrossNet is disabled or input is None, CrossNet part skipped.")
            cross_out = torch.zeros_like(dnn_logit)  # Ensure shape [B, 1]

        # --- 最終出力 ---
        # Ensure both parts have the same shape [B, 1] before adding
        if dnn_logit.shape != cross_out.shape:
            print(
                f"致命的エラー: dnn_logit ({dnn_logit.shape}) と cross_out ({cross_out.shape}) の形状が不一致。"
            )
            # Attempt to recover if one is valid? Or just return zeros?
            # Let's prioritize dnn_logit if shapes mismatch, assuming cross might be zero fallback
            final_logit = (
                dnn_logit
                if dnn_logit.shape[1] == 1
                else torch.zeros((batch_size, 1)).to(self.device)
            )
        else:
            final_logit = cross_out + dnn_logit

        y_pred = self.out(
            final_logit
        )  # self.out is defined in BaseModel for regression/binary
        # Ensure output is clean
        y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

        return y_pred


# === LSTM Hybrid モデルクラス定義 (ここまで) ===


# --- 予測関数 (変更なし) ---
def predict_and_evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for seq_batch, static_batch, _ in tqdm(
            data_loader, desc="予測中"
        ):  # ラベルは予測に不要
            seq_batch = seq_batch.to(device)
            static_batch = static_batch.to(device)
            y_pred = model(seq_batch, static_batch).squeeze(-1)  # モデル入力修正
            predictions.extend(y_pred.cpu().numpy())
    return predictions


# --- 回収率計算関数 (複勝用に修正) ---
def calculate_roi(
    df_results,
    bet_type,  # <-- ★ 追加: "単勝" or "複勝"
    pred_col="prediction",
    odds_col=None,  # <-- ★ 変更: 引数で指定
    win_flag_col=None,  # <-- ★ 変更: 引数で指定
    top_n=1,
):
    """
    レースごとに予測スコア上位N頭を選択し、指定された券種の回収率と的中レース率を計算する。
    """
    if odds_col is None or win_flag_col is None:
        print(
            f"エラー({bet_type}): odds_col または win_flag_col が指定されていません。"
        )
        return {
            "investment": 0,
            "payout": 0,
            "roi": 0,
            "accuracy": 0,
            "total_races": 0,
            "correct_races": 0,
            "avg_odds": 0,
            "median_odds": 0,
        }

    investment = 0
    payout = 0
    correct_races = 0
    total_races = df_results[RACE_ID_COL].nunique()
    hit_odds = []  # 的中時のオッズを記録

    # NaN予測を除外してからランク付け
    df_results_filtered = df_results.dropna(
        subset=[pred_col, odds_col, win_flag_col]
    )  # ★ odds_col, win_flag_col もチェック
    df_results_filtered["pred_rank_in_race"] = df_results_filtered.groupby(RACE_ID_COL)[
        pred_col
    ].rank(method="first", ascending=False)
    selected_horses = df_results_filtered[
        df_results_filtered["pred_rank_in_race"] <= top_n
    ]

    # 投資額は選んだ馬の数 (1頭100円)
    investment = len(selected_horses) * 100

    # ▼▼▼【修正】的中判定と払戻計算ロジックを券種に応じて変更 ▼▼▼
    if bet_type == "単勝":
        # win_flag_col (単勝) が 1 の馬が的中
        correct_bets = selected_horses[selected_horses[win_flag_col] == 1]
        # 払戻金 = 100円 * オッズ (odds_col = 単勝オッズ)
        payout = (correct_bets[odds_col] * 100).sum()
        hit_odds.extend(correct_bets[odds_col].tolist())
        print(
            f"情報({bet_type}): 的中判定に列 '{win_flag_col} == 1'、払戻に列 '{odds_col}' を使用します。"
        )

    elif bet_type == "複勝":
        # win_flag_col (複勝圏内_実) が 0 より大きい馬 (配当金がある馬) が的中
        correct_bets = selected_horses[selected_horses[win_flag_col] > 0]
        # 払戻金 = 配当金 (odds_col = 複勝圏内_実) ※100円あたりの値
        payout = correct_bets[odds_col].sum()
        # 複勝配当金を100で割ってオッズ相当に変換して記録 (分析用)
        hit_odds.extend((correct_bets[odds_col] / 100.0).tolist())
        print(
            f"情報({bet_type}): 的中判定に列 '{win_flag_col} > 0'、払戻に列 '{odds_col}' を使用します。"
        )

    else:
        print(f"エラー({bet_type}): 未知の券種です。")
        correct_bets = pd.DataFrame()  # 空にする
    # ▲▲▲【修正ここまで】▲▲▲

    # 的中したレース数をカウント
    correct_races = correct_bets[RACE_ID_COL].nunique()

    roi = (payout / investment * 100) if investment > 0 else 0
    accuracy = (correct_races / total_races * 100) if total_races > 0 else 0
    avg_odds = np.mean(hit_odds) if hit_odds else 0
    median_odds = np.median(hit_odds) if hit_odds else 0

    return {
        "investment": investment,
        "payout": payout,
        "roi": roi,
        "accuracy": accuracy,
        "total_races": total_races,
        "correct_races": correct_races,
        "avg_odds": avg_odds,  # ★ 追加: 平均的中オッズ
        "median_odds": median_odds,  # ★ 追加: 的中オッズ中央値
    }


# --- メイン関数 ---
def main():
    print("--- LSTM Hybrid モデル 評価プログラムを開始 ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")

    # --- 前処理済みデータ (.npz) のロード ---
    print(f"--- 事前処理済みデータ ({PREPROCESSED_DATA_PATH}) をロード中 ---")
    try:
        data = np.load(PREPROCESSED_DATA_PATH, allow_pickle=True)
        X_seq = data["X_seq"]
        X_static = data["X_static"]
        y = data["y"]

        # ▼▼▼【修正】META_COLS_FOR_EVAL リストを使って一括ロード ▼▼▼
        meta_data_dict = {}
        missing_meta_cols = []
        for col in META_COLS_FOR_EVAL:
            if col in data:
                meta_data_dict[col] = data[col]
            else:
                missing_meta_cols.append(col)

        if missing_meta_cols:
            print(
                f"警告: 必要なメタデータ列が .npz に見つかりません: {missing_meta_cols}"
            )
            print(
                "  preprocess_sequences.py の META_COLS_FOR_EVAL を確認してください。"
            )
            print("  （見つからない列は無視して続行します）")
        # ▲▲▲【修正ここまで】▲▲▲

        df_meta = pd.DataFrame(meta_data_dict)
        # 特徴量名リスト
        static_feature_names_ordered = data["static_feature_names_ordered"].tolist()
        sequence_feature_names_ordered = data["sequence_feature_names_ordered"].tolist()

        print("事前処理済みデータのロード完了")
        print(f"  X_seq shape: {X_seq.shape}")
        print(f"  X_static shape: {X_static.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Metadata shape: {df_meta.shape}")
        print(f"  Loaded {len(static_feature_names_ordered)} static feature names")
        print(f"  Loaded {len(sequence_feature_names_ordered)} sequence feature names")

        # データ型をTensorに変換
        X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32)
        X_static_tensor = torch.tensor(X_static, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        del data, X_seq, X_static, y  # メモリ解放
        gc.collect()

    except Exception as e:
        print(f"エラー: 事前処理済みデータ (.npz) のロードに失敗しました: {e}")
        return

    # --- マッピング情報とモデルのロードに必要な情報をロード ---
    print(f"--- マッピング情報等を '{PREPROCESSOR_BASE_DIR}' からロード中 ---")
    try:
        with open(
            os.path.join(PREPROCESSOR_BASE_DIR, "categorical_mappings.json"),
            "r",
            encoding="utf-8",
        ) as f:
            all_mappings = json.load(f)
        # sequence_feature_names = joblib.load(os.path.join(PREPROCESSOR_BASE_DIR, "sequence_feature_names.pkl")) # npzからロード済
        # static_feature_names = joblib.load(os.path.join(PREPROCESSOR_BASE_DIR, "static_feature_names.pkl")) # npzからロード済
    except Exception as e:
        print(f"エラー: マッピング情報のロードに失敗しました: {e}")
        return

    # --- 学習済みモデルと特徴量定義のロード ---
    print(f"--- 学習済みモデルを '{MODEL_OUTPUT_DIR}' からロード中 ---")
    try:
        model_path = os.path.join(MODEL_OUTPUT_DIR, "lstm_hybrid_model.pth")
        # feature_columns.pkl は使わず、npzのリストとマッピングから再構築
        feature_columns = joblib.load(
            os.path.join(MODEL_OUTPUT_DIR, "feature_columns.pkl")
        )  # ロードして整合性チェックに使う

        # --- 特徴量リストの再構築 (モデル定義のため) ---
        # npzからロードした順序付きリストと、ロードしたマッピング情報を使う
        static_fcs = []
        sequence_fcs = []

        # static_feature_names_ordered に基づき static_fcs を作成
        for name in static_feature_names_ordered:
            if name in all_mappings:  # カテゴリカル
                vocab_size = len(all_mappings[name]) + 1  # 0は未知用
                if name == "running_style_id":  # keiba_analysis.py L1155 のロジック
                    vocab_size = 6
                static_fcs.append(
                    SparseFeat(
                        name, vocabulary_size=vocab_size, embedding_dim=EMBEDDING_DIM
                    )
                )
            else:  # 数値 (DenseFeat)
                static_fcs.append(DenseFeat(name, 1))

        # sequence_feature_names_ordered に基づき sequence_fcs を作成
        for name in sequence_feature_names_ordered:
            if name == TARGET_COL:
                continue  # ターゲット列は除外
            if name in all_mappings:  # カテゴリカル
                vocab_size = len(all_mappings[name]) + 1
                if name == "running_style_id":  # 念のため
                    vocab_size = 6
                sequence_fcs.append(
                    SparseFeat(
                        name,
                        vocabulary_size=vocab_size,
                        embedding_dim=EMBEDDING_DIM,
                        embedding_name=name,
                    )
                )  # embedding_name指定
            else:  # 数値 (DenseFeat)
                sequence_fcs.append(DenseFeat(name, 1))

        static_linear_fcs = static_fcs
        static_dnn_fcs = static_fcs

        print(f"再構築後の static_fcs の数: {len(static_fcs)}")
        print(f"再構築後の sequence_fcs の数: {len(sequence_fcs)}")

        # --- モデルのインスタンス化 ---
        model = DCN_LSTM_Hybrid(
            static_linear_feature_columns=static_linear_fcs,
            static_dnn_feature_columns=static_dnn_fcs,
            sequence_feature_columns=sequence_fcs,
            # static_feature_names=static_feature_names_ordered, # 不要
            lstm_hidden_size=LSTM_HIDDEN_SIZE,
            lstm_layers=LSTM_LAYERS,
            lstm_dropout=LSTM_DROPOUT,
            cross_num=CROSS_NUM,  # <-- ★ グローバル設定から参照するように修正 ★
            dnn_hidden_units=DNN_HIDDEN_UNITS,
            dnn_use_bn=True,  # <-- ▼▼▼【ここを False(デフォルト) から True に変更】▼▼▼
            task="regression",
            device=device,
            # l2_reg_* や init_std など、学習時と同じ値を指定した方がより正確ですが、
            # state_dict のロードだけなら必須ではありません。
        ).to(device)

        # 保存された重みをロード
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )  # weights_only=True 推奨
        print("モデルのロード完了")

    except RuntimeError as e:
        print(f"モデルのロードエラー (RuntimeError): {e}")
        print("モデル定義と保存済みモデルの構造が一致しない可能性があります。")
        print("特に LSTM 入力サイズ、DNN/CrossNet 入力サイズを確認してください。")
        # Print dimensions for debugging
        try:
            print(f"  モデルが期待する LSTM 入力サイズ: {model.lstm.input_size}")
            print(f"  モデルが期待する DNN 入力サイズ: {model.dnn_input_dim}")
            print(f"  モデルが期待する Cross 入力サイズ: {model.cross_input_dim}")
        except AttributeError:
            pass  # In case model components are None
        return
    except Exception as e:
        print(f"モデルまたは特徴量定義のロード中に予期せぬエラー: {e}")
        import traceback

        traceback.print_exc()
        return

    # --- 評価用 TensorDataset と DataLoader 作成 ---
    print("\n--- 評価用 DataLoader を作成中 ---")
    try:
        eval_tensor_dataset = TensorDataset(X_seq_tensor, X_static_tensor, y_tensor)
        eval_loader = DataLoader(
            eval_tensor_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
        print("DataLoader 作成完了")
    except Exception as e:
        print(f"エラー: DataLoader の作成中に失敗しました: {e}")
        return

    # --- 予測実行 ---
    print("\n--- 予測を実行 ---")
    predictions = predict_and_evaluate(model, eval_loader, device)

    # --- 結果の集計と保存 ---
    print("\n--- 予測結果を集計中 ---")
    if len(predictions) != len(df_meta):
        print(
            f"エラー: 予測結果数 ({len(predictions)}) とメタデータ数 ({len(df_meta)}) が一致しません。"
        )
        return

    df_meta["prediction"] = predictions

    # 結果をCSVに保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_csv_path = os.path.join(OUTPUT_DIR, "predictions_lstm.csv")
    try:
        # float を小数点以下4桁に丸めて保存
        df_meta_save = df_meta.copy()
        float_cols = df_meta_save.select_dtypes(include=["float"]).columns
        for col in float_cols:
            # prediction も float のはず
            df_meta_save[col] = df_meta_save[col].round(4)
        df_meta_save.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
        print(f"予測結果を {output_csv_path} に保存しました。")
    except Exception as e:
        print(f"CSV保存エラー: {e}")

    # --- 回収率の計算 ---
    print("\n--- 回収率を計算中 ---")
    roi_results = {}

    # JSONエンコーダーのヘルパークラス (tryブロックの外で定義)
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    try:
        # --- 単勝の計算 ---
        win_odds_col = "単勝オッズ"
        win_flag_col = "単勝"
        if win_odds_col in df_meta.columns and win_flag_col in df_meta.columns:
            print(f"\n--- [単勝] 回収率 ---")
            roi_results["win"] = {}
            for n in [1, 2, 3]:
                result_n = calculate_roi(
                    df_meta,
                    bet_type="単勝",
                    odds_col=win_odds_col,
                    win_flag_col=win_flag_col,
                    top_n=n,
                )
                roi_results["win"][f"top_{n}"] = result_n
                print(f"--- 上位 {n} 頭選択 (単勝) ---")
                print(f"  投資金額: {result_n['investment']:,.0f} 円")
                print(f"  払戻金額: {result_n['payout']:,.0f} 円")
                print(f"  回収率 (ROI): {result_n['roi']:.2f}%")
                print(
                    f"  的中レース率: {result_n['accuracy']:.2f}% ({result_n['correct_races']}/{result_n['total_races']} レース)"
                )
                print(f"  平均的中オッズ: {result_n['avg_odds']:.2f} 倍")
        else:
            print(
                f"\n警告: 単勝回収率の計算に必要な列 ({win_odds_col}, {win_flag_col}) がメタデータにないためスキップします。"
            )

        # --- 複勝の計算 ---
        place_odds_col = ODDS_COL  # グローバル設定の "複勝圏内_実"
        place_flag_col = WIN_FLAG_COL  # グローバル設定の "複勝圏内_実"
        if place_odds_col in df_meta.columns and place_flag_col in df_meta.columns:
            print(f"\n--- [複勝] 回収率 ---")
            roi_results["place"] = {}
            for n in [1, 2, 3]:
                result_n = calculate_roi(
                    df_meta,
                    bet_type="複勝",
                    odds_col=place_odds_col,
                    win_flag_col=place_flag_col,
                    top_n=n,
                )
                roi_results["place"][f"top_{n}"] = result_n
                print(f"--- 上位 {n} 頭選択 (複勝) ---")
                print(f"  投資金額: {result_n['investment']:,.0f} 円")
                print(f"  払戻金額: {result_n['payout']:,.0f} 円")
                print(f"  回収率 (ROI): {result_n['roi']:.2f}%")
                print(
                    f"  的中レース率: {result_n['accuracy']:.2f}% ({result_n['correct_races']}/{result_n['total_races']} レース)"
                )
                print(f"  平均的中オッズ (配当金/100): {result_n['avg_odds']:.2f} 倍")
        else:
            print(
                f"\n警告: 複勝回収率の計算に必要な列 ({place_odds_col}, {place_flag_col}) がメタデータにないためスキップします。"
            )

        # 回収率結果をJSONに保存
        if roi_results:  # 結果があれば保存
            output_json_path = os.path.join(OUTPUT_DIR, "roi_summary_lstm.json")
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(roi_results, f, ensure_ascii=False, indent=4, cls=NpEncoder)
            print(f"回収率サマリーを {output_json_path} に保存しました。")

    except KeyError as e:
        print(f"回収率計算エラー: 必要な列 '{e}' がメタデータに見つかりません。")
        print(f"利用可能なメタデータ列: {df_meta.columns.tolist()}")
    except Exception as e:
        print(f"回収率計算中に予期しないエラー: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 50 + "\n🎉 評価プログラムが完了しました 🎉\n" + "=" * 50)


if __name__ == "__main__":
    main()
