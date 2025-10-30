# ファイル名: environment.py
# (辞書作成時のキー型を Python datetime.date に強制)

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete
import pandas as pd
import joblib
import polars as pl
import datetime  # ★ datetime をインポート


class KeibaBetEnv(gym.Env):
    """
    【12次元状態 / 報酬=損益(P&L) / 行動=6択 / 動的特徴量 / 11R,12R学習 版】
    (辞書作成時のキー型を Python datetime.date に強制)
    """

    metadata = {"render_modes": []}

    # ▼▼▼ 【★★★ __init__ を修正 ★★★】 ▼▼▼
    def __init__(
        self, df_features_filtered, df_full_data, scaler_path, is_training=True
    ):
        super(KeibaBetEnv, self).__init__()

        self.is_training = is_training
        self.all_test_races_done = False

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        self.action_space = Discrete(6)
        self.action_to_investment_map = {0: 0, 1: 100, 2: 200, 3: 300, 4: 400, 5: 500}

        if "単勝オッズ_実" not in df_features_filtered.columns:
            raise ValueError("'単勝オッズ_実' カラムが必要です。")
        if "Ｒ" not in df_features_filtered.columns:
            raise ValueError("'Ｒ' カラムが必要です。")

        try:
            self.scalers = joblib.load(scaler_path)
            print(f"スケーラーを {scaler_path} から読み込みました。")
            required_dynamic_features = [
                "daily_jockey_win_rate",
                "daily_track_winner_avg_pci",
                "daily_track_winner_avg_last3f",
                "daily_track_winner_avg_4c_pos_rate",
            ]
            if not all(f in self.scalers for f in required_dynamic_features):
                missing_scalers = [
                    f for f in required_dynamic_features if f not in self.scalers
                ]
                raise ValueError(
                    f"読み込んだスケーラーに動的特徴量 {missing_scalers} が不足しています。"
                )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"エラー: スケーラーファイル {scaler_path} が見つかりません。"
            )
        except Exception as e:
            raise RuntimeError(f"エラー: スケーラーの読み込みに失敗しました。 {e}")

        # --- レースデータの準備 ---
        print("環境内で日付データを整理中...")
        df_full_data = df_full_data.sort(by=["日付", "race_id", "馬番"])
        self.full_data_by_date = {}
        # ★★★ キーを Python の date オブジェクトに確実に変換 ★★★
        for date_pl in df_full_data["日付"].unique(maintain_order=True):
            # Polars Date -> Python datetime.date
            try:
                # None の場合はスキップ
                if date_pl is None:
                    continue
                # to_pydate() があれば呼び出す
                if hasattr(date_pl, "to_pydate"):
                    date_py = date_pl.to_pydate()
                # 既に datetime.date ならそのまま使う (念のため)
                elif isinstance(date_pl, datetime.date):
                    date_py = date_pl
                else:
                    # それ以外の型はエラーとして扱うか、変換を試みる
                    # ここでは単純に文字列に変換してからパースしてみる (Polars <-> Python 互換性問題対策)
                    date_str = str(date_pl)  # 例: "2024-01-07"
                    date_py = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                    # print(f"  Info: Converted date key {date_str} to {date_py}") # デバッグ用

            except Exception as e_conv:
                print(
                    f"警告: 日付キーの変換中にエラー ({type(date_pl)}: {date_pl}): {e_conv}。スキップします。"
                )
                continue  # この日付はスキップ

            # グループ化も pydate でフィルターして行う
            # Polars の filter は Python の date オブジェクトを直接比較できるはず
            date_group = df_full_data.filter(pl.col("日付") == date_py)
            if not date_group.is_empty():
                self.full_data_by_date[date_py] = date_group.clone()

        print(f"  -> 全データ辞書のキー数: {len(self.full_data_by_date)}")
        if len(self.full_data_by_date) == 0:
            raise ValueError(
                "全データ辞書の作成に失敗しました。日付カラムを確認してください。"
            )

        # ★★★ self.dates も Python の date オブジェクトで保持 ★★★
        df_features = df_features_filtered.sort(by=["日付", "race_id", "馬番"])
        unique_dates_pl = df_features["日付"].unique(maintain_order=True)
        self.dates = []
        for d in unique_dates_pl:
            if d is None:
                continue
            try:
                if hasattr(d, "to_pydate"):
                    date_pyd = d.to_pydate()
                elif isinstance(d, datetime.date):
                    date_pyd = d
                else:
                    date_pyd = datetime.datetime.strptime(str(d), "%Y-%m-%d").date()
                self.dates.append(date_pyd)
            except Exception as e_conv_list:
                print(
                    f"警告: 日付リストの変換中にエラー ({type(d)}: {d}): {e_conv_list}。スキップします。"
                )
        self.dates = sorted(self.dates)
        print(f"  -> 対象日付リストの要素数: {len(self.dates)}")

        if not self.dates:
            raise ValueError("渡された df_features に有効な日付が含まれていません。")

        self.date_iterator = None
        self.reset_date_iterator()
        self.current_date = None
        self.current_date_df_full = None
        self.current_date_race_results = []
        self.race_iterator_for_date = None
        self.current_race_df = None
        self.current_horse_index = 0
        self.current_race_bets = []

    # ▲▲▲ 【★★★ __init__ を修正 ★★★】 ▲▲▲

    # (以降のメソッド reset_date_iterator, _load_next_date などは変更なし)
    # ... (前回のコードと同じ) ...
    def reset_date_iterator(self):
        if self.is_training:
            np.random.shuffle(self.dates)
        self.date_iterator = iter(self.dates)

    def _load_next_date(self):
        try:
            target_date = next(self.date_iterator)
        except StopIteration:
            if self.is_training:
                self.reset_date_iterator()
                target_date = next(self.date_iterator)
            else:
                self.all_test_races_done = True
                return False
        date_df_full = self.full_data_by_date.get(target_date)
        if date_df_full is None:
            print(
                f"警告: 全データ辞書に日付 {target_date} が見つかりません。スキップします。"
            )
            return False
        self.current_date = target_date
        self.current_date_df_full = date_df_full
        self.current_date_race_results = []
        if self.is_training:
            race_ids_today = sorted(
                self.current_date_df_full.filter(pl.col("Ｒ").is_in([11, 12]))[
                    "race_id"
                ]
                .unique()
                .to_list()
            )
        else:
            race_ids_today = sorted(
                self.current_date_df_full["race_id"].unique().to_list()
            )
        if not race_ids_today:
            print(f"警告: 日付 {self.current_date} に対象レースが見つかりません。")
            return False
        self.race_iterator_for_date = iter(race_ids_today)
        return True

    def _load_next_race(self):
        if self.race_iterator_for_date is None:
            return False
        try:
            race_id = next(self.race_iterator_for_date)
        except StopIteration:
            return False
        self.current_race_df = self.current_date_df_full.filter(
            pl.col("race_id") == race_id
        )
        if self.current_race_df.is_empty():
            print(f"警告: race_id {race_id} のレースデータが空です。")
            return False
        self.current_horse_index = 0
        self.current_race_bets = []
        return True

    def _calculate_dynamic_features(self, current_row_dict):
        dynamic_features = {}
        jockey = current_row_dict.get("騎手")
        rides, wins = 0, 0
        if jockey:
            for rn, res_df in self.current_date_race_results:
                j_rides = res_df.filter(pl.col("騎手") == jockey)
                if not j_rides.is_empty():
                    rides += 1
                    wins += j_rides["着順_num"][0] == 1
        dynamic_features["daily_jockey_win_rate"] = (wins / rides) if rides > 0 else 0.0
        turf_winners_list = [
            res.filter((pl.col("芝・ダ") == "芝") & (pl.col("着順_num") == 1))
            for rn, res in self.current_date_race_results
        ]
        turf_winners = (
            pl.concat(turf_winners_list) if turf_winners_list else pl.DataFrame()
        )
        dynamic_features["daily_track_winner_avg_pci"] = (
            turf_winners["PCI_num"].mean() if not turf_winners.is_empty() else 0.0
        )
        dynamic_features["daily_track_winner_avg_last3f"] = (
            turf_winners["上り3F_num"].mean() if not turf_winners.is_empty() else 0.0
        )
        dirt_winners_list = [
            res.filter((pl.col("芝・ダ") == "ダ") & (pl.col("着順_num") == 1))
            for rn, res in self.current_date_race_results
        ]
        dirt_winners = (
            pl.concat(dirt_winners_list) if dirt_winners_list else pl.DataFrame()
        )
        dynamic_features["daily_track_winner_avg_4c_pos_rate"] = (
            dirt_winners["4角順位率_num"].mean() if not dirt_winners.is_empty() else 0.0
        )
        scaled_dynamic = {}
        dynamic_feature_names_to_scale = [
            "daily_jockey_win_rate",
            "daily_track_winner_avg_pci",
            "daily_track_winner_avg_last3f",
            "daily_track_winner_avg_4c_pos_rate",
        ]
        for key in dynamic_feature_names_to_scale:
            value = dynamic_features.get(key, 0.0)
            scaled_dynamic[f"{key}_scaled"] = (
                self.scalers[key].transform(np.array(value).reshape(-1, 1))[0, 0]
                if key in self.scalers
                else 0.0
            )
        return scaled_dynamic

    def _get_obs(self):
        static_features_scaled = {}
        static_feature_names = [
            "score_scaled",
            "odds_scaled",
            "rank_scaled",
            "diff_scaled",
            "headcount_scaled",
            "mean_scaled",
            "std_scaled",
            "gap_scaled",
        ]
        row_dict = None
        try:
            row_dict = self.current_race_df.row(self.current_horse_index, named=True)
            static_features_scaled = {
                name: row_dict.get(name, 0.0) for name in static_feature_names
            }
        except Exception as e:
            print(f"エラー: _get_obs で静的特徴量の取得に失敗 ({e})")
            static_features_scaled = {name: 0.0 for name in static_feature_names}
        if row_dict is not None:
            dynamic_features_scaled = self._calculate_dynamic_features(row_dict)
        else:
            dynamic_feature_names_scaled = [
                "daily_jockey_win_rate_scaled",
                "daily_track_winner_avg_pci_scaled",
                "daily_track_winner_avg_last3f_scaled",
                "daily_track_winner_avg_4c_pos_rate_scaled",
            ]
            dynamic_features_scaled = {
                name: 0.0 for name in dynamic_feature_names_scaled
            }
        obs_list = [
            static_features_scaled.get(k, 0.0) for k in static_feature_names
        ] + [
            dynamic_features_scaled.get("daily_jockey_win_rate_scaled", 0.0),
            dynamic_features_scaled.get("daily_track_winner_avg_pci_scaled", 0.0),
            dynamic_features_scaled.get("daily_track_winner_avg_last3f_scaled", 0.0),
            dynamic_features_scaled.get(
                "daily_track_winner_avg_4c_pos_rate_scaled", 0.0
            ),
        ]
        target_dim = self.observation_space.shape[0]
        if len(obs_list) != target_dim:
            print(f"警告: obs次元数({len(obs_list)})!=期待値({target_dim})")
            (
                obs_list.extend([0.0] * (target_dim - len(obs_list)))
                if len(obs_list) < target_dim
                else obs_list[:target_dim]
            )
        return np.array(obs_list, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        while True:
            date_loaded = self._load_next_date()
            if not date_loaded:
                if self.all_test_races_done:
                    obs = np.zeros(self.observation_space.shape, dtype=np.float32)
                    info = {}
                    return obs, info
                else:
                    continue
            race_loaded = self._load_next_race()
            if not race_loaded:
                continue
            obs = self._get_obs()
            info = {}
            return obs, info

    def step(self, action):
        if self.all_test_races_done:
            terminated = True
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            reward = 0
            info = {}
            truncated = False
            return obs, reward, terminated, truncated, info
        current_row_dict = self.current_race_df.row(
            self.current_horse_index, named=True
        )
        final_action = action
        investment = self.action_to_investment_map.get(final_action, 0)
        payout = 0
        is_win = False
        if investment > 0:
            payout_per_100yen = current_row_dict.get("単勝オッズ_実", 0.0)
            payout = payout_per_100yen * (investment / 100.0)
            self.current_race_bets.append((investment, payout))
            is_win = payout > 0
        self.current_horse_index += 1
        terminated = self.current_horse_index >= self.current_race_df.height
        truncated = False
        reward = 0
        info = {}
        if terminated:
            total_investment = sum(bet[0] for bet in self.current_race_bets)
            total_payout = sum(bet[1] for bet in self.current_race_bets)
            reward = total_payout - total_investment
            info["episode"] = {"r": reward, "l": self.current_race_df.height}
            total_bets_race = len(self.current_race_bets)
            total_wins_race = sum(1 for inv, pay in self.current_race_bets if pay > 0)
            info["total_investment_race"] = total_investment
            info["total_payout_race"] = total_payout
            info["total_bets_race"] = total_bets_race
            info["total_wins_race"] = total_wins_race
            try:
                rn = current_row_dict.get("Ｒ", 0)
                result_cols = [
                    "騎手",
                    "芝・ダ",
                    "着順_num",
                    "PCI_num",
                    "4角順位率_num",
                    "上り3F_num",
                ]
                self.current_date_race_results.append(
                    (rn, self.current_race_df.select(result_cols))
                )
            except Exception as e:
                print(f"警告: レース結果の記録に失敗: {e}")
            while True:  # 次の有効なレース/日付を探すループ
                race_loaded = self._load_next_race()
                if race_loaded:
                    obs = self._get_obs()
                    break
                else:
                    date_loaded = self._load_next_date()
                    if not date_loaded:
                        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
                        break
                    else:
                        continue
        else:
            obs = self._get_obs()
        return obs, reward, terminated, truncated, info
