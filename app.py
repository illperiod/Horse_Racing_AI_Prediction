# app.py
from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
from predict_logic import Predictor
import logging
import traceback
import os
import json
import subprocess
import sys

# ★★★ ロギング設定を一箇所に統一 ★★★
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # 標準出力に確実に出力
    ],
    force=True,  # 既存の設定を上書き
)

# predict_logic モジュールのロガーを明示的に有効化
predict_logger = logging.getLogger("predict_logic")
predict_logger.setLevel(logging.INFO)
predict_logger.propagate = True  # 親ロガーに伝播

# --- Flaskアプリケーションの初期化 ---
app = Flask(__name__, template_folder="templates")

# Flaskとwerkzeugのログレベルを調整
app.logger.setLevel(logging.INFO)
werkzeug_logger = logging.getLogger("werkzeug")
werkzeug_logger.setLevel(logging.INFO)

# --- グローバル設定 ---
HISTORICAL_DATA_PATH = "2010_2025_data.csv"
PEDIGREE_DATA_PATH = "2005_2025_Pedigree.csv"
PREPROCESSOR_BASE_DIR = "lstm_preprocessor_score"
MODEL_BASE_DIR = "lstm_models_score"
SHUTUBA_CSV_PATH = "shutuba_temp.csv"

# --- 予測器のインスタンスを生成 ---
try:
    logging.info("予測器の初期化を開始します...")
    predictor = Predictor(
        historical_data_path=HISTORICAL_DATA_PATH,
        pedigree_data_path=PEDIGREE_DATA_PATH,
        preprocessor_base_dir=PREPROCESSOR_BASE_DIR,
        model_base_dir=MODEL_BASE_DIR,
    )
    logging.info("予測器の初期化が完了しました。")
except Exception as e:
    logging.error(f"予測器の初期化中にエラーが発生しました: {e}")
    logging.error(traceback.format_exc())
    predictor = None


# --- ルーティング ---


@app.route("/", methods=["GET"])
def index():
    """
    静的な index.html を配信するルート。
    """
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    予測を実行し、結果をJSONで返すAPIエンドポイント。
    """
    print("\n" + "=" * 80)
    print("【app.py】/predict エンドポイントが呼ばれました")
    print("=" * 80)

    if not predictor:
        logging.error("予測器が初期化されていません")
        return (
            jsonify(
                {
                    "error": "予測器が初期化されていません。サーバーログを確認してください。"
                }
            ),
            500,
        )

    try:
        url = request.form.get("netkeiba_url")
        form_race_date = request.form.get("race_date")

        print(f"【app.py】受信データ:")
        print(f"  - URL: {url}")
        print(f"  - 日付: {form_race_date}")

        if not url:
            return jsonify({"error": "netkeibaのURLを入力してください。"}), 400

        logging.info(f"単一レース予測リクエストを受信 (API): {url}")

        # 1. スクレイパーを実行して shutuba_temp.csv を作成
        print(f"【app.py】スクレイパーを実行中... URL: {url}")
        logging.info(f"スクレイパーを実行中... URL: {url}")
        race_info = {}
        try:
            process = subprocess.run(
                ["python", "scraper.py", url, SHUTUBA_CSV_PATH],
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
                timeout=120,
            )

            stdout_lines = process.stdout.strip().split("\n")
            race_info = json.loads(stdout_lines[-1])
            print(f"【app.py】スクレイピング成功。レース情報: {race_info}")
            logging.info(f"スクレイピング成功。レース情報: {race_info}")

        except subprocess.CalledProcessError as e:
            logging.error(f"スクレイピング失敗: {e.stderr}")
            return jsonify({"error": f"スクレイピングに失敗しました: {e.stderr}"}), 500
        except Exception as e:
            logging.error(f"スクレイピング処理エラー: {e}")
            return jsonify({"error": f"スクレイピング処理中にエラーが発生: {e}"}), 500

        # 2. スクレイピング成功後、予測ロジック実行
        print(f"【app.py】予測ロジックを実行します (CSV: {SHUTUBA_CSV_PATH})...")
        logging.info(f"予測ロジックを実行中 (CSV: {SHUTUBA_CSV_PATH})...")

        # ★★★ ここで run_prediction が呼ばれる ★★★
        print(f"【app.py】predictor.run_prediction を呼び出します")
        sys.stdout.flush()  # バッファをフラッシュ

        prediction_context = predictor.run_prediction(
            url, SHUTUBA_CSV_PATH, form_race_date, race_info
        )

        print(
            f"【app.py】予測完了。結果: success={prediction_context.get('success', False)}"
        )
        sys.stdout.flush()

        # 一時ファイルの削除
        try:
            if os.path.exists(SHUTUBA_CSV_PATH):
                os.remove(SHUTUBA_CSV_PATH)
            for temp_file in ["temp_combined_race.csv", "temp_combined_ped.csv"]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        except Exception as e:
            logging.warning(f"一時ファイルの削除に失敗: {e}")

        # エラーチェック
        if prediction_context.get("error"):
            print(f"【app.py】エラーが返されました: {prediction_context['error']}")
            return jsonify({"error": prediction_context["error"]}), 500

        # 成功時は JSON を返す
        print("【app.py】JSONレスポンスを返します")
        return jsonify(prediction_context)

    except FileNotFoundError as e:
        logging.error(f"ファイルが見つかりません: {e}")
        return jsonify({"error": f"必要なファイルが見つかりません: {e}"}), 500
    except Exception as e:
        logging.error(f"予測処理中にエラーが発生: {e}")
        logging.error(traceback.format_exc())
        print(f"【app.py ERROR】予測処理中にエラー: {e}")
        traceback.print_exc()
        return jsonify({"error": f"予測処理中にエラーが発生しました: {e}"}), 500


@app.route("/fetch-race-info", methods=["POST"])
def fetch_race_info():
    """レース情報を取得するAPI"""
    try:
        url = request.json.get("url")
        if not url:
            return jsonify({"error": "URLが指定されていません"}), 400

        # スクレイピング実行
        temp_output = "temp_fetch.csv"
        process = subprocess.run(
            ["python", "scraper.py", url, temp_output],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )

        stdout_lines = process.stdout.strip().split("\n")
        race_info = json.loads(stdout_lines[-1])

        # 一時ファイル削除
        if os.path.exists(temp_output):
            os.remove(temp_output)

        return jsonify(race_info)

    except Exception as e:
        logging.error(f"レース情報取得エラー: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 80)
    print("Flask アプリケーションを起動します")
    print("=" * 80)
    app.run(debug=True, host="0.0.0.0", port=5000)
