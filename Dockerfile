# Dockerfile

# ベースイメージの指定 (推奨)
FROM python:3.11.9-slim-bookworm

# 開発環境の作業ディレクトリを設定
WORKDIR /app

# 依存関係ファイル（requirements.txt）をコピー
COPY requirements.txt /app/

# 依存関係をインストール
# --no-cache-dir をつけることでイメージサイズを削減
RUN pip install --no-cache-dir -r requirements.txt

# プロジェクトのコード全体をコンテナにコピー
COPY . /app/

# アプリケーションを起動するコマンド (例: Pythonスクリプトの実行)
CMD ["python", "app.py"]