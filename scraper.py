import sys
import pandas as pd
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlparse, parse_qs
import io

# 標準出力・標準エラー出力のエンコーディングをUTF-8に強制
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


def extract_race_info_from_url(url):
    """
    URLからrace_idを抽出し、レース情報を取得する

    例: https://race.netkeiba.com/race/shutuba.html?race_id=202509040912&rf=race_submenu
    -> race_id: 202509040912
    -> 年: 2025, 場所: 阪神, R: 12
    """
    try:
        # URLパラメータからrace_idを抽出
        parsed_url = urlparse(url)
        params = parse_qs(parsed_url.query)

        race_id_str = params.get("race_id", [None])[0]

        if not race_id_str or len(race_id_str) != 12:
            print(f"[URL Parse Warning] 不正なrace_id: {race_id_str}", file=sys.stderr)
            return {}

        # race_idをパース
        # フォーマット: YYYYPPKKDDRR
        # YYYY: 年 (4桁)
        # PP: 場所コード (2桁)
        # KK: 開催回 (2桁)
        # DD: 日目 (2桁)
        # RR: レース番号 (2桁)

        year = int(race_id_str[0:4])
        place_code = race_id_str[4:6]
        race_num = int(race_id_str[10:12])  # ← R数

        # 場所コードから場所名を取得
        place_map = {
            "01": "札幌",
            "02": "函館",
            "03": "福島",
            "04": "新潟",
            "05": "東京",
            "06": "中山",
            "07": "中京",
            "08": "京都",
            "09": "阪神",
            "10": "小倉",
        }

        place_name = place_map.get(place_code, "不明")

        print(f"[URL Parse] 場所: {place_name}, R: {race_num}", file=sys.stderr)

        return {
            "race_location": place_name,
            "race_number": race_num,
        }
    except Exception as e:
        print(f"[URL Parse Error] {e}", file=sys.stderr)
        return {}


def parse_race_info(soup):
    """
    BeautifulSoupオブジェクトからレースの詳細情報を抽出する。
    ★★★ <title> タグからグレード情報を取得するように修正 ★★★
    """
    info = {}
    try:
        # --- <title> タグから情報を抽出 (最優先) ---
        title_tag = soup.find("title")
        race_class_from_title = None
        if title_tag:
            title_text = title_tag.get_text(strip=True)
            # 例: "シリウスＳ(G3) 出馬表 | 2025年..." -> G3 を抽出
            grade_match_title = re.search(r"\((G[123I]+)\)", title_text)
            if grade_match_title:
                # 'GIII' を 'G3' に正規化
                race_class_from_title = grade_match_title.group(1).replace("I", "")
                info["race_class_from_title"] = (
                    race_class_from_title  # デバッグ用に保存
                )
                print(
                    f"[Scraper Debug] グレード(<title>): {race_class_from_title}",
                    file=sys.stderr,
                )

        # --- 基本情報 (レース名と隣接グレード) ---
        race_title_area = soup.select_one(
            ".RaceList_Item02"
        )  # レース名とグレードを含む要素
        race_class_from_span = None
        if race_title_area:
            race_name_tag = race_title_area.select_one(".RaceName")
            race_grade_tag = race_title_area.select_one(
                ".RaceGrade"
            )  # グレード用のspanタグ

            if race_name_tag:
                info["race_name"] = race_name_tag.get_text(strip=True)
                print(f"[Scraper Debug] レース名: {info['race_name']}", file=sys.stderr)

            if race_grade_tag:
                grade_text = race_grade_tag.get_text(strip=True)
                grade_match_span = re.search(r"G[123I]+", grade_text)
                if grade_match_span:
                    race_class_from_span = grade_match_span.group(0).replace("I", "")
                    info["race_class_from_span"] = (
                        race_class_from_span  # デバッグ用に保存
                    )
                    print(
                        f"[Scraper Debug] グレード(隣接Span): {race_class_from_span}",
                        file=sys.stderr,
                    )

        # --- レース詳細情報 ---
        race_data01_text = (
            soup.select_one(".RaceData01").get_text(strip=True)
            if soup.select_one(".RaceData01")
            else ""
        )
        race_data02_text = (
            soup.select_one(".RaceData02").get_text(strip=True)
            if soup.select_one(".RaceData02")
            else ""
        )
        print(f"[Scraper Debug] RaceData01: {race_data01_text}", file=sys.stderr)
        print(f"[Scraper Debug] RaceData02: {race_data02_text}", file=sys.stderr)

        # --- 日付と年 ---
        date_match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", race_data02_text)
        if date_match:
            dt = datetime.strptime(date_match.group(0), "%Y年%m月%d日")
            info["race_date_str"] = date_match.group(0)  # ← race_date_str として保存
        else:
            date_match_no_year = re.search(r"(\d{1,2})月(\d{1,2})日", race_data02_text)
            if date_match_no_year:
                current_year = datetime.now().year
                dt_str = f"{current_year}年{date_match_no_year.group(0)}"
                info["race_date_str"] = dt_str  # ← race_date_str として保存
        if "race_date_str" in info:
            print(f"[Scraper Debug] 日付: {info['race_date_str']}", file=sys.stderr)

        # --- 場所 (スクレイピングからも取得するが、URLからの情報を優先) ---
        venue_match = re.search(r"\d+回\s*(\S+?)\s*\d+日目", race_data02_text)
        if venue_match:
            info["race_venue_scraped"] = venue_match.group(1)
            print(
                f"[Scraper Debug] 場所(スクレイピング): {info['race_venue_scraped']}",
                file=sys.stderr,
            )

        # --- コース種別, 距離, 回り ---
        course_match = re.search(
            r"([芝ダ障])(\d+)m\s*?\(?(内|外|右|左)\)?", race_data01_text
        )
        if course_match:
            info["race_track_type"] = course_match.group(1)
            info["race_distance"] = int(course_match.group(2))
            info["mawari"] = course_match.group(3)
            print(
                f"[Scraper Debug] コース: {info['race_track_type']}{info['race_distance']}m ({info['mawari']})",
                file=sys.stderr,
            )
        else:
            course_match_simple = re.search(r"([芝ダ障])(\d+)m", race_data01_text)
            if course_match_simple:
                info["race_track_type"] = course_match_simple.group(1)
                info["race_distance"] = int(course_match_simple.group(2))
                print(
                    f"[Scraper Debug] コース(シンプル): {info['race_track_type']}{info['race_distance']}m",
                    file=sys.stderr,
                )

        # --- 天候と馬場状態 ---
        weather_match = re.search(r"天候:(\S+)", race_data01_text)
        if weather_match:
            info["race_weather"] = weather_match.group(1)
            print(f"[Scraper Debug] 天候: {info['race_weather']}", file=sys.stderr)

        baba_match = re.search(r"馬場:(\S+)", race_data01_text)
        if baba_match:
            info["race_track_condition"] = baba_match.group(1)
            print(
                f"[Scraper Debug] 馬場: {info['race_track_condition']}", file=sys.stderr
            )

        # --- クラス名 ---
        class_text_source = race_data02_text + (info.get("race_name", ""))

        # ▼▼▼【ここから修正】▼▼▼
        # 1. <title> タグからの情報を最優先
        if race_class_from_title:
            info["race_class"] = race_class_from_title
            print(
                f"[Scraper Debug] クラス(最終): {info['race_class']} (<title>情報を優先)",
                file=sys.stderr,
            )
        # 2. 次に隣接Spanからの情報
        elif race_class_from_span:
            info["race_class"] = race_class_from_span
            print(
                f"[Scraper Debug] クラス(最終): {info['race_class']} (隣接Span情報を優先)",
                file=sys.stderr,
            )
        else:
            # 3. それでもなければ、従来通り RaceData02 やレース名から推定
            class_match = re.search(
                r"(\d勝クラス|未勝利|新馬|オープン|OP|G1|G2|G3|L|GI|GII|GIII)",
                class_text_source,
            )
            if class_match:
                info["race_class"] = class_match.group(1).replace("I", "")
                print(
                    f"[Scraper Debug] クラス(最終): {info['race_class']} (従来の方法で推定)",
                    file=sys.stderr,
                )
            else:
                info["race_class"] = "オープン"  # デフォルト
                print(
                    f"[Scraper Debug] クラス(最終): {info['race_class']} (デフォルト)",
                    file=sys.stderr,
                )
        # ▲▲▲【ここまで修正】▲▲▲

    except Exception as e:
        print(f"レース情報の解析中にエラー: {e}", file=sys.stderr)

    return info


def extract_weight_from_text(text):
    """
    馬体重のテキストから数値を抽出

    Args:
        text: 馬体重の文字列 (例: "480(+2)", "456", "----")

    Returns:
        tuple: (馬体重, 増減) または (None, None)
    """
    if not text or pd.isna(text) or text == "----":
        return None, None

    text = str(text).strip()

    # パターン1: "480(+2)" のような形式
    match = re.match(r"(\d+)\(([+-]?\d+)\)", text)
    if match:
        weight = int(match.group(1))
        change = int(match.group(2))
        return weight, change

    # パターン2: "480" のような数値のみ
    match = re.match(r"(\d+)", text)
    if match:
        weight = int(match.group(1))
        return weight, 0

    return None, None


def parse_shutuba_table(soup):
    """
    pandas.read_htmlを使い、出馬表をDataFrameとして堅牢に解析する。
    馬体重を正確に抽出し、新馬の場合は適切に処理する。
    """
    try:
        table_tag = soup.find("table", class_="Shutuba_Table")
        if not table_tag:
            print("出馬表テーブルが見つかりませんでした。", file=sys.stderr)
            return None

        from io import StringIO

        df = pd.read_html(StringIO(str(table_tag)))[0]
        df.columns = [col[1] for col in df.columns]

        # --- 列名を正規化 ---
        new_columns = []
        for col in df.columns:
            col_str = str(col).strip()
            if "オッズ" in col_str:
                col_str = "オッズ"
            elif "馬体重" in col_str:
                col_str = "馬体重(増減)"
            new_columns.append(col_str)
        df.columns = new_columns

        print(f"[Scraper Debug] 取得した列名: {df.columns.tolist()}", file=sys.stderr)

        # --- 馬体重を処理 ---
        if "馬体重(増減)" in df.columns:
            print("[Scraper Debug] 馬体重データを処理中...", file=sys.stderr)

            # 馬体重と増減を分離
            weights = []
            changes = []

            for idx, row in df.iterrows():
                weight_text = row["馬体重(増減)"]
                weight, change = extract_weight_from_text(weight_text)

                if weight is None:
                    print(
                        f"[Scraper Debug] 行{idx}: 馬体重データなし ('{weight_text}')",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"[Scraper Debug] 行{idx}: 馬体重={weight}kg, 増減={change}kg",
                        file=sys.stderr,
                    )

                weights.append(weight)
                changes.append(change)

            # 馬体重列を上書き
            df["馬体重"] = weights
            df["馬体重増減"] = changes

            # 元の列を削除
            df = df.drop(columns=["馬体重(増減)"])

            # 馬体重がNoneの場合の処理
            none_count = df["馬体重"].isna().sum()
            if none_count > 0:
                print(
                    f"[Scraper Debug] 馬体重データなし: {none_count}頭。predict_logic側で補完するためNaNのままにします。",
                    file=sys.stderr,
                )
        else:
            print("[Scraper Debug] 馬体重(増減)列が見つかりません", file=sys.stderr)
            df["馬体重"] = None
            df["馬体重増減"] = None

        # 不要な列を削除
        df = df.drop(["印", "調教", "厩舎コメント", "備考"], axis=1, errors="ignore")
        df = df[df["馬 番"].notna()]

        print(f"[Scraper Debug] 最終的な列名: {df.columns.tolist()}", file=sys.stderr)
        print(
            f"[Scraper Debug] 馬体重サンプル:\n{df[['馬 番', '馬体重', '馬体重増減']].head()}",
            file=sys.stderr,
        )

        return df

    except Exception as e:
        print(f"出馬表の解析中にエラー: {e}", file=sys.stderr)
        import traceback

        print(traceback.format_exc(), file=sys.stderr)
        return None


def main():
    if len(sys.argv) < 3:
        print(
            "使い方: python scraper.py <netkeiba_url> <shutuba_output_path>",
            file=sys.stderr,
        )
        sys.exit(1)

    url = sys.argv[1]
    output_filename = sys.argv[2]

    # ★★★ URLから場所とR数を取得 ★★★
    url_info = extract_race_info_from_url(url)

    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(
        service=ChromeService(ChromeDriverManager().install()), options=options
    )
    wait = WebDriverWait(driver, 30)

    try:
        print(f"アクセス中: {url}", file=sys.stderr)
        driver.get(url)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".ShutubaTable")))
        soup = BeautifulSoup(driver.page_source, "lxml")

        # 1. レース情報を抽出
        race_info = parse_race_info(soup)

        # ★★★ URLからの情報をマージ（URLの情報を優先） ★★★
        race_info.update(url_info)

        # 最終的なrace_info辞書を整形
        final_race_info = {
            "race_name": race_info.get("race_name", "不明"),
            "race_location": race_info.get("race_location", "不明"),
            "race_number": race_info.get("race_number", 0),
            "race_distance": race_info.get("race_distance", 0),
            "race_track_type": race_info.get("race_track_type", "不明"),
            "race_track_condition": race_info.get("race_track_condition", "不明"),
            "race_weather": race_info.get("race_weather", "不明"),
            "race_class": race_info.get("race_class", "オープン"),
            "race_date_str": race_info.get("race_date_str", ""),
        }

        print(f"[Final Race Info] {final_race_info}", file=sys.stderr)
        print(json.dumps(final_race_info, ensure_ascii=False))

        # 2. 出馬表を取得してCSVに保存
        df_shutuba = parse_shutuba_table(soup)
        if df_shutuba is not None:
            df_shutuba.to_csv(output_filename, index=False, encoding="utf-8-sig")
            print(f"出馬表を '{output_filename}' に保存しました。", file=sys.stderr)
        else:
            print("出馬表データの取得に失敗しました。", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"スクレイピング中に致命的なエラーが発生: {e}", file=sys.stderr)
        import traceback

        print(f"詳細: {traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
