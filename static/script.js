// static/script.js
document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predict-form');
    const loadingSpinner = document.getElementById('loading');
    const predictButton = document.getElementById('predict-button');
    const errorMessageDiv = document.getElementById('error-message');
    const resultSection = document.getElementById('result-section');

    // レース情報
    const raceNameEl = document.getElementById('race-name');
    const raceDetailsEl = document.getElementById('race-details');

    // 予測テーブル
    const predictionTableBody = document.querySelector('#prediction-table tbody');

    // 詳細エリア
    const detailsContainer = document.getElementById('details-container');
    const detailsHorseNameEl = document.getElementById('details-horse-name');
    const factorsListEl = document.getElementById('factors-list');
    const historyTableBody = document.getElementById('history-table-body');

    // フォーム送信イベント
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // フォームデータを取得
        const formData = new FormData(form);
        const url = formData.get('netkeiba_url');
        const raceDate = formData.get('race_date');

        if (!url || !raceDate) {
            displayError('URLと開催日を入力してください。');
            return;
        }

        // ローディング開始
        showLoading(true);
        displayError(null);
        resultSection.style.display = 'none';
        detailsContainer.style.display = 'none';

        try {
            // FormData を使って POST
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok || data.error) {
                throw new Error(data.error || '予測サーバーとの通信に失敗しました。');
            }

            // 予測成功
            displayRaceInfo(data.race_info);
            displayPredictionTable(data.predictions);
            resultSection.style.display = 'block';

        } catch (error) {
            displayError(error.message);
        } finally {
            showLoading(false);
        }
    });

    // ローディング表示/非表示
    function showLoading(isLoading) {
        loadingSpinner.style.display = isLoading ? 'block' : 'none';
        predictButton.disabled = isLoading;
    }

    // エラー表示
    function displayError(message) {
        if (message) {
            errorMessageDiv.textContent = message;
            errorMessageDiv.style.display = 'block';
        } else {
            errorMessageDiv.style.display = 'none';
        }
    }

    // レース情報表示
    function displayRaceInfo(raceInfo) {
        if (!raceInfo) return;
        raceNameEl.textContent = `${raceInfo.location} ${raceInfo.number}R ${raceInfo.name}`;
        raceDetailsEl.textContent = `${raceInfo.date} | ${raceInfo.track}${raceInfo.distance}m | ${raceInfo.condition}`;
    }

    // 予測テーブル表示
    function displayPredictionTable(predictions) {
        predictionTableBody.innerHTML = ''; // クリア

        if (!predictions || predictions.length === 0) {
            predictionTableBody.innerHTML = '<tr><td colspan="6">予測データがありません。</td></tr>';
            return;
        }

        predictions.forEach(horse => {
            const tr = document.createElement('tr');

            // スコアを小数点以下4桁にフォーマット
            const score = parseFloat(horse.予測スコア).toFixed(4);
            const odds = parseFloat(horse.単勝オッズ).toFixed(1);

            tr.innerHTML = `
                <td>${horse.馬番}</td>
                <td>${horse.枠番}</td>
                <td>${horse.馬名}</td>
                <td>${horse.騎手}</td>
                <td>${odds}</td>
                <td>${score}</td>
            `;

            // クリックイベントで詳細表示
            tr.addEventListener('click', () => {
                // 他の行の選択解除
                document.querySelectorAll('#prediction-table tbody tr').forEach(row => {
                    row.classList.remove('selected-row');
                });
                // この行を選択
                tr.classList.add('selected-row');
                // 詳細表示
                displayDetails(horse);
            });

            predictionTableBody.appendChild(tr);
        });

        // 最初の行を自動的にクリック (デフォルト表示)
        if (predictionTableBody.firstChild) {
            predictionTableBody.firstChild.click();
        }
    }

    // 詳細エリア表示
    function displayDetails(horse) {
        detailsContainer.style.display = 'block';
        detailsHorseNameEl.textContent = `(${horse.馬番}) ${horse.馬名}`;

        // 1. ファクター (寄与度) 表示
        factorsListEl.innerHTML = ''; // クリア
        if (horse.factors && horse.factors.length > 0) {
            horse.factors.forEach(factor => {
                const [name, importance] = factor;
                const li = document.createElement('li');
                li.textContent = `${name}: ${parseFloat(importance).toFixed(5)}`;
                factorsListEl.appendChild(li);
            });
        } else {
            factorsListEl.innerHTML = '<li>寄与度データがありません。</li>';
        }

        // 2. 過去戦績表示
        historyTableBody.innerHTML = ''; // クリア
        if (horse.history && horse.history.length > 0) {
            horse.history.forEach(race => {
                const tr = document.createElement('tr');
                const raceDate = race.日付 ? new Date(race.日付).toLocaleDateString('ja-JP') : 'N/A';
                tr.innerHTML = `
                    <td>${raceDate}</td>
                    <td>${race.レース名 || 'N/A'}</td>
                    <td>${race.場所 || 'N/A'}</td>
                    <td>${race.着順 || 'N/A'}</td>
                    <td>${race.上り3F || 'N/A'}</td>
                `;
                historyTableBody.appendChild(tr);
            });
        } else {
            historyTableBody.innerHTML = '<tr><td colspan="5">過去戦績データがありません。</td></tr>';
        }
    }
});