from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
import subprocess, sys, os, json
from pathlib import Path
from datetime import datetime

app = Flask(__name__)
CORS(app)

PYTHON_EXE = sys.executable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT = os.path.join(BASE_DIR, "main.py")
DETECTIONS_JSONL = Path(BASE_DIR) / "detections.jsonl"

@app.route('/')
def dashboard():
    # detections.jsonlを読んで表示
    records = []
    if DETECTIONS_JSONL.exists():
        with open(DETECTIONS_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except:
                        continue
    
    rows = ""
    # 最新50件を表示
    for r in reversed(records[-50:]):
        frame_id = r.get('frame_id', '?')
        person_ids = r.get('person_ids', [])
        count = len(person_ids) if isinstance(person_ids, list) else '?'
        rows += f"<tr><td>Frame {frame_id}</td><td>{count} 人</td></tr>"
    
    # 最新検出数を取得（テーブルには載せているが、別途ページ上部に表示するため）
    latest_count = None
    if records:
      last = records[-1]
      person_ids = last.get('person_ids', [])
      latest_count = len(person_ids) if isinstance(person_ids, list) else None
    
    # 合計の一意な人数を集計（全フレーム通して）
    total_unique_people = set()
    for r in records:
      person_ids = r.get('person_ids', [])
      if isinstance(person_ids, list):
        total_unique_people.update(person_ids)
    total_count = len(total_unique_people)

    # 最新画像のファイルを探す
    latest_image = None
    try:
      import glob
      img_files = glob.glob(os.path.join(BASE_DIR, 'out', '*_detid.jpg'))
      if img_files:
        img_files.sort(key=lambda x: os.path.getmtime(x))
        latest_image = os.path.relpath(img_files[-1], BASE_DIR).replace('\\','/')
    except Exception:
      latest_image = None
    return render_template_string("""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>研究室 人数検出</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <h1>🚁 研究室 人数検出ダッシュボード</h1>
  
  <div class="stats-container">
    <div class="stat-card total">
      <div class="stat-label">合計検出人数（全フレーム）</div>
      <div class="stat-value">{{ total_count }}</div>
    </div>
    <div class="stat-card latest">
      <div class="stat-label">最新フレーム検出人数</div>
      <div class="stat-value">{{ latest_count if latest_count is not none else '0' }}</div>
    </div>
  </div>
  
  {% if latest_image %}
  <div class="image-container">
    <strong>最新検出画像:</strong>
    <img src="/{{ latest_image }}" />
  </div>
  {% endif %}
  
  <button id="btn" onclick="startMission()">ドローン飛行開始</button>
  <div id="status">待機中</div>

  <h2>最新の検出履歴</h2>
  <table>
    <thead>
        <tr><th>フレーム</th><th>検出人数</th></tr>
    </thead>
    <tbody>
        """ + (rows if rows else "<tr><td colspan='2'>データがありません</td></tr>") + """
    </tbody>
  </table>

  <script>
    async function startMission() {
      const btn = document.getElementById('btn');
      const status = document.getElementById('status');
      
      if(!confirm('ドローンの周囲の安全を確認しましたか？ミッションを開始します。')) return;

      btn.disabled = true;
      status.textContent = 'ミッション実行中... (WiFi切替・飛行・解析を含め数分かかります。ブラウザを閉じずに待ってください)';
      status.style.background = '#fff3cd';

      try {
        const res = await fetch('/start-mission');
        const data = await res.json();

        if (data.status === 'success') {
          // 結果に人数があれば表示
          if (data.latest_count !== undefined) {
            status.textContent = `ミッション完了！検出人数: ${data.latest_count}人。ページを更新して履歴を確認してください。`;
          } else {
            status.textContent = 'ミッション完了！ページを更新して結果を確認してください。';
          }
          status.style.background = '#d4edda';
          setTimeout(() => location.reload(), 3000); // 3秒後に自動更新
        } else {
          status.textContent = 'エラー: ' + data.message;
          status.style.background = '#f8d7da';
        }
      } catch (e) {
        status.textContent = '通信エラーが発生しました。サーバーの状況を確認してください。';
        status.style.background = '#f8d7da';
      } finally {
        btn.disabled = false;
      }
    }
  </script>
</body>
</html>
""", latest_image=latest_image, latest_count=latest_count, total_count=total_count, rows=rows)

@app.route('/start-mission', methods=['GET', 'POST'])
def start_mission():
    # --- ここがインデントエラーの箇所でした。処理を追加します ---
    try:
        print(f"[SERVER] Mission Triggered at {datetime.now()}")
        # main.py を呼び出す（タイムアウトを長めに設定するか、完了を待つ）
        result = subprocess.run(
            [PYTHON_EXE, MAIN_SCRIPT],
            cwd=BASE_DIR,
            capture_output=True,
            text=True
        )
        # ミッション後に最新の検出人数を取得
        latest_count = None
        if DETECTIONS_JSONL.exists():
          try:
            with open(DETECTIONS_JSONL, "r", encoding="utf-8") as f:
              lines = [l.strip() for l in f if l.strip()]
            if lines:
              last = json.loads(lines[-1])
              person_ids = last.get('person_ids', [])
              latest_count = len(person_ids) if isinstance(person_ids, list) else None
          except Exception:
            latest_count = None

        if result.returncode == 0:
          resp = {
            "status": "success",
            "message": "Mission completed successfully",
            "output": result.stdout,
          }
          if latest_count is not None:
            resp["latest_count"] = latest_count
          return jsonify(resp)
        else:
            return jsonify({
                "status": "error",
                "message": "main.py failed",
                "error": result.stderr
            }), 500

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    # Flaskサーバーを起動
    app.run(host='0.0.0.0', port=5000, debug=True)
