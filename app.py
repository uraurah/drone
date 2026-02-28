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
        t = r.get('timestamp', r.get('time', '不明'))
        count = r.get('person_count', r.get('count', '?'))
        rows += f"<tr><td>{t}</td><td>{count} 人</td></tr>"
    
    return render_template_string("""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>研究室 人数検出</title>
  <style>
    body { font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
    h1 { color: #333; }
    button { padding: 12px 24px; font-size: 16px; background: #1a1a1a; color: white; border: none; border-radius: 8px; cursor: pointer; margin: 8px 0; }
    button:disabled { background: #888; }
    #status { margin: 12px 0; padding: 10px 16px; border-radius: 8px; background: #e0f0ff; min-height: 20px; }
    table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.1); margin-top: 24px; }
    th { background: #1a1a1a; color: white; padding: 10px 16px; text-align: left; }
    td { padding: 10px 16px; border-bottom: 1px solid #eee; }
    tr:last-child td { border-bottom: none; }
  </style>
</head>
<body>
  <h1>🚁 研究室 人数検出ダッシュボード</h1>
  <button id="btn" onclick="startMission()">ドローン飛行開始</button>
  <div id="status">待機中</div>

  <h2>最新の検出履歴</h2>
  <table>
    <thead>
        <tr><th>日時</th><th>検出人数</th></tr>
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
          status.textContent = 'ミッション完了！ページを更新して結果を確認してください。';
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
""")

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

        if result.returncode == 0:
            return jsonify({
                "status": "success",
                "message": "Mission completed successfully",
                "output": result.stdout
            })
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
