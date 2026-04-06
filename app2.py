import threading, os, sys, time, json, cv2, subprocess
from flask import Flask, jsonify, render_template_string, Response, send_from_directory
from flask_cors import CORS
from pathlib import Path

app = Flask(__name__)
CORS(app)

# --- 設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = Path(BASE_DIR) / "images"
OUT_DIR = Path(BASE_DIR) / "out"
DETECTIONS_JSONL = Path(BASE_DIR) / "detections.jsonl"
MAIN_SCRIPT = os.path.join(BASE_DIR, "main.py")
PYTHON_EXE = sys.executable

# フォルダ初期化
for d in [IMAGES_DIR, OUT_DIR]: d.mkdir(exist_ok=True)

mission_running = False

@app.route('/out/<path:filename>')
def send_out_file(filename):
    return send_from_directory(OUT_DIR, filename)

@app.route('/')
def dashboard():
    mission_results = None
    if DETECTIONS_JSONL.exists():
        with open(DETECTIONS_JSONL, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # *** 最新の「5枚分」を1つのミッションとして集計 ***
            if len(lines) >= 5:
                latest_set = [json.loads(line) for line in lines[-5:]]
                total_count = sum(item.get('person_count', 0) for item in latest_set)
                timestamp = latest_set[0].get('timestamp', '-')
                images = [item.get('file', '') for item in latest_set]
                mission_results = {"time": timestamp, "total": total_count, "images": images}

    return render_template_string("""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8"><title>Drone AI Report</title>
    <style>
        body { background: #0d1117; color: #c9d1d9; font-family: sans-serif; text-align: center; padding: 20px; }
        .container { max-width: 900px; margin: auto; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 30px; margin-bottom: 20px; }
        .total-count { font-size: 72px; color: #00ffcc; font-weight: bold; margin: 10px 0; }
        .image-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-top: 30px; }
        .img-item { background: #000; border: 1px solid #444; border-radius: 8px; overflow: hidden; }
        .img-item img { width: 100%; height: auto; display: block; }
        .btn { background: #00ffcc; color: #0d1117; padding: 20px 60px; font-weight: bold; border: none; border-radius: 10px; cursor: pointer; font-size: 20px; }
        .btn:disabled { background: #444; color: #888; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚁 有人検知ミッション・レポート</h1>
        {% if mission %}
        <div class="card">
            <h3 style="color:#888;">ミッション完了時刻: {{ mission.time }}</h3>
            <div style="font-size: 24px;">今回の合計検出人数</div>
            <div class="total-count">{{ mission.total }} 名</div>
            <div class="image-grid">
                {% for img in mission.images %}
                <div class="img-item">
                    <img src="/out/{{ img }}">
                    <div style="font-size:11px; padding:5px;">{{ img }}</div>
                </div>
                {% endfor %}
            </div>
        </div>
        {% else %}
        <div class="card"><p>まだ解析データがありません。</p></div>
        {% endif %}
        <button id="btn" class="btn" onclick="start()">MISSION START</button>
        <p id="status" style="margin-top:20px; color:#f1c40f;"></p>
    </div>
    <script>
        async function start() {
            if(!confirm('飛行を開始しますか？')) return;
            document.getElementById('btn').disabled = true;
            document.getElementById('status').innerText = '⚠️ 実行中... 完了までブラウザを閉じないでください。';
            const res = await fetch('/start-mission');
            if(res.ok) {
                alert('ミッション開始。約2〜3分後に自動で更新されます。');
                setTimeout(() => location.reload(), 150000); 
            }
        }
    </script>
</body>
</html>
""", mission=mission_results)

def thread_task():
    global mission_running
    try:
        subprocess.run([PYTHON_EXE, MAIN_SCRIPT], cwd=BASE_DIR)
    finally:
        mission_running = False

@app.route('/start-mission')
def start_mission():
    global mission_running
    if mission_running: return jsonify({"status":"error"}), 400
    mission_running = True
    threading.Thread(target=thread_task).start()
    return jsonify({"status":"success"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
