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

for d in [IMAGES_DIR, OUT_DIR]: d.mkdir(exist_ok=True)

mission_running = False

@app.route('/out/<path:filename>')
def send_out_file(filename):
    return send_from_directory(OUT_DIR, filename)

@app.route('/')
def dashboard():
    all_missions = []
    
    if DETECTIONS_JSONL.exists():
        with open(DETECTIONS_JSONL, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            all_data = [json.loads(line) for line in lines]
            
            # 5枚1セットでミッションを分割
            for i in range(0, len(all_data), 5):
                chunk = all_data[i : i + 5]
                if len(chunk) < 5: continue # 5枚揃っていない回はスキップ
                
                mission_info = {
                    "id": i // 5 + 1,
                    "time": chunk[0].get('timestamp', '-'),
                    "total": sum(item.get('person_count', 0) for item in chunk),
                    "images": [item.get('file', '') for item in chunk]
                }
                all_missions.append(mission_info)

    # 履歴を新しい順（降順）に並べ替え
    all_missions.reverse()
    
    latest_mission = all_missions[0] if all_missions else None
    past_missions = all_missions[1:] if len(all_missions) > 1 else []

    return render_template_string("""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8"><title>Drone Mission History</title>
    <style>
        body { background: #0d1117; color: #c9d1d9; font-family: sans-serif; text-align: center; padding: 20px; }
        .container { max-width: 1000px; margin: auto; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 25px; margin-bottom: 30px; }
        .total-count { font-size: 64px; color: #00ffcc; font-weight: bold; }
        .image-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-top: 20px; }
        .img-item { background: #000; border: 1px solid #444; border-radius: 4px; overflow: hidden; }
        .img-item img { width: 100%; aspect-ratio: 4/3; object-fit: cover; }
        
        .history-section { text-align: left; margin-top: 50px; border-top: 1px solid #30363d; padding-top: 20px; }
        .history-item { background: #0d1117; border: 1px solid #30363d; margin-bottom: 10px; padding: 15px; border-radius: 8px; }
        .history-header { display: flex; justify-content: space-between; cursor: pointer; }
        .btn { background: #00ffcc; color: #0d1117; padding: 15px 40px; font-weight: bold; border: none; border-radius: 8px; cursor: pointer; font-size: 18px; }
        .btn:disabled { background: #444; }
        .badge { background: #00ffcc22; color: #00ffcc; padding: 2px 8px; border-radius: 4px; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚁 ドローン有人検知アーカイブ</h1>

        {% if latest %}
        <div class="card">
            <h2 style="color:#888; margin:0;">Latest Mission: {{ latest.time }}</h2>
            <div style="margin-top:10px;">合計検出人数: <span class="total-count">{{ latest.total }} 名</span></div>
            <div class="image-grid">
                {% for img in latest.images %}
                <div class="img-item"><img src="/out/{{ img }}"></div>
                {% endfor %}
            </div>
        </div>
        {% endif %}

        <button id="btn" class="btn" onclick="start()">NEW MISSION START</button>
        <p id="status" style="color: #f1c40f;"></p>

        <div class="history-section">
            <h3>過去のミッション履歴 ({{ past|length }}件)</h3>
            {% for m in past %}
            <div class="history-item">
                <div class="history-header" onclick="toggle('m{{m.id}}')">
                    <span><b>Mission #{{ m.id }}</b> - {{ m.time }}</span>
                    <span><span class="badge">{{ m.total }}名検知</span> ▼</span>
                </div>
                <div id="m{{m.id}}" style="display:none; margin-top:15px;">
                    <div class="image-grid">
                        {% for img in m.images %}
                        <div class="img-item"><img src="/out/{{ img }}"></div>
                        {% endfor %}
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>

    <script>
        function toggle(id) {
            const el = document.getElementById(id);
            el.style.display = el.style.display === 'none' ? 'grid' : 'none';
        }

        async function start() {
            if(!confirm('ミッションを開始しますか？')) return;
            const btn = document.getElementById('btn');
            btn.disabled = true;
            document.getElementById('status').innerText = '⚠️ ミッション実行中... 完了までお待ちください。';
            const res = await fetch('/start-mission');
            if(res.ok) {
                alert('開始しました。完了後にリロードします。');
                setTimeout(() => location.reload(), 150000);
            }
        }
    </script>
</body>
</html>
""", latest=latest_mission, past=past_missions)

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
