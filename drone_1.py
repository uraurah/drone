import socket
import threading
import cv2
import time
import sys
import os  # ← 追加：フォルダ作成用
import numpy as np
from ultralytics import YOLO

# ----------------------------
#      設定・保存用変数
# ----------------------------
SAVE_DIR = "sfm_images"        # 保存先フォルダ名
SAVE_INTERVAL = 0.5            # 保存間隔（秒）。0.5秒 = 1秒に2枚
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# (既存の設定はそのまま)
TELLO_IP = '192.168.10.1'
TELLO_PORT = 8889
TELLO_ADDRESS = (TELLO_IP, TELLO_PORT)
TELLO_CAMERA_ADDRESS = 'udp://@0.0.0.0:11111'

battery_text = "Battery: --%"
time_text = "Time: --s"
status_text = "Status: --"
running = True

# ソケット作成
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('', TELLO_PORT))

# --- (udp_receiver, ask, send, flight_pattern 関数は変更なしのため中略) ---
def udp_receiver():
    global battery_text, time_text, status_text
    while running:
        try:
            data, server = sock.recvfrom(1518)
            resp = data.decode('utf-8').strip()
            if resp.isdecimal(): battery_text = "Battery:" + resp + "%"
            elif resp.endswith("s"): time_text = "Time:" + resp
            else: status_text = "Status:" + resp
        except: pass

def ask():
    while running:
        try: sock.sendto('battery?'.encode(), TELLO_ADDRESS)
        except: pass
        time.sleep(2)
        try: sock.sendto('time?'.encode(), TELLO_ADDRESS)
        except: pass
        time.sleep(2)

def send(cmd):
    try:
        sock.sendto(cmd.encode('utf-8'), TELLO_ADDRESS)
        print(f"Send: {cmd}")
    except Exception as e:
        print(f"Error: {e}")

def flight_pattern():
    print("--- Flight Pattern Start ---")
    send('command'); time.sleep(1)
    send('streamon'); time.sleep(2)
    send('takeoff'); time.sleep(6)
    send('up 100'); time.sleep(4)
    send('forward 100'); time.sleep(4)
    send('back 100'); time.sleep(4)
    send('land'); time.sleep(5)
    send('streamoff')

# ----------------------------
#        メイン処理
# ----------------------------

# YOLO モデルロード
model = YOLO("yolov8n.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# スレッド開始
threading.Thread(target=udp_receiver, daemon=True).start()
threading.Thread(target=ask, daemon=True).start()

if "--auto" in sys.argv:
    threading.Thread(target=flight_pattern, daemon=True).start()

use_local = "--local" in sys.argv
camera_source = 0 if use_local else TELLO_CAMERA_ADDRESS
cap = cv2.VideoCapture(camera_source)

latest_frame = None
annotated_frame = None
frame_lock = threading.Lock()
annot_lock = threading.Lock()
inference_fps = 0.0

def capture_loop():
    global latest_frame, running
    while running:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue
        with frame_lock:
            latest_frame = frame.copy()

# ----------------------------
#   ★修正：推論＆画像保存ループ
# ----------------------------
def inference_and_save_loop():
    global latest_frame, annotated_frame, running, inference_fps
    prev_t = time.time()
    last_save_t = time.time()  # 画像保存用のタイマー

    while running:
        frame = None
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
                latest_frame = None

        if frame is None:
            time.sleep(0.01)
            continue

        # --- SfM用画像の保存ロジック ---
        now = time.time()
        if now - last_save_t >= SAVE_INTERVAL:
            # 推論用の枠線が入る前の「生フレーム」を保存
            img_filename = os.path.join(SAVE_DIR, f"tello_{int(now*100)}.jpg")
            cv2.imwrite(img_filename, frame)
            print(f"Saved: {img_filename}")
            last_save_t = now

        # --- YOLO推論処理 ---
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (640, int(h * (640/w))))
        results = model(resized, verbose=False) # verbose=Falseでログをスッキリさせる
        
        out = results[0].plot()
        out_display = cv2.resize(out, (w, h))

        with annot_lock:
            annotated_frame = out_display

        inference_fps = 1.0 / (time.time() - prev_t) if (time.time() - prev_t) > 0 else 0
        prev_t = time.time()

# スレッド起動
threading.Thread(target=capture_loop, daemon=True).start()
threading.Thread(target=inference_and_save_loop, daemon=True).start()

try:
    while True:
        display = None
        with annot_lock:
            if annotated_frame is not None:
                display = annotated_frame.copy()

        if display is not None:
            cv2.putText(display, f"{battery_text}  FPS: {inference_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Tello SfM Data Collection", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            if not use_local: send('land')
            break
finally:
    running = False
    time.sleep(0.5)
    cap.release()
    cv2.destroyAllWindows()
    sock.close()
