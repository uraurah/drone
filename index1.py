import socket
import threading
import cv2
import time
import queue
import torch
import sys
import numpy as np
from ultralytics import YOLO   # ← 新增：YOLO导入

# ----------------------------
#      グローバル変数・設定
# ----------------------------
TELLO_IP = '192.168.10.1'
TELLO_PORT = 8889
TELLO_ADDRESS = (TELLO_IP, TELLO_PORT)
TELLO_CAMERA_ADDRESS = 'udp://@0.0.0.0:11111'

battery_text = "Battery: --%"
time_text = "Time: --s"
status_text = "Status: --"
flight_thread_running = True

# ソケット作成
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('', TELLO_PORT))

# ----------------------------
#        関数定義
# ----------------------------

def udp_receiver():
    global battery_text, time_text, status_text
    while True:
        try:
            data, server = sock.recvfrom(1518)
            resp = data.decode('utf-8').strip()

            if resp.isdecimal():    
                battery_text = "Battery:" + resp + "%"
            elif resp.endswith("s"):
                time_text = "Time:" + resp
            else:
                status_text = "Status:" + resp
        except:
            pass

def ask():
    while True:
        try:
            sock.sendto('battery?'.encode(), TELLO_ADDRESS)
        except: pass
        time.sleep(2)

        try:
            sock.sendto('time?'.encode(), TELLO_ADDRESS)
        except: pass
        time.sleep(2)

def send(cmd):
    try:
        sock.sendto(cmd.encode('utf-8'), TELLO_ADDRESS)
        print(f"Send: {cmd}")
    except Exception as e:
        print(f"Error sending {cmd}: {e}")

def flight_pattern():
    print("--- Flight Pattern Start ---")
    send('command')
    time.sleep(1)
    
    send('streamon')
    time.sleep(2)
    
    send('takeoff')
    time.sleep(6)

    send('up 100')
    time.sleep(4)

    send('forward 100')
    time.sleep(4)

    send('back 100')
    time.sleep(4)

    send('land')
    time.sleep(5)
    
    print("--- Flight Pattern Finished ---")
    send('streamoff')

# ----------------------------
#        メイン処理
# ----------------------------

# YOLO モデルロード
model = YOLO("yolov8n.pt")   # ← 新增：YOLO模型加载（可以换 yolov8s/yolov8m）

# 尝试将模型移到 GPU 并启用半精度以加速（如果可用）
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model.to(device)
    if device == "cuda":
        try:
            model.model.half()
        except Exception:
            pass
    print("Model device:", device)
except Exception as e:
    print("Move model to device failed:", e)

recv_thread = threading.Thread(target=udp_receiver, daemon=True)
recv_thread.start()

ask_thread = threading.Thread(target=ask, daemon=True)
ask_thread.start()

flight_thread = threading.Thread(target=flight_pattern, daemon=True)
# 支持命令行参数 --auto 启动自动航线（谨慎：会发送 takeoff/land 等指令）
if "--auto" in sys.argv:
    flight_thread.start()
    print("Auto flight enabled: flight_pattern started")

time.sleep(1)
# 支持使用本地摄像头进行演示：运行 `python index1.py --local`
use_local = False
local_index = 0
if len(sys.argv) > 1 and sys.argv[1] in ("--local", "-l"):
    use_local = True
    # 可选第二个参数指定本地摄像头索引：python index1.py --local 1
    if len(sys.argv) > 2:
        try:
            local_index = int(sys.argv[2])
        except Exception:
            local_index = 0

camera_source = local_index if use_local else TELLO_CAMERA_ADDRESS
cap = cv2.VideoCapture(camera_source)

if not cap.isOpened():
    print("Camera not opened immediately. Waiting...")
    time.sleep(2)
    cap.open(camera_source)

print("Press 'q' to quit video and land emergency.")

# ----------------------------
#     异步抓帧与推理（提高实时性）
# ----------------------------
latest_frame = None
annotated_frame = None
frame_lock = threading.Lock()
annot_lock = threading.Lock()
running = True
inference_fps = 0.0

def capture_loop():
    global latest_frame, running
    while running:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue
        with frame_lock:
            latest_frame = frame.copy()

def inference_loop(target_width=640):
    global latest_frame, annotated_frame, running, inference_fps
    prev_t = time.time()
    while running:
        frame = None
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
                latest_frame = None

        if frame is None:
            time.sleep(0.01)
            continue

        # 缩放以加速推理
        h, w = frame.shape[:2]
        if w > target_width:
            scale = target_width / float(w)
            resized = cv2.resize(frame, (target_width, int(h * scale)))
        else:
            resized = frame

        # 推理
        try:
            results = model(resized)
            out = resized
            for r in results:
                out = r.plot()
            # 恢复到原始尺寸用于显示
            out_display = cv2.resize(out, (w, h)) if out.shape[:2] != (h, w) else out
        except Exception as e:
            print("Inference error:", e)
            out_display = frame

        with annot_lock:
            annotated_frame = out_display

        now = time.time()
        inference_fps = 1.0 / (now - prev_t) if now - prev_t > 0 else 0.0
        prev_t = now

cap_thread = threading.Thread(target=capture_loop, daemon=True)
infer_thread = threading.Thread(target=inference_loop, daemon=True)
cap_thread.start()
infer_thread.start()

try:
    while True:
        display = None
        with annot_lock:
            if annotated_frame is not None:
                display = annotated_frame.copy()

        if display is None:
            with frame_lock:
                if latest_frame is not None:
                    display = latest_frame.copy()

        if display is None:
            # nothing yet
            time.sleep(0.01)
            continue

        # 在画面上写入文字和 FPS
        cv2.putText(display, battery_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, time_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, status_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"FPS: {inference_fps:.1f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Tello YOLO Detection", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            if not use_local:
                send('land')
            break

finally:
    running = False
    time.sleep(0.2)
    cap.release()
    cv2.destroyAllWindows()
    sock.close()
