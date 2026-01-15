import socket
import threading
import cv2
import time
import os
import sys
import numpy as np
import torch
from ultralytics import YOLO

# ----------------------------
#      配置与全局变量
# ----------------------------
TELLO_IP = '192.168.10.1'
TELLO_PORT = 8889
TELLO_ADDRESS = (TELLO_IP, TELLO_PORT)
TELLO_CAMERA_ADDRESS = 'udp://@0.0.0.0:11111'

# 存储路径
SAVE_DIR = "scan_frames"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

battery_text = "Battery: --%"
time_text = "Time: --s"
status_text = "Status: --"
running = True
inference_fps = 0.0

# 初始化 Socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('', TELLO_PORT))

# ----------------------------
#        功能函数
# ----------------------------

def send(cmd):
    try:
        sock.sendto(cmd.encode('utf-8'), TELLO_ADDRESS)
        print(f"Send: {cmd}")
    except Exception as e:
        print(f"Error sending {cmd}: {e}")

def udp_receiver():
    global battery_text, time_text, status_text
    while running:
        try:
            data, _ = sock.recvfrom(1518)
            resp = data.decode('utf-8').strip()
            if resp.isdecimal():    
                battery_text = f"Battery: {resp}%"
            elif resp.endswith("s"):
                time_text = f"Time: {resp}"
            else:
                status_text = f"Status: {resp}"
        except: pass

def ask_status():
    while running:
        send('battery?')
        time.sleep(2)
        send('time?')
        time.sleep(2)

# ----------------------------
#    侧飞扫描逻辑 (关键修改)
# ----------------------------
def auto_flight_3d_scan():
    print("===== 准备执行侧飞建模扫描 =====")
    send("command")
    time.sleep(1)
    send("streamon")
    time.sleep(2)

    # 1. 起飞与上升
    send("takeoff")
    time.sleep(5)
    send("up 100")
    time.sleep(4)

    # 2. 转向人群 (假设人群在飞机起飞时的左侧，则左转90度)
    # 如果人群在右侧，请改为 cw 90
    send("ccw 90") 
    time.sleep(3)

    # 3. 侧向平移移动 (机头面对人群，向右侧平移，覆盖原定的 5 米航线)
    # 拆分为 5 段，每段 1 米，确保拍照清晰
    for i in range(5):
        print(f"正在扫描第 {i+1} 米...")
        send("right 100") # 因为机头转了90度，现在的 'right' 就是之前的 'forward'
        time.sleep(4)     # 停顿给 3D 建模留出清晰的照片时间

    # 4. 原地掉头 (180度) 准备返回
    send("cw 180")
    time.sleep(3)

    # 5. 侧向平移返回
    for i in range(5):
        print(f"正在返回扫描第 {i+1} 米...")
        send("right 100")
        time.sleep(4)

    # 6. 降落
    send("land")
    print("===== 建模扫描采集结束 =====")

# ----------------------------
#     视频处理与 AI 线程
# ----------------------------
latest_frame = None
annotated_frame = None
frame_lock = threading.Lock()
annot_lock = threading.Lock()

def capture_loop():
    global latest_frame, running
    cap = cv2.VideoCapture(TELLO_CAMERA_ADDRESS)
    
    # Windows 下如果打不开，尝试强制使用 FFMPEG
    if not cap.isOpened():
        cap = cv2.VideoCapture(TELLO_CAMERA_ADDRESS, cv2.CAP_FFMPEG)

    frame_count = 0
    while running:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
            
        with frame_lock:
            latest_frame = frame.copy()
        
        # 定时拍摄：每 30 帧（约1秒）保存一张高质量原图用于 3D 建模
        if frame_count % 30 == 0:
            timestamp = int(time.time() * 10)
            cv2.imwrite(f"{SAVE_DIR}/scan_{timestamp}.jpg", frame)
        
        frame_count += 1
    cap.release()

def inference_loop():
    global latest_frame, annotated_frame, running, inference_fps
    model = YOLO("yolov8n.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    prev_t = time.time()
    while running:
        frame = None
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
        
        if frame is None:
            time.sleep(0.01)
            continue

        # 推理并将结果画在图上
        results = model(frame, verbose=False)
        out = results[0].plot()

        with annot_lock:
            annotated_frame = out

        now = time.time()
        inference_fps = 1.0 / (now - prev_t) if now - prev_t > 0 else 0.0
        prev_t = now

# ----------------------------
#           主程序
# ----------------------------
if __name__ == "__main__":
    # 启动后台通讯
    threading.Thread(target=udp_receiver, daemon=True).start()
    threading.Thread(target=ask_status, daemon=True).start()
    
    # 启动视频抓取与 AI
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=inference_loop, daemon=True).start()

    # 如果输入 --auto 参数则自动起飞
    if "--auto" in sys.argv:
        threading.Thread(target=auto_flight_3d_scan, daemon=True).start()

    print("程序启动。按 'q' 键退出并降落。")
    print("照片将保存在 'scan_frames' 文件夹中用于三次元构建。")

    try:
        while True:
            display = None
            with annot_lock:
                if annotated_frame is not None:
                    display = annotated_frame.copy()
            
            if display is not None:
                # 叠加状态信息
                cv2.putText(display, f"{battery_text}  {time_text}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, f"FPS: {inference_fps:.1f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Tello 3D Scan Mode", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                send("land")
                break
    finally:
        running = False
        time.sleep(1)
        cv2.destroyAllWindows()
        sock.close()