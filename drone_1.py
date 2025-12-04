import socket
import threading
import cv2
import time
import numpy as np

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
flight_thread_running = True # 飛行スレッド管理用

# ソケット作成
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('', TELLO_PORT))

# ----------------------------
#        関数定義
# ----------------------------

# データ受け取り用の関数
def udp_receiver():
    global battery_text, time_text, status_text
    while True:
        try:
            data, server = sock.recvfrom(1518)
            resp = data.decode(encoding="utf-8").strip()
            # バッテリー残量 (数字のみの場合はバッテリーとみなす簡易判定)
            if resp.isdecimal():    
                battery_text = "Battery:" + resp + "%"
            # 飛行時間
            elif resp.endswith("s"):
                time_text = "Time:" + resp
            else:
                status_text = "Status:" + resp
        except:
            pass

# 問い合わせ（バッテリー・飛行時間）
def ask():
    while True:
        try:
            sock.sendto('battery?'.encode(), TELLO_ADDRESS)
        except:
            pass
        time.sleep(2) # 頻度を少し下げて負荷軽減

        try:
            sock.sendto('time?'.encode(), TELLO_ADDRESS)
        except:
            pass
        time.sleep(2)

# コマンド送信関数
def send(cmd):
    try:
        sock.sendto(cmd.encode('utf-8'), TELLO_ADDRESS)
        print(f"Send: {cmd}")
    except Exception as e:
        print(f"Error sending {cmd}: {e}")

# 自動飛行ロジック（別スレッドで実行）
def flight_pattern():
    print("--- Flight Pattern Start ---")
    
    # コマンドモード開始
    send('command')
    time.sleep(1)
    
    # カメラストリーミング開始
    send('streamon')
    time.sleep(2)
    
    # 離陸
    send('takeoff')
    time.sleep(6)

    # 2m 上昇 (安全のため値を小さくテストすることをお勧めします)
    send('up 100')
    time.sleep(4)

    # 2m 前進
    send('forward 100')
    time.sleep(4)

    # 元の位置に戻る（2m後退）
    send('back 100')
    time.sleep(4)

    # 着陸
    send('land')
    time.sleep(5)
    
    print("--- Flight Pattern Finished ---")
    send('streamoff')

# ----------------------------
#        メイン処理
# ----------------------------

# 受信スレッド開始
recv_thread = threading.Thread(target=udp_receiver)
recv_thread.daemon = True
recv_thread.start()

# 問い合わせスレッド開始
ask_thread = threading.Thread(target=ask)
ask_thread.daemon = True
ask_thread.start()

# ★重要★ 飛行コマンドを別スレッドで開始
flight_thread = threading.Thread(target=flight_pattern)
flight_thread.daemon = True
flight_thread.start()

# カメラ受信準備（少し待ってから）
time.sleep(1) 
cap = cv2.VideoCapture(TELLO_CAMERA_ADDRESS)

if not cap.isOpened():
    print("Camera not opened immediately. Waiting...")
    time.sleep(2)
    cap.open(TELLO_CAMERA_ADDRESS)

print("Press 'q' to quit video and land emergency.")

# 映像を表示し続ける（メインスレッド）
while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        # 映像が来ないときのエラー回避（少し待機）
        time.sleep(0.1)
        continue

    # リサイズは不要であれば削除（負荷軽減）
    # 元コードのresizeは同じサイズ指定だったので実質無意味でした
    # 画面が大きすぎる場合のみ以下のように縮小してください
    # frame = cv2.resize(frame, (640, 480))

    # テキスト描画
    cv2.putText(frame, battery_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, time_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, status_text, (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Tello Camera View", frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        send('land') # 強制着陸コマンド
        break

# 終了処理
cap.release()
cv2.destroyAllWindows()
sock.close()
