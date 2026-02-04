import socket
import threading
import time
import os
import cv2
import sys

# =========================
# 設定
# =========================
TELLO_IP = "192.168.10.1"
TELLO_PORT = 8889
TELLO_ADDR = (TELLO_IP, TELLO_PORT)
TELLO_CAMERA_ADDRESS = "udp://@0.0.0.0:11111"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 保存先をワークスペース直下（このスクリプトのあるフォルダ）に設定
SAVE_DIR = BASE_DIR
os.makedirs(SAVE_DIR, exist_ok=True)

running = True
latest_frame = None
frame_lock = threading.Lock()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("", TELLO_PORT))


def send(cmd: str, wait: float = 0.0):
    """Telloへコマンド送信。wait>0なら送信後に待つ。"""
    try:
        sock.sendto(cmd.encode("utf-8"), TELLO_ADDR)
        print(f"[SEND] {cmd}")
    except Exception as e:
        print(f"[ERR] send {cmd}: {e}")
    if wait > 0:
        time.sleep(wait)


def udp_receiver():
    """Telloからの応答を受ける（ログ用）。"""
    while running:
        try:
            data, _ = sock.recvfrom(1518)
            resp = data.decode("utf-8", errors="ignore").strip()
            if resp:
                print(f"[TELLO] {resp}")
        except:
            pass


def capture_loop():
    """映像を受け取り、最新フレームを共有する。"""
    global latest_frame, running
    cap = cv2.VideoCapture(TELLO_CAMERA_ADDRESS)
    if not cap.isOpened():
        cap = cv2.VideoCapture(TELLO_CAMERA_ADDRESS, cv2.CAP_FFMPEG)

    while running:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue
        with frame_lock:
            latest_frame = frame.copy()

    cap.release()


def save_current_frame(prefix: str):
    """最新フレームを保存する。"""
    with frame_lock:
        frame = None if latest_frame is None else latest_frame.copy()

    if frame is None:
        print("[WARN] frame is None (まだ映像が来ていません)")
        return None

    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SAVE_DIR, f"{prefix}_{ts}.jpg")
    cv2.imwrite(path, frame)
    print(f"[SAVE] {path}")
    return path


def side_scan_distance(
    direction: str = "left",   # "right" or "left"
    total_cm: int = 1200,        # 合計移動距離
    step_cm: int = 600,          # 何cmごとに撮影
    height_cm: int = 150,       # 離陸後の上昇
    settle_sec: float = 1.2,    # 停止→安定待ち（ブレ対策）
):
    """
    機首は固定のまま、横移動で直線スキャン。
    停止→撮影→横移動→停止→撮影... を繰り返します。
    """
    assert direction in ("right", "left"), "direction は 'right' か 'left'"

    print("===== Side Scan START =====")

    # 1) SDK & ストリーム開始
    send("command", wait=1.0)
    send("streamon", wait=1.8)

    # 映像が来るまで少し待つ
    time.sleep(1.0)

    # 2) 離陸・上昇
    send("takeoff", wait=4.5)
    # 明確なホバリング: 上昇前に2秒停止
    time.sleep(2.0)
    send(f"up {height_cm}", wait=3.0)

    # 3) スタート地点を撮影
    time.sleep(settle_sec)
    save_current_frame(prefix="p000")

    # 4) 横移動を刻んで撮影
    # Tello の横移動は 20〜500cm が基本なので、step_cm がこれを超える場合は分割する
    if step_cm < 20:
        print(f"[WARN] step_cm ({step_cm}) is below minimum (20). Using 20 cm steps.")
    if step_cm > 500:
        print(f"[WARN] step_cm ({step_cm}) is above maximum (500). Splitting into multiple moves.")

    move_unit = max(20, min(step_cm, 500))
    remaining = total_cm
    i = 1
    while remaining > 0 and running:
        move = min(move_unit, remaining)

        # 横移動（20〜500cmの範囲）
        send(f"{direction} {move}", wait=2.6)

        # ブレ対策：停止後に待ってから撮影
        time.sleep(settle_sec)
        save_current_frame(prefix=f"p{i:03d}")

        remaining -= move
        i += 1

    # 5) 着陸
    send("land", wait=3.0)
    print("===== Side Scan END =====")


def move_distance(direction: str, total_cm: int, settle_sec: float = 1.2, wait_per_move: float = 2.6):
    """指定方向へ合計 `total_cm` 移動する（Tello の制限 20-500 cm を分割して送る）。"""
    if total_cm <= 0:
        return

    remaining = total_cm
    i = 1
    while remaining > 0 and running:
        move = min(500, remaining)
        if move < 20:
            move = 20

        send(f"{direction} {move}", wait=wait_per_move)
        time.sleep(settle_sec)
        remaining -= move
        i += 1


def move_with_midpoint(direction: str, total_cm: int, midpoint_cm: int = 600, prefix: str = "mission", settle_sec: float = 1.2):
    """total_cm の途中 midpoint_cm で停止して写真を撮る移動。
    Tello の制約に従い内部で分割移動を行う。
    """
    if midpoint_cm <= 0 or midpoint_cm >= total_cm:
        # ミッドポイントが不正なら通常移動
        move_distance(direction, total_cm, settle_sec=settle_sec)
        return

    # 前半を移動してミッドポイントで撮影
    move_distance(direction, midpoint_cm, settle_sec=settle_sec)
    time.sleep(0.2)
    save_current_frame(prefix=f"{prefix}_mid")

    # 後半を移動
    remaining = total_cm - midpoint_cm
    if remaining > 0:
        move_distance(direction, remaining, settle_sec=settle_sec)


def perform_mission(height_cm: int = 100, seg_cm: int = 250, settle_sec: float = 1.2):
    """
    離陸 → 1m上昇 → 撮影
    → 2.5m左 → 停止して撮影
    → 2.5m左
    → 45度左回転×4（合計180度）
    → 2.5m左 → 停止して撮影
    → 2.5m左
    → 着陸
    """
    global running

    print("===== Mission START =====")

    # SDK開始 & 映像開始
    send("command", wait=1.0)
    send("streamon", wait=1.5)
    time.sleep(1.0)

    # 離陸 & 上昇
    send("takeoff", wait=4.5)
    # 明確なホバリング: 上昇前に2秒停止
    time.sleep(2.0)
    send(f"up {height_cm}", wait=3.0)

    # 上昇直後は明確に2秒停止してから撮影①（上昇後）
    time.sleep(2.0)
    time.sleep(settle_sec)
    save_current_frame(prefix="mission_start")

    # 2.5m左 → 撮影②
    move_distance("left", seg_cm, settle_sec=settle_sec)
    # 区間完了後は明確に2秒停止して撮影
    time.sleep(2.0)
    save_current_frame(prefix="mission_step1")

    # 2.5m左（撮影なし）
    time.sleep(2.0)  # 明確な停止: 2秒間ホバリング
    move_distance("left", seg_cm, settle_sec=settle_sec)
    # 移動後に2秒停止
    time.sleep(2.0)
    # 撮影②：2回目の左移動後にその場で撮影
    save_current_frame(prefix="mission_step2")

    # 45度左回転×4
    for i in range(4):
        if not running:
            break
        send("ccw 45", wait=2.0)
        # 回転後は明確に2秒停止
        time.sleep(2.0)
        # 各回転後に撮影
        save_current_frame(prefix=f"mission_rot{i+1:02d}")

    # 2.5m左 → 撮影③
    move_distance("left", seg_cm, settle_sec=settle_sec)
    # 区間完了後は明確に2秒停止して撮影
    time.sleep(2.0)
    save_current_frame(prefix="mission_step3")

    # 2.5m左（撮影なし）
    move_distance("left", seg_cm, settle_sec=settle_sec)
    # 最終移動後に明確に2秒停止してから着陸
    time.sleep(2.0)
    # 着陸前に撮影
    time.sleep(settle_sec)
    save_current_frame(prefix="mission_before_land")

    # 着陸
    send("land", wait=3.0)
    print("===== Mission END =====")



def preview_window():
    global running
    while running:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            time.sleep(0.01)
            continue

        cv2.putText(frame, f"Saving to: {SAVE_DIR}  (L=land, Q=quit)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Tello Side Scan Preview", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('l'):   # Lで即着陸
            print("[HOTKEY] LAND!")
            send("land")
            running = False
            break

        if key == ord('q'):   # Qで終了（着陸）
            send("land")
            running = False
            break

    cv2.destroyAllWindows()


    """プレビュー表示（qで終了&着陸）"""
    while running:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.01)
            continue

        cv2.putText(frame, f"Saving to: {SAVE_DIR}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Tello Side Scan Preview (press q to land & quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            running = False
            send("land")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    threading.Thread(target=udp_receiver, daemon=True).start()
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=preview_window, daemon=True).start()

    if "--auto" in sys.argv:
        # 例：右へ5mを50cm刻みで撮影（start+10枚）
        side_scan_distance(direction="left", total_cm=1200, step_cm=600, height_cm=150, settle_sec=1.2)
        running = False
        time.sleep(1.0)
        sock.close()
    elif "--mission" in sys.argv:
        # ユーザー指定ミッション: 左に7m (途中3.5mで撮影) -> 45度左回転×4 -> 左に7m (途中3.5mで撮影) -> 着陸
        perform_mission(height_cm=100, seg_cm=250, settle_sec=1.2)
        running = False
        time.sleep(1.0)
        sock.close()
    else:
        print("起動しました。")
        print("自動スキャン: python script.py --auto")
        print("保存先:", SAVE_DIR)
        print("終了: プレビュー画面で q")
        try:
            while running:
                time.sleep(0.1)
        finally:
            running = False
            send("land")
            time.sleep(1.0)
            sock.close()
