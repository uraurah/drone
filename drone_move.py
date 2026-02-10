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
LOCAL_CMD_PORT = 9000
TELLO_ADDR = (TELLO_IP, TELLO_PORT)
TELLO_CAMERA_ADDRESS = "udp://0.0.0.0:11111"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 保存先を images ディレクトリに設定
SAVE_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(SAVE_DIR, exist_ok=True)

running = True
latest_frame = None
frame_lock = threading.Lock()
save_counter = 0  # t1～t9の循環カウンター

last_resp = None
resp_lock = threading.Lock()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("", LOCAL_CMD_PORT))


def send(cmd: str, wait: float = 0.0):
    """Telloへコマンド送信。wait>0なら送信後に待つ。"""
    try:
        sock.sendto(cmd.encode("utf-8"), TELLO_ADDR)
        print(f"[SEND] {cmd}")
    except Exception as e:
        print(f"[ERR] send {cmd}: {e}")
    if wait > 0:
        time.sleep(wait)


def send_wait_ok(cmd: str, timeout=5.0):
    global last_resp
    with resp_lock:
        last_resp = None
    sock.sendto(cmd.encode("utf-8"), TELLO_ADDR)
    print(f"[SEND] {cmd}")

    t0 = time.time()
    while time.time() - t0 < timeout:
        with resp_lock:
            r = last_resp
        if r in ("ok", "error"):
            return r == "ok", r
        time.sleep(0.01)
    return False, "timeout"


def send_wait_resp(cmd: str, timeout=5.0):
    """
    battery? や height? みたいに, ok/error ではなく値を返すコマンド用.
    何か1つでも応答が来たらそれを返す.
    """
    global last_resp
    with resp_lock:
        last_resp = None

    sock.sendto(cmd.encode("utf-8"), TELLO_ADDR)
    print(f"[SEND] {cmd}")

    t0 = time.time()
    while time.time() - t0 < timeout:
        with resp_lock:
            r = last_resp
        if r is not None:
            return True, r
        time.sleep(0.01)

    return False, "timeout"


def tello_query_int(cmd: str, timeout=3.0):
    ok, resp = send_wait_resp(cmd, timeout=timeout)
    if not ok:
        return None, resp
    try:
        return int(resp), resp
    except:
        return None, resp


def safe_stream_reset():
    # ストリームを確実に初期化
    send("streamoff", wait=0.8)
    send("streamon", wait=1.5)
    time.sleep(2.5)


def recover_and_end(reason: str = "abort"):
    # 失敗時に機体をできるだけ正常状態に戻して終わる
    print(f"[RECOVER] {reason}")
    try:
        send("land", wait=3.0)
    except:
        pass
    try:
        send("streamoff", wait=0.8)
    except:
        pass
    # SDKを戻す目的で command を再送, ここでok返ると復帰してる可能性が上がる
    try:
        send("command", wait=1.0)
    except:
        pass


def udp_receiver():
    global last_resp
    while running:
        try:
            data, _ = sock.recvfrom(1518)
            resp = data.decode("utf-8", errors="ignore").strip()
            if resp:
                with resp_lock:
                    last_resp = resp
                print(f"[TELLO] {resp}")
        except:
            pass


def capture_loop():
    global latest_frame, running

    url = "udp://0.0.0.0:11111"
    print("[VIDEO] capture thread started")

    cap = None
    fail = 0

    while running:
        if cap is None or not cap.isOpened():
            print("[VIDEO] opening stream ...")
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(0.5)

        ret, frame = cap.read()
        if not ret or frame is None:
            fail += 1
            time.sleep(0.02)
            # 連続失敗したら開き直す
            if fail >= 60:  # 約1-2秒
                print("[VIDEO] no frames, reopening ...")
                try:
                    cap.release()
                except:
                    pass
                cap = None
                fail = 0
            continue

        fail = 0
        with frame_lock:
            latest_frame = frame


def save_current_frame(prefix: str = None):
    """最新フレームをt1.jpg～t9.jpgで保存する（上書き許可）。"""
    global save_counter

    with frame_lock:
        frame = None if latest_frame is None else latest_frame.copy()

    if frame is None:
        print("[WARN] フレームが受信されていません")
        print("[WARN] ドローンの電源とWi-Fi接続を確認してください")
        print("[WARN] 'streamon'コマンドが正しく実行されたか確認してください")
        return None

    # 1～9で循環保存
    save_counter = (save_counter % 9) + 1
    path = os.path.join(SAVE_DIR, f"t{save_counter}.jpg")
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
    send("streamoff", wait=0.8)
    send("streamon", wait=1.5)

    # 映像が来るまで少し待つ
    time.sleep(2.5)

    # 2) 離陸・上昇
    ok, resp = send_wait_ok("takeoff", timeout=8.0)
    if not ok:
        print(f"[ABORT] takeoff failed: {resp}")
        send("land")
        return
    # 明確なホバリング: 上昇前に2秒停止
    time.sleep(2.0)
    ok, resp = send_wait_ok(f"up {height_cm}", timeout=8.0)
    if not ok:
        print(f"[ABORT] up {height_cm} failed: {resp}")
        send("land")
        return

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
        ok, resp = send_wait_ok(f"{direction} {move}", timeout=10.0)
        if not ok:
            print(f"[ABORT] {direction} {move} failed: {resp}")
            send("land")
            return

        # ブレ対策：停止後に待ってから撮影
        time.sleep(settle_sec)
        save_current_frame(prefix=f"p{i:03d}")

        remaining -= move
        i += 1

    # 5) 着陸
    send("land", wait=3.0)
    print("===== Side Scan END =====")


def move_distance(direction: str, total_cm: int, settle_sec: float = 1.2, wait_per_move: float = 3.2):
    """指定方向へ合計 `total_cm` 移動する（Tello の制限 20-500 cm を分割して送る）。"""
    if total_cm <= 0:
        return True

    remaining = total_cm
    i = 1
    while remaining > 0 and running:
        move = min(500, remaining)
        if move < 20:
            move = 20

        ok, resp = send_wait_ok(f"{direction} {move}", timeout=10.0)
        if not ok:
            print(f"[ABORT] {direction} {move} failed: {resp}")
            return False
        time.sleep(settle_sec)
        remaining -= move
        i += 1
    return True


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
    global running

    print("===== Mission START =====")

    # SDK開始
    ok, resp = send_wait_ok("command", timeout=3.0)
    if not ok:
        recover_and_end("command failed")
        return

    # バッテリー確認（低いと赤点滅→復帰不能に入りやすい）
    bat, raw = tello_query_int("battery?", timeout=3.0)
    print(f"[TELLO] battery={bat} raw={raw}")
    if bat is None:
        recover_and_end("battery? failed")
        return
    if bat < 45:
        # 実験なら45%未満はやめた方がいい（赤点滅回避）
        print("[ABORT] battery too low, stop mission")
        recover_and_end("low battery")
        return

    # ストリーム初期化（PPS問題も減る）
    print("[STREAM] reset stream")
    safe_stream_reset()

    # 離陸
    ok, resp = send_wait_ok("takeoff", timeout=10.0)
    if not ok:
        recover_and_end(f"takeoff failed: {resp}")
        return
    time.sleep(1.0)

    # 上昇
    ok, resp = send_wait_ok(f"up {height_cm}", timeout=10.0)
    if not ok:
        recover_and_end(f"up failed: {resp}")
        return

    time.sleep(settle_sec)
    if save_current_frame() is None:
        recover_and_end("no video frames")
        return

    # 左移動1
    if not move_distance("left", seg_cm, settle_sec=settle_sec):
        recover_and_end("move left failed")
        return
    time.sleep(settle_sec)
    save_current_frame()

    # 左移動2
    if not move_distance("left", seg_cm, settle_sec=settle_sec):
        recover_and_end("move left failed")
        return
    time.sleep(settle_sec)
    save_current_frame()

    # 回転（ここがNot joystick起きやすいので待ち長め）
    for i in range(4):
        if not running:
            break
        ok, resp = send_wait_ok("ccw 45", timeout=10.0)
        if not ok:
            recover_and_end(f"ccw failed: {resp}")
            return
        time.sleep(max(1.8, settle_sec))  # 回転後は余分に待つ
        save_current_frame()

    # 左移動3
    if not move_distance("left", seg_cm, settle_sec=settle_sec):
        recover_and_end("move left failed")
        return
    time.sleep(settle_sec)
    save_current_frame()

    # 左移動4
    if not move_distance("left", seg_cm, settle_sec=settle_sec):
        recover_and_end("move left failed")
        return
    time.sleep(settle_sec)
    save_current_frame()

    # 着陸前に180度回転（撮影なし）
    ok, resp = send_wait_ok("ccw 180", timeout=10.0)
    if not ok:
        recover_and_end(f"ccw 180 failed: {resp}")
        return
    time.sleep(2.0)  # 回転後の安定待ち

    # 正常終了シーケンス（ここが大事）
    send("land", wait=3.5)
    send("streamoff", wait=0.8)
    send("command", wait=1.0)  # 次回接続を楽にする保険
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


if __name__ == "__main__":
    # UDP受信スレッドを開始
    threading.Thread(target=udp_receiver, daemon=True).start()

    # 映像キャプチャスレッドを開始（先に開始して準備を待つ）
    threading.Thread(target=capture_loop, daemon=True).start()
    # time.sleep(2.0)  # capture_loop内で待つので削除

    # プレビューウィンドウスレッドを開始
    threading.Thread(target=preview_window, daemon=True).start()

    if "--auto" in sys.argv:
        # 例：右へ5mを50cm刻みで撮影（start+10枚）
        side_scan_distance(direction="left", total_cm=1200, step_cm=600, height_cm=150, settle_sec=1.2)
        running = False
        time.sleep(1.0)
        sock.close()
    elif "--mission" in sys.argv:
        # ミッション実行: 離陸→上昇→2.5m左×2→回転×4→2.5m左×2→着陸（各所で撮影）
        perform_mission(height_cm=100, seg_cm=250, settle_sec=1.2)
        running = False
        time.sleep(1.0)
        sock.close()
    else:
        print("起動しました。")
        print("ミッション実行: python drone_move.py --mission")
        print("保存先:", SAVE_DIR)
        print("終了: プレビュー画面で q or l")
        try:
            while running:
                time.sleep(0.1)
        finally:
            running = False
            send("land")
            time.sleep(1.0)
            sock.close()

