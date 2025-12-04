# tello_simple_move.py
import socket
import time

# Tello ドローンの IP／ポート
TELLO_IP   = '192.168.10.1'
TELLO_PORT = 8889

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
tello_address = (TELLO_IP, TELLO_PORT)

def send_command(cmd: str):
    """Tello にコマンドを送る関数"""
    sock.sendto(cmd.encode('utf-8'), tello_address)
    print(f"Sent command: {cmd}")

def receive_response(timeout=5):
    """Tello からのレスポンスを受け取る（タイムアウト付き）"""
    sock.settimeout(timeout)
    try:
        response, _ = sock.recvfrom(1024)
        print("Response:", response.decode('utf-8'))
    except socket.timeout:
        print("No response (timeout)")

def main():
    # SDK モード開始
    send_command("command")
    time.sleep(2)
    receive_response()

    # 離陸
    send_command("takeoff")
    # 離陸後、安定のため少し待つ
    time.sleep(5)
    receive_response()

    # 前方に 100 cm (1 m) 移動
    send_command("up 100")
    time.sleep(5)  # 移動待機（必要なら適宜長めでもOK）
    receive_response()

    # 移動後、10 秒ホバリング → ただそのまま待つ
    print("Hovering for 10 seconds...")
    time.sleep(10)

    # 着陸
    send_command("land")
    time.sleep(5)
    receive_response()

    # ソケットを閉じる
    sock.close()

if __name__ == "__main__":
    main()
