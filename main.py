import os
import sys
import time
import glob
import subprocess
from pathlib import Path

# =========================
# 設定, ここだけ変えればOK
# =========================
TELLO_PROFILE = "TELLO-CCE182"     # netsh wlan show profiles に出る "プロファイル名" を入れる
CAMPUS_PROFILE = "Rits-1Xauth"     # 学内WiFiのプロファイル名

PYTHON_EXE = sys.executable

MOVE_SCRIPT = "drone_move.py"      # ← 修正済み
MOVE_ARGS = ["--mission"]          # ["--mission"] or ["--auto"] など

SCAN_SCRIPT = "scan3.py"
SCAN_ARGS = []                     # scan3に引数を足すならここ
SCAN_T_SAME = "0.25"               # scan3の入力待ちに自動入力する値, 例 "0.23" 等

PROJECT_DIR = Path(__file__).resolve().parent
RESULT_DIR = PROJECT_DIR / "out"
RESULT_GLOB = str(RESULT_DIR / "*_detid.jpg")
DETECTIONS_JSONL = PROJECT_DIR / "detections.jsonl"

WAIT_AFTER_WIFI_SWITCH_SEC = 2.0   # 接続後の安定待ち
WIFI_SWITCH_TIMEOUT_SEC = 20.0     # SSID確認で待つ最大秒


# =========================
# netshの出力デコード対策
# =========================
def _safe_decode(b: bytes) -> str:
    if not b:
        return ""
    # Windowsの環境によりutf-8/cp932が混在するので両対応
    for enc in ("utf-8", "cp932"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", errors="replace")


def run_cmd_bytes(cmd, check=False):
    print("[CMD]", " ".join(cmd))
    return subprocess.run(cmd, check=check, capture_output=True)


def wifi_profiles_windows_text():
    res = run_cmd_bytes(["netsh", "wlan", "show", "profiles"], check=False)
    return _safe_decode(res.stdout) + "\n" + _safe_decode(res.stderr)


def current_wifi_ssid_windows():
    res = run_cmd_bytes(["netsh", "wlan", "show", "interfaces"], check=False)
    txt = _safe_decode(res.stdout) + "\n" + _safe_decode(res.stderr)

    # "SSID : xxxx" を探す（BSSID行は除外）
    for line in txt.splitlines():
        if "SSID" in line and "BSSID" not in line:
            parts = line.split(":")
            if len(parts) >= 2:
                return parts[1].strip()
    return None


def connect_wifi_windows(profile_name: str) -> bool:
    res = run_cmd_bytes(["netsh", "wlan", "connect", f"name={profile_name}"], check=False)

    # netsh自体の失敗
    if res.returncode != 0:
        print("[WARN] netsh connect failed")
        print(_safe_decode(res.stdout).strip())
        print(_safe_decode(res.stderr).strip())
        return False

    # 実際に繋がったかをSSIDで確認する
    t0 = time.time()
    while time.time() - t0 < WIFI_SWITCH_TIMEOUT_SEC:
        ssid = current_wifi_ssid_windows()
        if ssid == profile_name:
            time.sleep(WAIT_AFTER_WIFI_SWITCH_SEC)
            print("[OK] connected:", ssid)
            return True
        time.sleep(0.5)

    print("[WARN] wifi switch timeout")
    print("[INFO] current ssid:", current_wifi_ssid_windows())
    return False


def ensure_scripts_exist():
    for s in [MOVE_SCRIPT, SCAN_SCRIPT]:
        p = PROJECT_DIR / s
        if not p.exists():
            raise FileNotFoundError(f"script not found: {p}")


def run_python_script(script_name: str, args=None, stdin_text: str | None = None):
    if args is None:
        args = []
    script_path = str(PROJECT_DIR / script_name)
    cmd = [PYTHON_EXE, script_path] + args
    print("[RUN]", " ".join(cmd))

    if stdin_text is None:
        p = subprocess.run(cmd, cwd=str(PROJECT_DIR))
    else:
        p = subprocess.run(cmd, cwd=str(PROJECT_DIR), input=stdin_text, text=True)

    if p.returncode != 0:
        raise RuntimeError(f"script failed: {script_name}, code={p.returncode}")


def latest_result_image():
    files = glob.glob(RESULT_GLOB)
    if not files:
        return None
    files.sort(key=lambda x: os.path.getmtime(x))
    return files[-1]


def open_file(path: str):
    try:
        os.startfile(path)  # Windows
    except Exception as e:
        print("[WARN] cannot open file:", e)


def print_scan_summary():
    if not DETECTIONS_JSONL.exists():
        print("[INFO] detections.jsonl not found")
        return
    try:
        with open(DETECTIONS_JSONL, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print("[INFO] detections.jsonl lines:", len(lines))
        if lines:
            print("[INFO] last record preview:", lines[-1].strip()[:240])
    except Exception as e:
        print("[WARN] failed to read detections.jsonl:", e)


def main():
    print("[INFO] main.py path =", __file__)
    ensure_scripts_exist()

    print("=== STEP 0, check wifi profiles ===")
    prof_text = wifi_profiles_windows_text()
    if TELLO_PROFILE not in prof_text or CAMPUS_PROFILE not in prof_text:
        print("[WARN] WiFi profile name might not match.")
        print("[HINT] run: netsh wlan show profiles")
        print("[HINT] set TELLO_PROFILE and CAMPUS_PROFILE to EXACT profile names.")

    print("=== STEP 1, switch WiFi to TELLO ===")
    ok = connect_wifi_windows(TELLO_PROFILE)
    if not ok:
        print("[ABORT] cannot connect to TELLO wifi, so drone_move.py will NOT run.")
        return

    print("=== STEP 2, run drone_move.py ===")
    # drone_move.pyはTELLO接続が前提
    run_python_script(MOVE_SCRIPT, MOVE_ARGS)

    print("=== STEP 3, switch WiFi back to campus ===")
    ok = connect_wifi_windows(CAMPUS_PROFILE)
    if not ok:
        print("[WARN] failed to connect back to campus wifi, but will continue to scan3.py")

    print("=== STEP 4, run scan3.py ===")
    # scan3.pyの閾値入力を自動で渡す（Enter待ちで止まらない）
    run_python_script(SCAN_SCRIPT, SCAN_ARGS, stdin_text=SCAN_T_SAME + "\n")

    print("=== STEP 5, show results ===")
    print_scan_summary()

    img = latest_result_image()
    if img is None:
        print("[WARN] result image not found in out/.")
        print("[HINT] check scan3 output naming or RESULT_GLOB.")
    else:
        print("[OK] latest result:", img)
        open_file(img)

    print("=== DONE ===")


if __name__ == "__main__":
    main()
