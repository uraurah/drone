import os
import glob
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# torchreid
from torchreid.reid.utils import FeatureExtractor



def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # a,b: (D,) float32
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(1.0 - np.dot(a, b) / denom)


def crop_with_padding(img, x1, y1, x2, y2, pad_ratio=0.05):
    h, w = img.shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)
    x1p = max(0, int(x1 - px))
    y1p = max(0, int(y1 - py))
    x2p = min(w - 1, int(x2 + px))
    y2p = min(h - 1, int(y2 + py))
    return img[y1p:y2p, x1p:x2p], (x1p, y1p, x2p, y2p)


def draw_person(img, x1, y1, x2, y2, text, color=(0, 255, 0)):
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        img,
        text,
        (x1, max(0, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


class PersonDB:
    def ranked_matches(self, emb: np.ndarray):
        pairs = []
        for p in self.people:
            d = cosine_distance(emb, p["emb_mean"])
            pairs.append((p["id"], d))
        pairs.sort(key=lambda x: x[1])  # dist昇順
        return pairs
    
    def __init__(self):
        self.people = []  # list of dict {id, emb_mean, n}

    def add_new(self, emb: np.ndarray) -> int:
        new_id = len(self.people)
        self.people.append({"id": new_id, "emb_mean": emb.copy(), "n": 1})
        return new_id

    def update(self, pid: int, emb: np.ndarray):
        p = self.people[pid]
        n = p["n"]
        # moving average
        p["emb_mean"] = (p["emb_mean"] * n + emb) / (n + 1)
        p["n"] = n + 1

    def match(self, emb: np.ndarray, T_same: float):
        if not self.people:
            return None, None

        best_id = None
        best_dist = 1e9
        for p in self.people:
            d = cosine_distance(emb, p["emb_mean"])
            if d < best_dist:
                best_dist = d
                best_id = p["id"]

        if best_dist < T_same:
            return best_id, best_dist
        return None, best_dist


def main():
    in_dir = "images"
    out_dir = "out"
    crops_dir = os.path.join(out_dir, "crops")
    ensure_dir(out_dir)
    ensure_dir(crops_dir)

    # 1) Person detection
    det_model = YOLO("yolov8n.pt")
    PERSON_CLASS_ID = 0

    # 2) ReID feature extractor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"ReID device = {device}")

    # OSNetが軽くて強いです.
    extractor = FeatureExtractor(
        model_name="osnet_x1_0",
        model_path="",  # empty = pretrained
        device=device,
    )

    exts = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(in_dir, e))
    paths.sort()

    if not paths:
        print("images/ に画像が見つかりません.")
        return

    # ユーザー入力: ファイル名または範囲指定
    print(f"\n合計 {len(paths)} 枚の画像が見つかりました")
    print("\n利用可能なファイル:")
    for i, p in enumerate(paths, 1):
        print(f"  {i}: {os.path.basename(p)}")
    
    print("\n範囲指定方法:")
    print("  方法1: インデックス指定 -> 開始: 1, 終了: 5")
    print("  方法2: ファイル名指定 -> 開始: test1, 終了: test5")
    print("  方法3: ファイル名パターン -> 開始: test1, 終了: test5; 開始: test11, 終了: test16")
    
    while True:
        try:
            start_input = input("\n開始 (インデックスまたはファイル名, デフォルト: 1): ").strip()
            end_input = input("終了 (インデックスまたはファイル名, デフォルト: {}): ".format(len(paths))).strip()
            
            # インデックスまたはファイル名から実際のインデックスを取得
            if start_input.isdigit():
                start_idx = int(start_input)
            else:
                # ファイル名で検索
                start_idx = None
                for i, p in enumerate(paths, 1):
                    if start_input in os.path.basename(p):
                        start_idx = i
                        break
                if start_idx is None:
                    print(f"エラー: '{start_input}' に該当するファイルが見つかりません")
                    continue
            
            if end_input.isdigit():
                end_idx = int(end_input)
            else:
                # ファイル名で検索
                end_idx = None
                for i, p in enumerate(paths, 1):
                    if end_input in os.path.basename(p):
                        end_idx = i
                        break
                if end_idx is None:
                    print(f"エラー: '{end_input}' に該当するファイルが見つかりません")
                    continue
            
            if start_idx is None:
                start_idx = 1
            if end_idx is None:
                end_idx = len(paths)
            
            if start_idx < 1 or end_idx > len(paths) or start_idx > end_idx:
                print(f"エラー: 1～{len(paths)} の範囲内で指定してください")
                continue
            
            threshold_input = input("閾値 (デフォルト: 0.35): ").strip()
            T_same = float(threshold_input) if threshold_input else 0.35
            
            break
        except ValueError:
            print("エラー: 有効な入力をしてください")

    # 範囲でフィルタリング
    start = start_idx - 1
    end = end_idx
    selected_paths = paths[start:end]
    print(f"\n選択ファイル範囲: {start_idx} ~ {end_idx} ({len(selected_paths)}枚)")
    for i, p in enumerate(selected_paths, start_idx):
        print(f"  {i}: {os.path.basename(p)}")
    print(f"閾値: {T_same}\n")

    if not paths:
        print("images/ に画像が見つかりません.")
        return

    db = PersonDB()

    for img_idx, path in enumerate(selected_paths):
        img = cv2.imread(path)
        if img is None:
            print(f"読み込み失敗: {path}")
            continue

        results = det_model.predict(img, conf=0.25, verbose=False)
        r = results[0]

        dets = []
        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                cls = int(b.cls.item())
                if cls != PERSON_CLASS_ID:
                    continue
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                conf = float(b.conf.item())
                dets.append((x1, y1, x2, y2, conf))

        vis = img.copy()
        
        
        used_ids = set()
        # 各人物のcropを作って, ReID特徴を抽出して, DBと照合
        for p_idx, (x1, y1, x2, y2, conf) in enumerate(dets):
            crop, (cx1, cy1, cx2, cy2) = crop_with_padding(img, x1, y1, x2, y2, pad_ratio=0.05)
            if crop.size == 0:
                continue

            # extractorはBGR/np.arrayも受けますが, 内部でPIL変換されます.
            emb_t = extractor([crop])  # torch tensor (1, D)
            emb = emb_t[0].detach().cpu().numpy().astype(np.float32)

            candidates = db.ranked_matches(emb)

            pid = None
            best_dist = None

            for cid, dist in candidates:
                if cid in used_ids:
                    continue
                if dist < T_same:
                    pid = cid
                    best_dist = dist
                    break

            if pid is None:
                pid = db.add_new(emb)
                matched = "NEW"
                best_dist = candidates[0][1] if candidates else 0.0
            else:
                db.update(pid, emb)
                matched = "OK"
                used_ids.add(pid)

            show_dist = best_dist


            # crop保存
            id_dir = os.path.join(crops_dir, f"ID_{pid:03d}")
            ensure_dir(id_dir)
            crop_out = os.path.join(id_dir, f"img{img_idx:03d}_p{p_idx:02d}.jpg")
            cv2.imwrite(crop_out, crop)

            # 可視化
            txt = f"ID {pid:03d} {matched} d={show_dist:.3f}"
            draw_person(vis, cx1, cy1, cx2, cy2, txt)

        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(out_dir, f"{img_idx:03d}_{base}_detid.jpg")
        cv2.imwrite(out_path, vis)
        print(f"[{img_idx+1}/{len(selected_paths)}] dets={len(dets)} -> {out_path}")

    print(f"Unique persons = {len(db.people)}")
    print(f"Threshold T_same = {T_same}")


if __name__ == "__main__":
    main()
