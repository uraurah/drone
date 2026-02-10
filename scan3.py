import os
import glob
import shutil
import cv2
import json
import numpy as np
import torch
from ultralytics import YOLO

from torchreid.reid.utils import FeatureExtractor

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
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
        img, text, (x1, max(0, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
    )

def extract_face_crop(crop):
    if crop.size == 0:
        return None

    # Haarはグレースケール前提の方が安定
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        return None

    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    pad = int(max(fw, fh) * 0.2)

    fx1 = max(0, fx - pad)
    fy1 = max(0, fy - pad)
    fx2 = min(crop.shape[1], fx + fw + pad)
    fy2 = min(crop.shape[0], fy + fh + pad)  # バグ修正, fy+fh

    face_crop = crop[fy1:fy2, fx1:fx2].copy()
    if face_crop.size == 0:
        return None
    return face_crop

def box_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    area1 = max(0, (x1_max - x1_min)) * max(0, (y1_max - y1_min))
    area2 = max(0, (x2_max - x2_min)) * max(0, (y2_max - y2_min))
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def remove_duplicate_boxes_with_reid(
    dets, embs,
    iou_threshold=0.5,
    emb_dist_threshold=0.35
):
    """
    IoUが高い重複候補でも, embeddingが十分違うなら別人として残す.
    dets: [(x1,y1,x2,y2,conf), ...]
    embs: [emb, ...]  detsと同順
    """
    if len(dets) == 0:
        return [], []

    order = sorted(range(len(dets)), key=lambda i: dets[i][4], reverse=True)
    keep = []

    for i in order:
        b1 = dets[i][:4]
        e1 = embs[i]
        dup = False
        for j in keep:
            b2 = dets[j][:4]
            iou = box_iou(b1, b2)
            if iou > iou_threshold:
                # IoUが高いなら, 見た目が同じかをReIDで確認
                d = cosine_distance(e1, embs[j])
                if d < emb_dist_threshold:
                    dup = True
                    break
                # dが大きいなら別人の可能性が高いので残す
        if not dup:
            keep.append(i)

    keep.sort()
    dets2 = [dets[i] for i in keep]
    embs2 = [embs[i] for i in keep]
    return dets2, embs2

class PersonDB:
    def __init__(self):
        self.people = []

    def ranked_matches(self, emb: np.ndarray):
        pairs = []
        for p in self.people:
            min_dist = min(cosine_distance(emb, e) for e in p["embs"])
            pairs.append((p["id"], min_dist))
        pairs.sort(key=lambda x: x[1])
        return pairs

    def add_new(self, emb: np.ndarray, face_emb: np.ndarray = None) -> int:
        new_id = len(self.people)
        face_embs = [face_emb.copy()] if face_emb is not None else []
        self.people.append({"id": new_id, "embs": [emb.copy()], "face_embs": face_embs, "n": 1})
        return new_id

    def update(self, pid: int, emb: np.ndarray, face_emb: np.ndarray = None, T_update: float = 0.20):
        p = self.people[pid]
        min_dist = min(cosine_distance(emb, e) for e in p["embs"])
        if min_dist < T_update and len(p["embs"]) < 10:
            p["embs"].append(emb.copy())
        if face_emb is not None and len(p["face_embs"]) < 5:
            p["face_embs"].append(face_emb.copy())
        p["n"] += 1

    def match(self, emb: np.ndarray, T_same: float, T_gap: float = 0.05, exclude_ids: set = None):
        if not self.people:
            return None, None

        pairs = self.ranked_matches(emb)
        if not pairs:
            return None, None

        exclude_ids = exclude_ids or set()
        filtered = [(pid, dist) for pid, dist in pairs if pid not in exclude_ids]
        if not filtered:
            return None, pairs[0][1]

        d1 = filtered[0][1]
        d2 = filtered[1][1] if len(filtered) >= 2 else 999.0

        if d1 < T_same and (d2 - d1) > T_gap:
            return filtered[0][0], d1
        return None, d1

    def verify_face(self, pid: int, face_emb: np.ndarray, face_threshold=0.4):
        p = self.people[pid]
        if len(p["face_embs"]) < 2:
            return True, 0.0
        min_dist = min(cosine_distance(face_emb, fe) for fe in p["face_embs"])
        return (min_dist < face_threshold), min_dist

def main():
    in_dir = "images"
    out_dir = "out"
    crops_dir = os.path.join(out_dir, "crops")

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        print(f"[RESET] {out_dir} ディレクトリを削除しました")

    ensure_dir(out_dir)
    ensure_dir(crops_dir)

    detections_file = "detections.jsonl"
    detection_records = []

    det_model = YOLO("yolov8m-pose.pt")
    PERSON_CLASS_ID = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"ReID device = {device}")

    extractor = FeatureExtractor(
        model_name="osnet_x1_0",
        model_path="",
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

    print(f"\n合計 {len(paths)} 枚の画像が見つかりました")
    print("ファイル: " + ", ".join([os.path.basename(p) for p in paths]))

    threshold_input = input("\n閾値 (デフォルト: 0.25, 小さいほど厳しい): ").strip()
    T_same = float(threshold_input) if threshold_input else 0.25
    print(f"閾値: {T_same}\n")

    db = PersonDB()

    # YOLO設定, 直線密集対策
    yolo_conf = 0.35
    yolo_iou = 0.70
    yolo_imgsz = 1280

    for img_idx, path in enumerate(paths):
        img = cv2.imread(path)
        if img is None:
            print(f"読み込み失敗: {path}")
            continue

        # 1, YOLOで候補を多めに拾う, 潰しにくいNMS設定
        results = det_model.predict(
            img,
            conf=yolo_conf,
            iou=yolo_iou,
            imgsz=yolo_imgsz,
            verbose=False,
            classes=[PERSON_CLASS_ID],
            max_det=300,
        )
        r = results[0]

        dets = []
        H, W = img.shape[:2]
        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                conf = float(b.conf.item())
                
                dets.append((x1, y1, x2, y2, conf))

        # 2, 重複排除は, IoUだけで消さずReIDで判定して残す
        #   先にembeddingを計算してから dedupする
        det_embs = []
        det_crops = []
        det_pad_boxes = []
        for (x1, y1, x2, y2, conf) in dets:
            crop, (cx1, cy1, cx2, cy2) = crop_with_padding(img, x1, y1, x2, y2, pad_ratio=0.05)
            if crop.size == 0:
                continue
            emb_t = extractor([crop])
            emb = emb_t[0].detach().cpu().numpy().astype(np.float32)
            det_embs.append(emb)
            det_crops.append(crop)
            det_pad_boxes.append((cx1, cy1, cx2, cy2))
        # detsとdet_embsの対応を揃えるため, detsも作り直す
        dets = [dets[i] for i in range(len(det_embs))]

        dets, det_embs = remove_duplicate_boxes_with_reid(
            dets, det_embs,
            iou_threshold=0.55,
            emb_dist_threshold=0.35
        )

        # dedup後にcrop情報を引き直す, 簡単のため再crop
        vis = img.copy()
        used_ids = set()
        frame_boxes_with_ids = []  # JSONL用にここで確定したpidを保存

        for p_idx, (x1, y1, x2, y2, conf) in enumerate(dets):
            crop, (cx1, cy1, cx2, cy2) = crop_with_padding(img, x1, y1, x2, y2, pad_ratio=0.05)
            if crop.size == 0:
                continue

            emb_t = extractor([crop])
            emb = emb_t[0].detach().cpu().numpy().astype(np.float32)

            face_crop = extract_face_crop(crop)
            face_emb = None
            if face_crop is not None:
                face_emb_t = extractor([face_crop])
                face_emb = face_emb_t[0].detach().cpu().numpy().astype(np.float32)

            pid, best_dist = db.match(emb, T_same, T_gap=0.05, exclude_ids=used_ids)

            if pid is None:
                pid = db.add_new(emb, face_emb=face_emb)
                matched = "NEW"
            else:
                face_valid = True
                face_dist = 0.0
                if face_emb is not None:
                    face_valid, face_dist = db.verify_face(pid, face_emb, face_threshold=0.4)

                if not face_valid:
                    pid = db.add_new(emb, face_emb=face_emb)
                    matched = "NEW_FACE"
                else:
                    db.update(pid, emb, face_emb=face_emb, T_update=0.20)
                    matched = "OK"

            used_ids.add(pid)

            show_dist = best_dist if best_dist is not None else 0.0

            id_dir = os.path.join(crops_dir, f"ID_{pid:03d}")
            ensure_dir(id_dir)
            crop_out = os.path.join(id_dir, f"img{img_idx:03d}_p{p_idx:02d}.jpg")
            cv2.imwrite(crop_out, crop)

            txt = f"ID {pid:03d} {matched} d={show_dist:.3f}"
            draw_person(vis, cx1, cy1, cx2, cy2, txt)

            frame_boxes_with_ids.append({
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "person_id": int(pid),
                "confidence": float(conf),
            })

        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(out_dir, f"{img_idx:03d}_{base}_detid.jpg")
        cv2.imwrite(out_path, vis)
        print(f"[{img_idx+1}/{len(paths)}] dets={len(dets)} -> {out_path}")

        detection_records.append({
            "frame_id": img_idx + 1,
            "filename": base,
            "boxes": [b["box"] for b in frame_boxes_with_ids],
            "person_ids": [b["person_id"] for b in frame_boxes_with_ids],
            "confidences": [b["confidence"] for b in frame_boxes_with_ids],
        })

    print(f"Unique persons = {len(db.people)}")
    print(f"Threshold T_same = {T_same}")

    with open(detections_file, 'w') as f:
        for record in detection_records:
            f.write(json.dumps(record) + "\n")
    print(f"\nDetections saved to {detections_file}")

if __name__ == "__main__":
    main()


