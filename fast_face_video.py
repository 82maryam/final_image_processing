import os
import json
import cv2
import math
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from insightface.app import FaceAnalysis


SAMPLE_FPS = 4.0          
DET_SIZE = (640, 640)    
MIN_FACE_SIZE = 40     
DBSCAN_EPS = 0.32      
DBSCAN_MIN_SAMPLES = 4  
MATCH_COS_SIM = 0.45      
CROP_MARGIN = 0.25       


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def clamp_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1: x2 = min(w - 1, x1 + 1)
    if y2 <= y1: y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2

def crop_with_margin(img, bbox, margin=CROP_MARGIN):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    mx = bw * margin
    my = bh * margin
    x1m, y1m, x2m, y2m = clamp_bbox(x1 - mx, y1 - my, x2 + mx, y2 + my, w, h)
    return img[y1m:y2m, x1m:x2m].copy(), (x1m, y1m, x2m, y2m)

def quality_score(face_crop_bgr, det_score: float, bbox):

    h, w = face_crop_bgr.shape[:2]
    area = w * h
    gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
    sharp = variance_of_laplacian(gray)

    area_term = math.log(1 + area)              
    sharp_term = math.log(1 + max(sharp, 0.0))  

    return 1.6 * float(det_score) + 0.8 * area_term + 0.6 * sharp_term


def build_face_analyzer():
    app = FaceAnalysis(
        name="buffalo_l",  
        providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=-1, det_size=DET_SIZE)
    return app

def iter_video_frames(video_path: str, sample_fps: float):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        fps = 25.0

    step = max(1, int(round(fps / sample_fps)))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            ts_sec = idx / fps
            yield idx, ts_sec, frame
        idx += 1

    cap.release()

def extract_faces_from_video(video_path: str, app, out_debug_dir=None):

    results = []
    pbar = tqdm(list(iter_video_frames(video_path, SAMPLE_FPS)), desc=f"Scanning {os.path.basename(video_path)}")

    for frame_idx, ts, frame_bgr in pbar:
        faces = app.get(frame_bgr)
        if not faces:
            continue

        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(np.float32).tolist()
            bw = x2 - x1
            bh = y2 - y1
            if min(bw, bh) < MIN_FACE_SIZE:
                continue

            det_score = float(getattr(f, "det_score", 0.0))
            emb = np.array(f.embedding, dtype=np.float32)

            crop, bbox_m = crop_with_margin(frame_bgr, (x1, y1, x2, y2), margin=CROP_MARGIN)
            q = quality_score(crop, det_score, bbox_m)

            results.append({
                "video": os.path.basename(video_path),
                "frame_idx": int(frame_idx),
                "ts_sec": float(ts),
                "bbox": [int(bbox_m[0]), int(bbox_m[1]), int(bbox_m[2]), int(bbox_m[3])],
                "det_score": det_score,
                "quality": float(q),
                "embedding": emb,
                "crop_bgr": crop
            })

    return results

def cluster_identities(face_items):

    if len(face_items) == 0:
        return []

    X = np.stack([it["embedding"] for it in face_items], axis=0)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    clustering = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        metric="cosine",
        n_jobs=-1
    ).fit(X)

    labels = clustering.labels_.astype(int)

    clusters = {}
    for it, lb in zip(face_items, labels):
        if lb == -1:
            continue
        clusters.setdefault(lb, []).append(it)

    cluster_list = sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True)
    return cluster_list

def pick_representative(cluster_items):
    best = max(cluster_items, key=lambda it: it["quality"])
    embs = np.stack([it["embedding"] for it in cluster_items], axis=0)
    centroid = embs.mean(axis=0).astype(np.float32)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
    return best, centroid

def save_cluster_outputs(video_name, clusters, out_dir):

    ensure_dir(out_dir)
    people = []
    for i, (label, items) in enumerate(clusters, start=1):
        person_id = f"Person_{i:02d}"
        person_dir = os.path.join(out_dir, person_id)
        ensure_dir(person_dir)

        best, centroid = pick_representative(items)
        best_path = os.path.join(person_dir, "best.jpg")
        cv2.imwrite(best_path, best["crop_bgr"])

        people.append({
            "person_id": person_id,
            "cluster_label": int(label),
            "num_samples": int(len(items)),
            "best": {
                "video": best["video"],
                "frame_idx": best["frame_idx"],
                "ts_sec": best["ts_sec"],
                "det_score": best["det_score"],
                "quality": best["quality"],
                "bbox": best["bbox"],
                "image_path": os.path.relpath(best_path, out_dir)
            },
            "centroid_embedding": centroid.tolist()
        })

    report_path = os.path.join(out_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "video": video_name,
            "people_count": len(people),
            "people": people
        }, f, ensure_ascii=False, indent=2)

    return people, report_path

def load_people(report_json_path):
    with open(report_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    people = data["people"]
    for p in people:
        p["centroid_embedding"] = np.array(p["centroid_embedding"], dtype=np.float32)
    return people

def make_side_by_side(img1_path, img2_path, out_path):
    a = cv2.imread(img1_path)
    b = cv2.imread(img2_path)
    if a is None or b is None:
        return False
    h = max(a.shape[0], b.shape[0])
    def pad_to_h(img, h):
        if img.shape[0] == h:
            return img
        pad = h - img.shape[0]
        top = pad // 2
        bot = pad - top
        return cv2.copyMakeBorder(img, top, bot, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
    a2 = pad_to_h(a, h)
    b2 = pad_to_h(b, h)
    s = np.concatenate([a2, b2], axis=1)
    cv2.imwrite(out_path, s)
    return True

def match_reference_to_video(ref_people, vid_people, ref_dir, vid_dir, out_match_dir):
    ensure_dir(out_match_dir)
    matches = []

    for rp in ref_people:
        best_sim = -1.0
        best_vp = None

        for vp in vid_people:
            sim = cosine_sim(rp["centroid_embedding"], vp["centroid_embedding"])
            if sim > best_sim:
                best_sim = sim
                best_vp = vp

        if best_vp is not None and best_sim >= MATCH_COS_SIM:
            ref_best = os.path.join(ref_dir, rp["person_id"], "best.jpg")
            vid_best = os.path.join(vid_dir, best_vp["person_id"], "best.jpg")
            out_pair = os.path.join(out_match_dir, f"{rp['person_id']}__{best_vp['person_id']}__sim_{best_sim:.3f}.jpg")
            make_side_by_side(ref_best, vid_best, out_pair)

            matches.append({
                "ref_person": rp["person_id"],
                "video_person": best_vp["person_id"],
                "cosine_sim": float(best_sim),
                "pair_image": os.path.relpath(out_pair, out_match_dir)
            })

    out_json = os.path.join(out_match_dir, "matches.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "match_threshold_cos_sim": MATCH_COS_SIM,
            "matches": matches
        }, f, ensure_ascii=False, indent=2)

    return matches, out_json

def process_one_video(video_path: str, app, out_root: str):
    video_name = os.path.basename(video_path)
    out_dir = os.path.join(out_root, os.path.splitext(video_name)[0])
    ensure_dir(out_dir)

    face_items = extract_faces_from_video(video_path, app)
    clusters = cluster_identities(face_items)
    people, report_path = save_cluster_outputs(video_name, clusters, out_dir)
    return out_dir, report_path

def main():
    reference_video = "user_94842_video_0.mp4"
    other_videos = ["user_94225_video_71.mp4", "user_95293_video_60.mp4"]
    out_root = "output"

    for p in [reference_video] + other_videos:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Video not found next to script: {p}")

    ensure_dir(out_root)
    app = build_face_analyzer()

    print("\n[1/3] Processing reference video ...")
    ref_dir, ref_report = process_one_video(reference_video, app, out_root)
    ref_people = load_people(ref_report)

    all_results = {
        "reference": {
            "video": reference_video,
            "out_dir": ref_dir,
            "report": ref_report,
            "people_count": len(ref_people)
        },
        "comparisons": []
    }

    for v in other_videos:
        print(f"\n[2/3] Processing video: {v}")
        vid_dir, vid_report = process_one_video(v, app, out_root)
        vid_people = load_people(vid_report)

        print("[3/3] Matching with reference ...")
        match_dir = os.path.join(out_root, f"matches__{os.path.splitext(reference_video)[0]}__vs__{os.path.splitext(v)[0]}")
        matches, matches_json = match_reference_to_video(ref_people, vid_people, ref_dir, vid_dir, match_dir)

        all_results["comparisons"].append({
            "video": v,
            "video_out_dir": vid_dir,
            "video_report": vid_report,
            "matches_dir": match_dir,
            "matches_json": matches_json,
            "match_count": len(matches)
        })

    summary_path = os.path.join(out_root, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"- Summary: {summary_path}")
    print(f"- Reference people gallery: {ref_dir}")
    print("- Matches folders are inside output/")

if __name__ == "__main__":
    main()
