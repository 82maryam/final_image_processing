import os, json ,re,hashlib
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN



SAMPLE_FPS = 3.0
DET_SIZE = (768, 768)
MIN_FACE_SIZE = 36
DBSCAN_EPS = 0.34
DBSCAN_MIN_SAMPLES = 4
MATCH_COS_SIM = 0.52
CROP_MARGIN = 0.30


def ensure_dir(p): os.makedirs(p, exist_ok=True)

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

def variance_of_laplacian(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
#---------------------------------------------------------------------------------------------------------------------- 

USER_RE = re.compile(r'^user_(\d+)_video_(\d+)$', re.IGNORECASE)

def parse_userid(video_path: str) -> str:
    base = os.path.splitext(os.path.basename(video_path))[0]
    m = USER_RE.match(base)
    if not m:
        raise ValueError(f"Bad filename: {base}. Expected user_<id>_video_<n>.mp4")
    return m.group(1)


def fingerprint(video_path: str, head_mb: int = 8) -> dict:
    st = os.stat(video_path)
    h = hashlib.sha1()
    with open(video_path, "rb") as f:
        h.update(f.read(head_mb * 1024 * 1024)) 
    return {
        "size": int(st.st_size),
        "mtime": int(st.st_mtime),
        "head_sha1": h.hexdigest()
    }
#---------------------------------------------------------------------------------------------------------------------

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
    return img[y1m:y2m, x1m:x2m].copy()

def quality_score(face_crop_bgr, det_score: float):
    h, w = face_crop_bgr.shape[:2]
    area = w * h
    gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
    sharp = variance_of_laplacian(gray)
    area_term = np.log(1 + area)
    sharp_term = np.log(1 + max(sharp, 0.0))
    return  0.8 * float(area_term) + 0.6 * float(sharp_term)

def build_face_analyzer():
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=DET_SIZE)
    return app

def iter_video_frames(video_path: str, sample_fps: float):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / sample_fps)))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % step == 0:
            yield idx, frame
        idx += 1
    cap.release()

def extract_faces(video_path, app):
    items = []
    for frame_idx, frame in iter_video_frames(video_path, SAMPLE_FPS):
        faces = app.get(frame)
        if not faces:
            continue
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(np.float32).tolist()
            det_score = float(getattr(f, "det_score", 0.0))
            emb = np.array(f.embedding, dtype=np.float32)
            crop = crop_with_margin(frame, (x1, y1, x2, y2))
            q = quality_score(crop, det_score)
            items.append({
                "frame_idx": int(frame_idx),
                "det_score": det_score,
                "quality": float(q),
                "embedding": emb,
                "crop": crop
            })
    return items

def cluster_people(face_items):
    if not face_items:
        return []

    X = np.stack([it["embedding"] for it in face_items], axis=0)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    labels = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        metric="cosine",
        n_jobs=-1
    ).fit(X).labels_

    clusters = {}
    for it, lb in zip(face_items, labels):
        if lb == -1:
            continue
        clusters.setdefault(int(lb), []).append(it)

    return sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True)

def pick_rep_and_centroid(items):
    best = max(items, key=lambda it: it["quality"])
    embs = np.stack([it["embedding"] for it in items], axis=0)
    c = embs.mean(axis=0).astype(np.float32)
    c = c / (np.linalg.norm(c) + 1e-9)
    return best, c

def save_people(video_tag, clusters, out_dir):
    ensure_dir(out_dir)
    people = []
    for i, (_, items) in enumerate(clusters, start=1):
        pid = f"{i:02d}"
        pdir = os.path.join(out_dir, pid)
        ensure_dir(pdir)

        best, centroid = pick_rep_and_centroid(items)
        best_path = os.path.join(pdir, "best.jpg")
        cv2.imwrite(best_path, best["crop"])

        people.append({
            "person_id": pid,
            "best_image_path": best_path,
            "centroid": centroid
        })
    return people

def make_side_by_side(img1_path, img2_path, out_path):
    a = cv2.imread(img1_path); b = cv2.imread(img2_path)
    if a is None or b is None: return False
    h = max(a.shape[0], b.shape[0])

    def pad(img):
        if img.shape[0] == h: return img
        pad = h - img.shape[0]
        top = pad//2; bot = pad-top
        return cv2.copyMakeBorder(img, top, bot, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))

    s = np.concatenate([pad(a), pad(b)], axis=1)
    cv2.imwrite(out_path, s)
    return True

def run_pipeline(reference_video_path: str, other_video_paths: list[str], out_root: str):

    ensure_dir(out_root)
    app = build_face_analyzer()

    ref_items = extract_faces(reference_video_path, app)
    ref_clusters = cluster_people(ref_items)
    ref_dir = os.path.join(out_root, "reference")
    ref_people = save_people("reference", ref_clusters, ref_dir)

    matches = []
    for vp in other_video_paths:
        vid_name = os.path.splitext(os.path.basename(vp))[0]
        vid_out = os.path.join(out_root, "videos", vid_name)
        ensure_dir(vid_out)

        v_items = extract_faces(vp, app)
        v_clusters = cluster_people(v_items)
        v_people = save_people(vid_name, v_clusters, vid_out)

        for rp in ref_people:
            best_sim = -1.0
            best_v = None
            for vppl in v_people:
                sim = cosine_sim(rp["centroid"], vppl["centroid"])
                if sim > best_sim:
                    best_sim = sim
                    best_v = vppl

            if best_v is not None and best_sim >= MATCH_COS_SIM:
                pair_dir = os.path.join(out_root, "matches")
                ensure_dir(pair_dir)
                pair_path = os.path.join(pair_dir, f"{rp['person_id']}__{vid_name}__sim_{best_sim:.3f}.jpg")
                make_side_by_side(rp["best_image_path"], best_v["best_image_path"], pair_path)
                matches.append({
                    "ref_person": rp["person_id"],
                    "video_name": vid_name,
                    "similarity": float(best_sim),
                    "pair_image_path": pair_path
                })

    return {
        "reference_people": ref_people,
        "matches": matches
    }
