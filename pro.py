"""
Face clustering & matching pipeline with PCA + DBSCAN.

مراحل:
1) پردازش ویدیوهای مرجع → جمع‌آوری embedding چهره‌ها
2) کاهش بعد (PCA) و خوشه‌بندی (DBSCAN)
3) انتخاب بهترین چهره در هر خوشه و ذخیره‌ی عکس + بردار
4) پردازش ویدیوهای دیگر و پیدا کردن حضور افراد مرجع
"""

import os
import glob
import math
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import face_recognition
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from joblib import dump, load

# ==========================
# تنظیمات کلی اسکریپت
# ==========================

class Config:
    # مسیرها را با توجه به سیستم خودت تنظیم کن
    REFERENCE_VIDEOS_DIR = r"hi"   # ویدیوهای مرجع
    QUERY_VIDEOS_DIR     = r"downloaded_videos"       # بقیه ویدیوها
    OUTPUT_DIR           = r"D:\path\to\output"             # خروجی

    # تنظیمات نمونه‌برداری فریم
    FRAME_STEP_SHORT = 2   # اگر ویدیو < 60 ثانیه
    FRAME_STEP_LONG  = 5   # اگر ویدیو >= 60 ثانیه

    # حداقل اندازه چهره (بر حسب پیکسل)
    MIN_FACE_SIZE = 60     # حداقل طول و عرض چهره

    # اندازه نهایی چهره‌ی crop شده
    FACE_IMAGE_SIZE = 160  # 160x160

    # مدل تشخیص چهره در face_recognition: 'hog' (سریع‌تر) یا 'cnn' (دقیق‌تر)
    FACE_DETECT_MODEL = "hog"

    # تنظیمات PCA
    USE_PCA = False
    PCA_COMPONENTS = 64  # بعد نهایی بعد از PCA

    # تنظیمات DBSCAN
    DBSCAN_EPS = 0.7
    DBSCAN_MIN_SAMPLES = 2

    # آستانه‌ی تشخیص تطبیق با افراد مرجع (فاصله‌ی اقلیدسی روی بردارهای نرمال‌شده یا PCA شده)
    MATCH_THRESHOLD = 0.6

    # ذخیره / بارگذاری مدل‌های آموزش‌دیده
    PCA_MODEL_PATH = os.path.join(OUTPUT_DIR, "pca_model.joblib")
    REPRESENTATIVES_PATH = os.path.join(OUTPUT_DIR, "representatives.json")
    REPRESENTATIVE_EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "representative_embeddings.npy")

    # پوشه‌ی ذخیره عکس چهره‌های نهایی
    REPRESENTATIVE_FACES_DIR = os.path.join(OUTPUT_DIR, "persons")

# ==========================
# ساختار داده‌ها
# ==========================

@dataclass
class FaceMeta:
    video_path: str
    frame_index: int
    bbox: Tuple[int, int, int, int]   # (top, right, bottom, left) در سیستم face_recognition
    quality_score: float
    cluster_label: Optional[int] = None

@dataclass
class RepresentativeFace:
    person_id: int
    image_path: str
    embedding_index: int
    meta: FaceMeta

# ==========================
# توابع کمکی تصویر
# ==========================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def video_duration_in_seconds(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    if fps <= 0:
        return 0.0
    return frame_count / fps


def sample_frames(video_path: str, frame_step_short: int, frame_step_long: int):
    """
    generator: (frame_index, frame_bgr)
    """
    dur = video_duration_in_seconds(video_path)
    if dur < 60:
        step = frame_step_short
    else:
        step = frame_step_long

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return

    frame_index = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_index % step == 0:
            yield frame_index, frame_bgr

        frame_index += 1

    cap.release()


def preprocess_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """
    1) تنظیم اندازه
    2) افزایش جزئی کنتراست برای کیفیت پایین
    3) تبدیل به RGB
    """
    h, w = frame_bgr.shape[:2]

    # resize بر اساس عرض
    if w > 1500:
        scale = 1280 / w
    elif w > 800:
        scale = 720 / w
    else:
        scale = 1.0

    if scale != 1.0:
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))

    # تبدیل به فضای YCrCb برای بهبود روشنایی (CLAHE)
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_eq = clahe.apply(y)

    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    frame_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

    # تبدیل به RGB برای face_recognition
    frame_rgb = cv2.cvtColor(frame_eq, cv2.COLOR_BGR2RGB)
    return frame_rgb


def compute_sharpness(gray_face: np.ndarray) -> float:
    """
    استفاده از variance of Laplacian به عنوان معیار شفافیت
    """
    lap = cv2.Laplacian(gray_face, cv2.CV_64F)
    return float(lap.var())


def align_face(face_rgb: np.ndarray, landmarks: Dict[str, List[Tuple[int, int]]],
               output_size: int) -> Tuple[np.ndarray, float]:
    """
    چرخاندن صورت بر اساس چشم ها برای align شدن.
    خروجی: تصویر RGB هم‌تراز شده و زاویه‌ی چرخش (درجه).
    اگر نتوانستیم چشم‌ها را پیدا کنیم، همان تصویر اصلی و زاویه 0 برمی‌گردد.
    """
    if "left_eye" not in landmarks or "right_eye" not in landmarks:
        return cv2.resize(face_rgb, (output_size, output_size)), 0.0

    left_eye_pts = np.array(landmarks["left_eye"])
    right_eye_pts = np.array(landmarks["right_eye"])

    left_center = left_eye_pts.mean(axis=0)
    right_center = right_eye_pts.mean(axis=0)

    dy = right_center[1] - left_center[1]
    dx = right_center[0] - left_center[0]
    angle = math.degrees(math.atan2(dy, dx))

    # چرخاندن حول مرکز تصویر
    (h, w) = face_rgb.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(face_rgb, M, (w, h), flags=cv2.INTER_LINEAR)

    aligned = cv2.resize(rotated, (output_size, output_size))
    return aligned, angle


def compute_quality_score(face_rgb: np.ndarray,
                          bbox_size: Tuple[int, int],
                          tilt_angle: float) -> float:
    """
    امتیاز کیفیت ترکیب:
    - وضوح
    - اندازه چهره
    - میزان کج بودن
    """
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    sharpness = compute_sharpness(gray)

    w, h = bbox_size
    area = w * h

    # نرمال‌سازی ساده (تنها برای مقیاس دهی نسبی)
    sharp_norm = sharpness
    area_norm = area ** 0.5      # ریشه دوم برای کاهش مقیاس
    angle_penalty = abs(tilt_angle)

    # وزن‌ها را می‌توانی تنظیم کنی
    score = 0.5 * sharp_norm + 0.3 * area_norm - 0.2 * angle_penalty
    return float(score)


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

# ==========================
# پردازش چهره در یک فریم
# ==========================

def process_frame_faces(frame_rgb: np.ndarray,
                        video_path: str,
                        frame_index: int,
                        config: Config) -> Tuple[List[np.ndarray], List[FaceMeta], List[np.ndarray]]:
    """
    برای یک فریم:
    - تشخیص چهره
    - crop + align
    - محاسبه quality
    - استخراج embedding

    خروجی:
    - لیست تصویر چهره (RGB)
    - لیست meta
    - لیست embedding (np.ndarray)
    """
    faces_imgs: List[np.ndarray] = []
    faces_meta: List[FaceMeta] = []
    embeddings: List[np.ndarray] = []

    # تشخیص چهره: locations = [(top, right, bottom, left), ...]
    face_locations = face_recognition.face_locations(frame_rgb, model=config.FACE_DETECT_MODEL)

    if not face_locations:
        return faces_imgs, faces_meta, embeddings

    # landmarks برای تمام چهره‌ها
    all_landmarks = face_recognition.face_landmarks(frame_rgb, face_locations)

    # استخراج embedding برای هر چهره
    face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

    for loc, lm, enc in zip(face_locations, all_landmarks, face_encodings):
        top, right, bottom, left = loc
        w = right - left
        h = bottom - top

        # فیلتر بر اساس اندازه
        if w < config.MIN_FACE_SIZE or h < config.MIN_FACE_SIZE:
            continue

        face_crop = frame_rgb[top:bottom, left:right]

        aligned_face, angle = align_face(face_crop, lm, config.FACE_IMAGE_SIZE)

        quality = compute_quality_score(aligned_face, (w, h), angle)

        meta = FaceMeta(
            video_path=video_path,
            frame_index=frame_index,
            bbox=(top, right, bottom, left),
            quality_score=quality
        )

        faces_imgs.append(aligned_face)
        faces_meta.append(meta)
        embeddings.append(l2_normalize(np.array(enc, dtype=np.float32)))

    return faces_imgs, faces_meta, embeddings

# ==========================
# پردازش ویدیوها و جمع‌آوری embedding ها
# ==========================

def process_videos_in_folder(folder: str, config: Config):
    """
    ویدیوها را می‌خواند و embedding ها و اطلاعات مربوطه را جمع می‌کند.
    """
    all_embeddings: List[np.ndarray] = []
    all_metas: List[FaceMeta] = []
    all_faces: List[np.ndarray] = []

    video_paths = []
    for ext in ("*.mp4", "*.avi", "*.mkv", "*.mov"):
        video_paths.extend(glob.glob(os.path.join(folder, ext)))

    print(f"[INFO] Found {len(video_paths)} videos in {folder}")

    for vid in video_paths:
        print(f"[INFO] Processing video: {vid}")
        for frame_index, frame_bgr in sample_frames(
                vid, config.FRAME_STEP_SHORT, config.FRAME_STEP_LONG):
            frame_rgb = preprocess_frame(frame_bgr)
            faces_imgs, faces_meta, embeddings = process_frame_faces(
                frame_rgb, vid, frame_index, config)

            all_faces.extend(faces_imgs)
            all_metas.extend(faces_meta)
            all_embeddings.extend(embeddings)

    if not all_embeddings:
        print("[WARN] No faces detected in folder:", folder)

    embeddings_array = np.vstack(all_embeddings) if all_embeddings else np.empty((0, 128), dtype=np.float32)
    return embeddings_array, all_metas, all_faces

# ==========================
# PCA + DBSCAN روی ویدیوهای مرجع
# ==========================

def train_pca(embeddings: np.ndarray, config: Config) -> Optional[PCA]:
    if not config.USE_PCA or embeddings.shape[0] == 0:
        return None
    print("[INFO] Training PCA...")
    pca = PCA(n_components=config.PCA_COMPONENTS, svd_solver="auto", whiten=False)
    pca.fit(embeddings)
    ensure_dir(config.OUTPUT_DIR)
    dump(pca, config.PCA_MODEL_PATH)
    print(f"[INFO] Saved PCA model to {config.PCA_MODEL_PATH}")
    return pca


def load_pca_if_exists(config: Config) -> Optional[PCA]:
    if not config.USE_PCA:
        return None
    if os.path.exists(config.PCA_MODEL_PATH):
        print(f"[INFO] Loading PCA model from {config.PCA_MODEL_PATH}")
        return load(config.PCA_MODEL_PATH)
    return None


def apply_pca(embeddings: np.ndarray, pca: Optional[PCA]) -> np.ndarray:
    if pca is None or embeddings.shape[0] == 0:
        return embeddings
    transformed = pca.transform(embeddings)
    # بعد از PCA هم می‌توانیم نرمال‌سازی کنیم
    transformed = np.array([l2_normalize(e) for e in transformed], dtype=np.float32)
    return transformed


def cluster_with_dbscan(embeddings: np.ndarray,
                        metas: List[FaceMeta],
                        config: Config) -> np.ndarray:
    """
    خوشه‌بندی با DBSCAN.
    اگر با eps اولیه هیچ خوشه‌ای پیدا نشد،
    به صورت تطبیقی eps را بزرگ‌تر می‌کنیم تا حداقل ۱ خوشه پیدا شود.
    اگر باز هم نشد، همه را یک خوشه واحد در نظر می‌گیریم.
    """
    if embeddings.shape[0] == 0:
        print("[WARN] No embeddings to cluster.")
        return np.array([])

    eps = config.DBSCAN_EPS
    best_labels = None

    for step in range(10):  # حداکثر ۱۰ بار با eps های بزرگ‌تر امتحان می‌کنیم
        db = DBSCAN(
            eps=eps,
            min_samples=config.DBSCAN_MIN_SAMPLES,
            metric="euclidean"  # چون قبلاً L2 normalize کردیم
        )
        labels = db.fit_predict(embeddings)
        n_clusters = len(set(labels) - {-1})
        n_noise = int(np.sum(labels == -1))

        print(f"[INFO] DBSCAN try {step+1}: eps={eps:.3f} -> "
              f"{n_clusters} clusters, {n_noise} noise points.")

        if n_clusters > 0:
            best_labels = labels
            break

        eps += 0.1  # هر بار کمی eps را بالا می‌بریم

    if best_labels is None:
        # هنوز هیچ خوشه‌ای پیدا نشده → همه را یک خوشه در نظر می‌گیریم
        print("[WARN] No clusters found even with large eps. "
              "Forcing single cluster (all faces = person 0).")
        best_labels = np.zeros(embeddings.shape[0], dtype=int)

    # ذخیره label داخل meta
    for meta, lbl in zip(metas, best_labels):
        meta.cluster_label = int(lbl) if lbl != -1 else None

    final_clusters = len(set(best_labels) - {-1})
    final_noise = int(np.sum(best_labels == -1))
    print(f"[INFO] Final DBSCAN result: {final_clusters} clusters, {final_noise} noise points.")

    return best_labels


# ==========================
# انتخاب نماینده هر خوشه
# ==========================

def select_representative_faces(labels: np.ndarray,
                                metas: List[FaceMeta],
                                faces: List[np.ndarray],
                                embeddings: np.ndarray,
                                config: Config) -> List[RepresentativeFace]:
    """
    برای هر خوشه‌ی معتبر:
    - صورت با بالاترین quality_score را انتخاب و ذخیره می‌کنیم.
    """
    ensure_dir(config.REPRESENTATIVE_FACES_DIR)

    reps: List[RepresentativeFace] = []

    if labels.size == 0:
        return reps

    unique_labels = sorted(list(set(labels) - {-1}))  # برچسب -1 = نویز
    print(f"[INFO] Selecting representatives for {len(unique_labels)} clusters.")

    for person_id, lbl in enumerate(unique_labels):
        idxs = [i for i, lab in enumerate(labels) if lab == lbl]
        if not idxs:
            continue

        # انتخاب ایندکس با بیشترین quality_score
        best_index = max(idxs, key=lambda i: metas[i].quality_score)
        best_face_img = faces[best_index]
        best_meta = metas[best_index]
        best_embedding = embeddings[best_index]

        filename = f"person_{person_id:03d}.jpg"
        save_path = os.path.join(config.REPRESENTATIVE_FACES_DIR, filename)
        # تبدیل RGB به BGR برای ذخیره با OpenCV
        cv2.imwrite(save_path, cv2.cvtColor(best_face_img, cv2.COLOR_RGB2BGR))

        rep = RepresentativeFace(
            person_id=person_id,
            image_path=save_path,
            embedding_index=best_index,
            meta=best_meta
        )
        reps.append(rep)

    # ذخیره اطلاعات متنی
    rep_dicts = [asdict(r) for r in reps]
    with open(config.REPRESENTATIVES_PATH, "w", encoding="utf-8") as f:
        json.dump(rep_dicts, f, ensure_ascii=False, indent=2)

    # ذخیره embedding نمایندگان (ترتیب همان person_id)
    rep_embeddings = np.vstack([
        embeddings[r.embedding_index] for r in reps
    ]) if reps else np.empty((0, embeddings.shape[1]), dtype=np.float32)

    np.save(config.REPRESENTATIVE_EMBEDDINGS_PATH, rep_embeddings)

    print(f"[INFO] Saved {len(reps)} representative faces to {config.REPRESENTATIVE_FACES_DIR}")
    print(f"[INFO] Saved representative embeddings to {config.REPRESENTATIVE_EMBEDDINGS_PATH}")
    print(f"[INFO] Saved meta info to {config.REPRESENTATIVES_PATH}")

    return reps

# ==========================
# مقایسه ویدیوهای دیگر با افراد مرجع
# ==========================

def load_representatives(config: Config) -> Tuple[np.ndarray, List[RepresentativeFace]]:
    if not os.path.exists(config.REPRESENTATIVE_EMBEDDINGS_PATH):
        print("[WARN] Representative embeddings file not found.")
        return np.empty((0, 0), dtype=np.float32), []

    rep_embeddings = np.load(config.REPRESENTATIVE_EMBEDDINGS_PATH)

    reps: List[RepresentativeFace] = []
    if os.path.exists(config.REPRESENTATIVES_PATH):
        with open(config.REPRESENTATIVES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for d in data:
            meta_dict = d["meta"]
            meta = FaceMeta(
                video_path=meta_dict["video_path"],
                frame_index=meta_dict["frame_index"],
                bbox=tuple(meta_dict["bbox"]),
                quality_score=meta_dict["quality_score"],
                cluster_label=meta_dict.get("cluster_label")
            )
            rep = RepresentativeFace(
                person_id=d["person_id"],
                image_path=d["image_path"],
                embedding_index=d["embedding_index"],
                meta=meta
            )
            reps.append(rep)

    return rep_embeddings, reps


def match_query_videos(config: Config, pca: Optional[PCA] = None):
    """
    ویدیوهای QUERY_VIDEOS_DIR را پردازش می‌کند و برای هر چهره، نزدیک‌ترین فرد مرجع را پیدا می‌کند.
    اگر فاصله < MATCH_THRESHOLD بود، آن را حضور همان شخص در آن فریم حساب می‌کند.
    """
    rep_embeddings, reps = load_representatives(config)
    if rep_embeddings.size == 0:
        print("[WARN] No representative embeddings found. Skipping query videos matching.")
        return

    print(f"[INFO] Loaded {rep_embeddings.shape[0]} representative persons.")

    # اگر PCA استفاده شده، لازم است روی نماینده‌ها هم اعمال شده باشد (در همان مرحله آموزش انجام شده است).
    # اینجا فقط مطمئن می‌شویم که اگر PCA هست، روی embedding های جدید هم اعمال می‌شود.
    # (فرض می‌کنیم rep_embeddings از قبل در فضای PCA هستند.)

    results = []  # برای گزارش نهایی

    video_paths = []
    for ext in ("*.mp4", "*.avi", "*.mkv", "*.mov"):
        video_paths.extend(glob.glob(os.path.join(config.QUERY_VIDEOS_DIR, ext)))

    print(f"[INFO] Matching on {len(video_paths)} query videos.")

    for vid in video_paths:
        print(f"[INFO] Processing query video: {vid}")
        for frame_index, frame_bgr in sample_frames(
                vid, config.FRAME_STEP_SHORT, config.FRAME_STEP_LONG):
            frame_rgb = preprocess_frame(frame_bgr)
            faces_imgs, faces_meta, embeddings = process_frame_faces(
                frame_rgb, vid, frame_index, config)

            if not embeddings:
                continue

            emb_array = np.vstack(embeddings).astype(np.float32)

            # اگر PCA داریم، روی embedding های جدید هم اعمال می‌کنیم
            emb_array = apply_pca(emb_array, pca)

            for i, emb in enumerate(emb_array):
                # فاصله‌ی اقلیدسی با همه‌ی نماینده‌ها
                dists = np.linalg.norm(rep_embeddings - emb, axis=1)
                min_idx = int(np.argmin(dists))
                min_dist = float(dists[min_idx])

                if min_dist < config.MATCH_THRESHOLD:
                    person_id = reps[min_idx].person_id
                    result = {
                        "video": vid,
                        "frame_index": faces_meta[i].frame_index,
                        "person_id": person_id,
                        "distance": min_dist
                    }
                    results.append(result)
                    print(f"[MATCH] video={os.path.basename(vid)}, "
                          f"frame={faces_meta[i].frame_index}, "
                          f"person={person_id}, dist={min_dist:.3f}")

    # ذخیره نتایج
    results_path = os.path.join(config.OUTPUT_DIR, "matches.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved query matches to {results_path}")

# ==========================
# main
# ==========================

def main():
    cfg = Config()
    ensure_dir(cfg.OUTPUT_DIR)

    # 1) پردازش ویدیوهای مرجع
    ref_embeddings, ref_metas, ref_faces = process_videos_in_folder(
        cfg.REFERENCE_VIDEOS_DIR, cfg)

    if ref_embeddings.shape[0] == 0:
        print("[ERROR] No embeddings found in reference videos. Exiting.")
        return

    # 2) PCA
    pca = load_pca_if_exists(cfg)
    if pca is None and cfg.USE_PCA:
        pca = train_pca(ref_embeddings, cfg)

    ref_embeddings_transformed = apply_pca(ref_embeddings, pca)

    # 3) DBSCAN روی ویدیوهای مرجع
    labels = cluster_with_dbscan(ref_embeddings_transformed, ref_metas, cfg)

    # 4) انتخاب نماینده‌ی هر خوشه و ذخیره عکس + embedding
    select_representative_faces(labels, ref_metas, ref_faces, ref_embeddings_transformed, cfg)

    # 5) پردازش ویدیوهای دیگر و پیدا کردن حضور افراد مرجع
    match_query_videos(cfg, pca=pca)


if __name__ == "__main__":
    main()
