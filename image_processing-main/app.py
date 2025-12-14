import os, uuid, shutil
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for

# این تابع را از core صدا می‌زنیم
from fast_face_video_core import run_pipeline  # پایین‌تر فایل core را می‌دهم

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_ROOT = os.path.join(APP_ROOT, "uploads")
RESULTS_ROOT = os.path.join(APP_ROOT, "static", "results")

os.makedirs(UPLOAD_ROOT, exist_ok=True)
os.makedirs(RESULTS_ROOT, exist_ok=True)

app = Flask(__name__)

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/api/process")
def process():
    """
    ورودی:
      - reference: فایل مرجع (mp4)
      - videos[]: چند فایل ویدئویی (mp4) از داخل پوشه
    خروجی:
      - لیست چهره‌های مرجع (best faces)
      - لیست تصاویر match (کنار هم)
    """
    if "reference" not in request.files:
        return jsonify({"ok": False, "error": "reference video is missing"}), 400

    ref_file = request.files["reference"]
    video_files = request.files.getlist("videos")

    if not video_files:
        return jsonify({"ok": False, "error": "no videos selected"}), 400

    job_id = str(uuid.uuid4())[:8]
    job_upload_dir = os.path.join(UPLOAD_ROOT, job_id)
    job_result_dir = os.path.join(RESULTS_ROOT, job_id)
    os.makedirs(job_upload_dir, exist_ok=True)
    os.makedirs(job_result_dir, exist_ok=True)

    # ذخیره فایل‌ها
    ref_path = os.path.join(job_upload_dir, "reference.mp4")
    ref_file.save(ref_path)

    saved_videos = []
    for vf in video_files:
        # بعضی مرورگرها مسیر نسبی میدن، فقط basename رو نگه می‌داریم
        name = os.path.basename(vf.filename)
        if not name.lower().endswith(".mp4"):
            continue
        outp = os.path.join(job_upload_dir, name)
        vf.save(outp)
        saved_videos.append(outp)

    if not saved_videos:
        return jsonify({"ok": False, "error": "no mp4 videos found in folder"}), 400

    # اجرای پایپ‌لاین
    # خروجی‌ها را داخل static/results/<job_id>/ می‌ریزیم تا از وب قابل نمایش باشد
    result = run_pipeline(
        reference_video_path=ref_path,
        other_video_paths=saved_videos,
        out_root=job_result_dir
    )

    # لینک‌های وب‌قابل نمایش بسازیم
    def to_url(path_abs):
        # path_abs داخل static/results/... است
        rel = os.path.relpath(path_abs, os.path.join(APP_ROOT, "static")).replace("\\", "/")
        return url_for("static", filename=rel)

    ref_faces = []
    for p in result["reference_people"]:
        ref_faces.append({
            "person_id": p["person_id"],
            "image_url": to_url(p["best_image_path"])
        })

    matches = []
    for m in result["matches"]:
        matches.append({
            "ref_person": m["ref_person"],
            "video_name": m["video_name"],
            "similarity": m["similarity"],
            "pair_image_url": to_url(m["pair_image_path"])
        })

    return jsonify({
        "ok": True,
        "job_id": job_id,
        "reference_faces": ref_faces,
        "matches": matches
    })

@app.post("/api/cleanup")
def cleanup():
    """
    اختیاری: پاک کردن نتایج قدیمی
    """
    job_id = request.json.get("job_id")
    if not job_id:
        return jsonify({"ok": False, "error": "job_id missing"}), 400

    shutil.rmtree(os.path.join(UPLOAD_ROOT, job_id), ignore_errors=True)
    shutil.rmtree(os.path.join(RESULTS_ROOT, job_id), ignore_errors=True)
    return jsonify({"ok": True})

if __name__ == "__main__":
    # روی لپ‌تاپ لوکال
    app.run(host="127.0.0.1", port=5000, debug=True)
