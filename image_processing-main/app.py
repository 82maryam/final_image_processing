import os, uuid, shutil
from flask import Flask, render_template, request, jsonify,url_for
from fast_face_video_core import run_pipeline

ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD = os.path.join(ROOT, "uploads")
RESULTS = os.path.join(ROOT, "static", "results")

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

app = Flask(__name__)

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/api/process")
def process():
    ref_file = request.files["reference"]
    video_files = request.files.getlist("videos")
    job_id = str(uuid.uuid4())[:8]
    job_upload_dir = os.path.join(UPLOAD, job_id)
    job_result_dir = os.path.join(RESULTS, job_id)
    os.makedirs(job_upload_dir, exist_ok=True)
    os.makedirs(job_result_dir, exist_ok=True)

    ref_path = os.path.join(job_upload_dir, "reference.mp4")
    ref_file.save(ref_path)

    saved_videos = []
    for vf in video_files:
        name = os.path.basename(vf.filename)
        outp = os.path.join(job_upload_dir, name)
        vf.save(outp)
        saved_videos.append(outp)
    result = run_pipeline(
        reference_video_path=ref_path,
        other_video_paths=saved_videos,
        out_root=job_result_dir
    )

    def to_url(path_abs):
        rel = os.path.relpath(path_abs, os.path.join(ROOT, "static")).replace("\\", "/")
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

    job_id = request.json.get("job_id")
    shutil.rmtree(os.path.join(UPLOAD, job_id), ignore_errors=True)
    shutil.rmtree(os.path.join(RESULTS, job_id), ignore_errors=True)
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
