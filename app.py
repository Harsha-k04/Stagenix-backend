# app.py
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context, abort
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import requests   # <-- REQUIRED
import os
import random
from flask_cors import CORS
import uuid
import time
import threading

# ================================================================
# 🔶 ORIGINAL BACKEND CONFIG (unchanged)
# ================================================================
app = Flask(
    __name__,
    static_folder="public",
    static_url_path=""
)

MODEL_URL = "https://github.com/Harsha-k04/Stagenix-backend/releases/download/v1.0/perfect_stage_corrected.glb?raw=1"

CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["Content-Type", "Authorization"],
    supports_credentials=True
)

model = YOLO("yolov8l-seg.pt")

os.makedirs("uploads", exist_ok=True)

# ================================================================
# 🔍 Helper: Simple keyword-based scene generator (unchanged)
# ================================================================
def generate_objects_from_prompt(prompt: str):
    prompt = prompt.lower()

    object_library = {
        "plant": "pottedplant",
        "tree": "pottedplant",
        "vase": "vase",
        "chair": "chair",
        "table": "table",
        "lamp": "lamp",
        "sofa": "sofa",
        "carpet": "carpet",
        "stage": "stage",
        "wedding": "wedding",
    }

    objects = []
    for keyword, obj_name in object_library.items():
        if keyword in prompt:
            objects.append({
                "name": obj_name,
                "position": [random.uniform(-1, 1), 0, random.uniform(-1, 1)],
                "rotation": [0, 0, 0]
            })

    if not objects:
        objects.append({
            "name": "cube",
            "position": [0, 0, 0],
            "rotation": [0, 0, 0]
        })

    return objects


# ================================================================
# 🔶 ORIGINAL ROUTES (unchanged)
# ================================================================
@app.route("/model/perfect_stage_corrected.glb")
def serve_model():
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "*/*"
    }

    try:
        r = requests.get(MODEL_URL, headers=headers, stream=True, allow_redirects=True)

        if r.status_code != 200:
            return {"error": f"GitHub returned {r.status_code}"}, 500

        response = Response(
            stream_with_context(r.iter_content(chunk_size=8192)),
            content_type="model/gltf-binary"
        )
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response

    except Exception as e:
        return {"error": str(e)}, 500


@app.route("/ping")
def ping():
    return {"status": "ok", "message": "backend alive"}, 200


@app.route("/predict", methods=["POST"])
def predict():
    response = {}

    if "image" in request.files:
        img = request.files["image"]
        img_path = os.path.join("uploads", secure_filename(img.filename))
        img.save(img_path)

        results = model(img_path)
        detections = results[0].tojson()

        response = {
            "status": "ok",
            "source": "image",
            "results": detections,
            "segmented_image": f"/uploads/{secure_filename(img.filename)}"
        }

    elif "prompt" in request.form:
        prompt = request.form["prompt"]
        objects = generate_objects_from_prompt(prompt)

        response = {
            "status": "ok",
            "source": "prompt",
            "prompt": prompt,
            "objects": objects
        }

    else:
        return jsonify({"error": "No image or prompt provided"}), 400

    print("Response:", response)
    return jsonify(response)


@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory("uploads", filename)


@app.route("/assets/<path:filename>")
def assets(filename):
    return send_from_directory("public/assets", filename)


@app.route("/")
def home():
    return app.send_static_file("index.html")


# ================================================================
# 🔶 MERGED: KAGGLE WORKER QUEUE SYSTEM
# ================================================================

UPLOAD_DIR = "generated_models"
os.makedirs(UPLOAD_DIR, exist_ok=True)

JOBS = {}
JOBS_LOCK = threading.Lock()

def new_job(prompt, meta=None):
    job_id = uuid.uuid4().hex
    job = {
        "id": job_id,
        "prompt": prompt,
        "meta": meta or {},
        "status": "queued",
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "result": None,
        "worker_id": None,
        "error": None,
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
    return job


def next_queued_job_and_claim(worker_id):
    with JOBS_LOCK:
        for job in JOBS.values():
            if job["status"] == "queued":
                job["status"] = "running"
                job["started_at"] = time.time()
                job["worker_id"] = worker_id
                return job
    return None


def update_job_done(job_id, result_filename):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return None
        job["status"] = "done"
        job["finished_at"] = time.time()
        job["result"] = {"file": result_filename, "url": f"/result/{job_id}"}
        return job


def update_job_failed(job_id, error_msg):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return None
        job["status"] = "failed"
        job["finished_at"] = time.time()
        job["error"] = error_msg
        return job


# ---- API: create job ----
@app.route("/generate", methods=["POST"])
def generate():
    body = request.get_json(force=True)
    prompt = body.get("prompt")
    meta = body.get("meta", {})

    if not prompt:
        return jsonify({"error": "missing prompt"}), 400

    job = new_job(prompt, meta)
    return jsonify({"job_id": job["id"]}), 201


# ---- API: job status ----
@app.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    return jsonify({k: v for k, v in job.items() if k != "meta"}), 200


# ---- API: get result GLB ----
@app.route("/result/<job_id>", methods=["GET"])
def get_result(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    if job["status"] != "done":
        return jsonify({"error": "result not ready"}), 404

    if job["result"]["file"]:
        return send_from_directory(UPLOAD_DIR, job["result"]["file"], as_attachment=False)
    else:
        return jsonify({"url": job["result"]["url"]})


# ---- Worker polling endpoint (Kaggle pulls jobs) ----
@app.route("/job/next", methods=["POST"])
def job_next():
    body = request.get_json(force=True) if request.data else {}
    worker_id = body.get("worker_id") or request.remote_addr

    job = next_queued_job_and_claim(worker_id)
    if not job:
        return "", 204
    return jsonify(job), 200


# ---- Worker reports completion ----
@app.route("/job/<job_id>/complete", methods=["POST"])
def job_complete(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404

    if "model" in request.files:
        f = request.files["model"]
        filename = secure_filename(f.filename)
        out_fname = f"{job_id}__{filename}"
        out_path = os.path.join(UPLOAD_DIR, out_fname)
        f.save(out_path)
        update_job_done(job_id, out_fname)
        return jsonify({"status": "ok", "file": f"/result/{job_id}"}), 200

    body = request.get_json(force=True) if request.data else {}
    model_url = body.get("model_url")

    if model_url:
        with JOBS_LOCK:
            job["status"] = "done"
            job["finished_at"] = time.time()
            job["result"] = {"file": None, "url": model_url}
        return jsonify({"status": "ok", "url": model_url}), 200

    return jsonify({"error": "no model provided"}), 400


# ---- list jobs (debug only) ----
@app.route("/_jobs", methods=["GET"])
def list_jobs():
    with JOBS_LOCK:
        return jsonify(list(JOBS.values())), 200

# app.py

# ... imports and setup ...

@app.route("/api/upload-sketch", methods=["POST"])
def upload_sketch():
    if "sketch" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["sketch"]
    filename = secure_filename(file.filename)
    save_path = os.path.join("uploads", filename)
    file.save(save_path)

    # --- 🎯 FIX: Construct the public URL ---
    # `request.host_url` gives the base URL (e.g., "https://stagenix-backend.onrender.com/")
    # The file is served via the /uploads route.
    public_url = f"{request.host_url.rstrip('/')}/uploads/{filename}"
    
    print("Sketch uploaded and public URL generated:", public_url)

    return jsonify({
        "status": "ok",
        "sketch_url": public_url # <--- CHANGED KEY NAME for clarity/consistency
    })
# ================================================================
# 🎯 RUN APP (unchanged)
# ================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
