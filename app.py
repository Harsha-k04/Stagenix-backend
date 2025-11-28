from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import os
import random
from flask_cors import CORS

# --- Flask app configuration ---
app = Flask(
    __name__,
    static_folder="public",
    static_url_path=""
)
GITHUB_MODEL_URL = "https://github.com/Harsha-k04/Stagenix-backend/releases/download/v1.0/perfect_stage_corrected.glb"
# --- Allow CORS for Next.js frontend (UPDATED) ---
CORS(
    app,
    resources={r"/*": {"origins": "*"}},   # allow all origins for Render/Vercel
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["Content-Type", "Authorization"],
    supports_credentials=True
)


# --- Load YOLO model ---
model = YOLO("yolov8l-seg.pt")

# --- Ensure upload folder exists ---
os.makedirs("uploads", exist_ok=True)


# ----------------------------------------
# 🔍 Helper: Simple keyword-based parser
# ----------------------------------------
def generate_objects_from_prompt(prompt: str):
    """
    Simulate 3D object layout based on keywords in the prompt.
    You can expand this as you add new 3D assets.
    """
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
@app.route("/model/<path:filename>")
def proxy_model(filename):
    r = requests.get(GITHUB_MODEL_URL, stream=True)

    def generate():
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                yield chunk

    return Response(
        generate(),
        content_type="model/gltf-binary",
        headers={"Access-Control-Allow-Origin": "*"}
    )
@app.route("/ping")
def ping():
    return {"status": "ok", "message": "backend alive"}, 200

# ----------------------------------------
# 🧠 Prediction Route
# ----------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles:
    - Image uploads → YOLO segmentation
    - Prompt text → keyword-based layout generation
    """
    response = {}

    # --- Image input ---
    if "image" in request.files:
        img = request.files["image"]
        img_path = os.path.join("uploads", secure_filename(img.filename))
        img.save(img_path)

        # Run YOLO detection
        results = model(img_path)
        detections = results[0].tojson()

        response = {
            "status": "ok",
            "source": "image",
            "results": detections,
            "segmented_image": f"/uploads/{secure_filename(img.filename)}"
        }

    # --- Text prompt input ---
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


# ----------------------------------------
# 🗂 Serve static files
# ----------------------------------------
@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory("uploads", filename)


@app.route("/assets/<path:filename>")
def assets(filename):
    return send_from_directory("public/assets", filename)


@app.route("/")
def home():
    return app.send_static_file("index.html")


# ----------------------------------------
# ▶ Run Flask locally AND Render
# ----------------------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
