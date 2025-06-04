from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
import mediapipe as mp

app = Flask(__name__)

mp_face_mesh = mp.solutions.face_mesh

HAIRSTYLES = {
    "oval": {
        "male": [
            {"name": "Classic Short", "img": "/static/male1.jpg"},
            {"name": "Textured Crop", "img": "/static/male2.jpg"},
        ],
        "female": [
            {"name": "Layered Bob", "img": "/static/female1.jpg"},
            {"name": "Wavy Lob", "img": "/static/female2.jpg"},
        ],
    },
    "round": {
        "male": [
            {"name": "Pompadour", "img": "/static/male3.jpg"},
            {"name": "Side Part", "img": "/static/male4.jpg"},
        ],
        "female": [
            {"name": "Asymmetrical Bob", "img": "/static/female3.jpg"},
            {"name": "High Ponytail", "img": "/static/female4.jpg"},
        ],
    },
    "square": {
        "male": [
            {"name": "Buzz Cut", "img": "/static/male1.jpg"},
            {"name": "Crew Cut", "img": "/static/male2.jpg"},
        ],
        "female": [
            {"name": "Straight Bob", "img": "/static/female1.jpg"},
            {"name": "Side Swept Bangs", "img": "/static/female2.jpg"},
        ],
    },
}

def classify_face_shape(landmarks):
    left = np.array(landmarks[234])
    right = np.array(landmarks[454])
    jaw = np.array(landmarks[152])
    forehead = np.array(landmarks[10])

    width = np.linalg.norm(left - right)
    height = np.linalg.norm(jaw - forehead)
    ratio = height / width

    if ratio > 1.4:
        return "oval"
    elif ratio > 1.2:
        return "round"
    else:
        return "square"

def detect_accessories(image_rgb):
    # Placeholder for accessories detection - always false for now
    return False

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    img_data = data.get("image")
    if not img_data:
        return jsonify({"error": "No image provided"}), 400

    header, encoded = img_data.split(",", 1)
    decoded = base64.b64decode(encoded)
    nparr = np.frombuffer(decoded, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            return jsonify({"error": "No face detected, please try again"}), 400

        landmarks = []
        for lm in results.multi_face_landmarks[0].landmark:
            landmarks.append((lm.x, lm.y))

        if detect_accessories(img_rgb):
            return jsonify({"accessoryDetected": True, "message": "Please remove glasses, mask, or headwear and try again."})

        face_shape = classify_face_shape(landmarks)
        hairstyles = HAIRSTYLES.get(face_shape, {"male": [], "female": []})

        return jsonify({"face_shape": face_shape, "hairstyles": hairstyles, "accessoryDetected": False})

if __name__ == "__main__":
    app.run(debug=True)
