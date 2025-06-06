from flask import Flask, render_template, request, redirect, url_for

import os
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

def detect_face_shape(landmarks, image_width, image_height):
    def to_pixel(point):
        return int(point.x * image_width), int(point.y * image_height)

    def dist(p1, p2):
        x1, y1 = to_pixel(p1)
        x2, y2 = to_pixel(p2)
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Get necessary landmarks
    forehead_top = landmarks[10]
    chin = landmarks[152]
    jaw_left = landmarks[234]
    jaw_right = landmarks[454]
    cheek_left = landmarks[93]
    cheek_right = landmarks[323]
    temple_left = landmarks[338]
    temple_right = landmarks[127]

    # Measure distances
    face_length = dist(forehead_top, chin)
    forehead_width = dist(temple_left, temple_right)
    cheekbone_width = dist(cheek_left, cheek_right)
    jaw_width = dist(jaw_left, jaw_right)

    # Ratios
    width_avg = (forehead_width + cheekbone_width + jaw_width) / 3
    length_ratio = face_length / width_avg

    # Classification logic
    if length_ratio < 1.1:
        if abs(jaw_width - cheekbone_width) < 15:
            return "Round"
        else:
            return "Square"
    elif 1.1 <= length_ratio < 1.4:
        if forehead_width > jaw_width:
            return "Heart"
        else:
            return "Oval"
    elif length_ratio >= 1.4:
        return "Oblong"
    else:
        return "Unknown"


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['image']
    if not file:
        return "No file uploaded", 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'input.jpg')
    file.save(filepath)

    # Load image for face shape analysis
    image = cv2.imread(filepath)
    if image is None:
        return "Invalid image", 400
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    face_shape = "Unknown"
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = image.shape
        face_shape = detect_face_shape(landmarks, w, h)

    return redirect(url_for('result', shape=face_shape))

@app.route('/result')
def result():
    face_shape = request.args.get('shape', 'Unknown')
    return render_template('result.html', face_shape=face_shape)

if __name__ == '__main__':
    app.run(debug=True)

