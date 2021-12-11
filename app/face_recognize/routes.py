import os

from flask import render_template, request, flash, redirect, url_for
from flask.json import jsonify

from app.face_recognize import recognition
from source.face_detection import detect_faces_with_ssd
from source.face_recognition import FaceRecognition
from source.utils import draw_rectangles, read_image, prepare_image
from source.model_training import create_mlp_model
from config import DETECTION_THRESHOLD
from flask import current_app

recognizer = FaceRecognition()


@recognition.route('/recognize', methods=['POST'])
def recognize():
    file = request.files['image']

    # Read image
    image = read_image(file)

    # Recognize faces
    classifier_model_path = "models" + os.sep + "lotr_mlp_10c_recognizer.pickle"
    label_encoder_path = "models" + os.sep + "lotr_mlp_10c_labelencoder.pickle"
    faces = recognize_faces(image, classifier_model_path, label_encoder_path,
                            detection_api_url=current_app.config["DETECTION_API_URL"])

    return jsonify(recognitions=faces)


@recognition.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']

    # Read image
    image = read_image(file)

    # Detect faces
    faces = detect_faces_with_ssd(image, min_confidence=DETECTION_THRESHOLD)

    return jsonify(detections=faces)


@recognition.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    filename = file.filename
    image_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)

    # Read image
    image = read_image(file)
    file.save(image_path)

    # Recognize faces
    # faces = recognize_faces(image, classifier_model_path, label_encoder_path,
    #                         detection_api_url=current_app.config["DETECTION_API_URL"])

    faces = recognizer.identify_face(image_path)

    # Draw detection rects
    draw_rectangles(image, faces)

    # Prepare image for html
    to_send = prepare_image(image)

    return render_template('index.html', face_recognized=len(faces) > 0, num_faces=len(faces), image_to_show=to_send,
                           init=True)



@recognition.route('/add-image', methods=['POST'])
def add_image():

    pass

