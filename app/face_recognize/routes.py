import json
import os

from flask import render_template, request
from app.auth.models import Counter
from app.face_recognize import recognition
from source.create_classifier_model import add_new_class
from source.face_recognition import FaceRecognition
from source.utils import draw_rectangles, read_image, prepare_image
from flask import current_app
import cv2

recognizer = FaceRecognition()


@recognition.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    filename = file.filename
    image_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)

    # Read image
    image = read_image(file)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(image_path, image)

    # Recognize faces
    faces = recognizer.identify_face(image_path)

    # Draw detection rects
    draw_rectangles(image, faces)

    # Prepare image for html
    to_send = prepare_image(image)
    Counter.update()
    counter_value = Counter.get()

    with open(f"{current_app.config['ROOT_PATH']}/classes.json", encoding='utf-8') as file:
        classes = json.load(file)

    return render_template('index.html', face_recognized=len(faces) > 0, num_faces=len(faces), image_to_show=to_send,
                           init=True, counter_value=counter_value, classes=classes)


@recognition.route('/add-image', methods=['POST'])
def add_image():

    selected_class = ""
    if request.form["option1"] == 'true':
        selected_class = request.form["text1"]
    elif request.form["option2"] == 'true':
        selected_class = request.form["text2"]

    file = request.files['image']
    filename = file.filename

    # Read image
    image = read_image(file)
    output_message = f"Клас <strong>{selected_class}</strong> успішно додано!"
    error = False
    try:
        add_new_class(selected_class, image)
    except:
        output_message = "Виникла помилка при додаванні класу!"
        error = True

    Counter.update()
    counter_value = Counter.get()

    with open(f"{current_app.config['ROOT_PATH']}/classes.json", encoding='utf-8') as file:
        classes = json.load(file)

    classes.append(selected_class)
    with open(f"{current_app.config['ROOT_PATH']}/classes.json", 'w', encoding='utf-8') as file:
        json.dump(classes, file, ensure_ascii=False, indent=2)

    return render_template('index.html', output_message=output_message, error=error, init_add=True,
                           counter_value=counter_value, classes=classes)



