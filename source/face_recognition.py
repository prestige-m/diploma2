import os
import cv2
import pickle
import imutils
import numpy as np

from deepface.basemodels.Facenet import InceptionResNetV2
from collections import Counter
from mtcnn import MTCNN
from numpy import ndarray
from .utils import calc_threshold
from .create_classifier_model import label_encoder_path, classifier_model_path


BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_path = os.path.join(BASE_PATH, "models")
facenet_model_path = os.path.join(models_path, "facenet_weights.h5")
test_images_path = os.path.join(BASE_PATH, "test_images")


class FaceRecognition:

    def __init__(self):
        self.deepface_model = InceptionResNetV2()
        self.deepface_model.load_weights(facenet_model_path)
        self.detector = MTCNN()

        with open(classifier_model_path, 'rb') as handle:
            self.recognizer = pickle.load(handle)

        with open(label_encoder_path, 'rb') as handle:
            self.label_encoder = pickle.load(handle)

    @staticmethod
    def load_image(image_path: str):
        image = None

        try:
            image = cv2.imread(image_path)  # open the image
            image = image.astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert the image to RGB format
        except (AttributeError, cv2.error) as e:
            print(f'load image error: {e}')

        return image

    @staticmethod
    def preprocess_face(face_array: ndarray, target_size: tuple = (224, 224)):

        face_array = cv2.resize(face_array, dsize=target_size,
                                   interpolation=cv2.INTER_CUBIC)

        # scale pixel values
        face_array = face_array.astype('float32')
        face_array = face_array / 255.0  # normalize input in [0, 1]
        face_array = np.expand_dims(face_array, axis=0)

        return face_array

    def detect_faces(self, image_array: ndarray):

        faces = self.detector.detect_faces(image_array)  # get list of face boxes
        image_height, image_width = image_array.shape[:2]

        face_arrays = []
        for face in faces:
            x, y, w, h = face.get('box')

            start_x, start_y = (max(0, x), max(0, y))
            end_x, end_y = (min(image_width - 1, abs(start_x + w)), min(image_height - 1, abs(start_y + h)))
            extracted_face = image_array[start_y:end_y, start_x:end_x]

            face_arrays.append({
                'confidence': face.get('confidence'),
                'keypoints': face.get('keypoints'),
                'box': (x, y, w, h),
                'rect': (start_x, start_y, end_x, end_y),
                'rect_size': end_x - start_x + end_y - start_y,
                'array': extracted_face
            })

        return face_arrays


    def align_face(self, image_array: ndarray, face_keypoints: dict=None, face_size=(200, 250),
                    desiredLeftEye = (0.3, 0.26)):

        if face_keypoints is None:
            detector = MTCNN()
            faces = detector.detect_faces(image_array)
            face_keypoints = faces[0]['keypoints']

        face_width, face_height = face_size[0], face_size[1]
        left_eye_center = np.array(face_keypoints['left_eye'])
        right_eye_center = np.array(face_keypoints['right_eye'])

        dx = right_eye_center[0] - left_eye_center[0]
        dy = right_eye_center[1] - left_eye_center[1]
        angle = np.arctan2(dy, dx) * 180. / np.pi

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the *desired* image
        dist = np.sqrt((dx ** 2) + (dy ** 2))
        desiredDist = (desiredRightEyeX - desiredLeftEye[0])
        desiredDist *= face_width
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        rotate_x = (left_eye_center[0] + right_eye_center[0]) // 2
        rotate_y = (left_eye_center[1] + right_eye_center[1]) // 2
        rotate_center = (int(rotate_x), int(rotate_y))

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(rotate_center, angle, scale)

        # update the translation component of the matrix
        tX = face_width * 0.5
        tY = face_height * (desiredLeftEye[1] + 0.1)
        M[0, 2] += (tX - rotate_center[0])
        M[1, 2] += (tY - rotate_center[1])

        # apply the affine transformation
        (w, h) = (face_width, face_height)
        output = cv2.warpAffine(image_array, M, (w, h),
                                flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output


    def get_embeddings(self, face_pixels: ndarray):
        embeddings = self.deepface_model.predict(face_pixels)

        return embeddings[0]

    def check_threshold(self, img_embedding, threshold_value=0.80, counts_value=0.55):
        dataset = np.load(f"{models_path}/embeddings-dataset.npz")
        embeddings_x, embeddings_y = dataset['arr_0'], dataset['arr_1']

        distances = np.array([calc_threshold(img_embedding, x) for x in embeddings_x])
        indices = np.where(distances <= threshold_value)[0]
        found_classes = np.take(embeddings_y, indices)
        most_common = Counter(found_classes).most_common(1)[0]
        counts = Counter(embeddings_y)

        return most_common[-1] / counts[most_common[0]] >= counts_value

    def identify_face(self, image_path: str, probability_level: float = 0.0):

        image_array = self.load_image(image_path)
        result = []

        if image_array is not None:
            height, width = image_array.shape[:2]
            if width > 600:
                image_array = imutils.resize(image_array, width=600)
            faces = self.detect_faces(image_array)

            for face in faces:
                face_array = face['array']
                face_keypoints = face['keypoints']
                face_array = self.align_face(image_array, face_keypoints)
                face_array = self.preprocess_face(face_array, target_size=(160, 160))
                embedding = self.get_embeddings(face_array)

                person_name = "Unknown"
                probability = 100
                if self.check_threshold(embedding):
                    sample_x = np.asarray([embedding])
                    yhat_class = self.recognizer.predict(sample_x)
                    yhat_prob = self.recognizer.predict_proba(sample_x)

                    class_index = yhat_class[0]
                    probability = np.max(yhat_prob)
                    predict_names = self.label_encoder.inverse_transform(yhat_class)
                    person_name = predict_names[0] if probability >= probability_level else "Unknown"

                result.append({
                    'name': person_name,
                    'rect': list(face['rect']),
                    'detection_prob': face['confidence'],
                    'recognition_prob': probability * 100
                })

        return result


if __name__ == '__main__':
    recognizer = FaceRecognition()

    image_path = os.path.join(test_images_path, "tom_hiddleston.jpg")
    data = recognizer.identify_face(image_path)

    afasfsa = 32523
    afafasd = 326523