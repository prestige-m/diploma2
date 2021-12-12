import os
import imutils
import numpy as np
from imutils import paths
from matplotlib import pyplot as plt

from source.face_recognition import FaceRecognition


def extract_embeddings(classes_dir: str = "", model_path: str = "", default_classname: str=None,
                       one_face_limit=True, save_dataset=True):
    image_paths = list(paths.list_images(classes_dir))

    recognition = FaceRecognition()

    X = []
    y = []
    for k, image_path in enumerate(image_paths):
        print(f"[INFO] processing image {image_path} --- {k + 1}/{len(image_paths)}")

        label = image_path.split(os.path.sep)[-2] if not default_classname else default_classname
        image_array = recognition.load_image(image_path)
        if image_array is not None:
            height, width = image_array.shape[:2]
            if width > 600:
                image_array = imutils.resize(image_array, width=600)
            faces = recognition.detect_faces(image_array)

            if faces:
                if one_face_limit and len(faces) > 1:
                    print(f'ERROR - {image_path} - faces - {len(faces)}')
                    continue
                    # plt.imshow(face_array)
                    # plt.title(image_path.split(os.path.sep)[-1])
                    # plt.show()

                face_array = faces[0]['array']
                face_keypoints = faces[0]['keypoints']
                face_array = recognition.align_face(image_array, face_keypoints)

                face_array = recognition.preprocess_face(face_array, target_size=(160, 160))
                embedding = recognition.get_embeddings(face_array)
                X.append(embedding)
                y.append(label)
            else:
                print(f'[WARNING] Face is not found: {image_path}')

    if save_dataset:
        np.savez_compressed(f"{model_path}/embeddings-dataset.npz", np.asarray(X), np.asarray(y))

    return np.asarray(X), np.asarray(y)


if __name__ == '__main__':
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_path, "dataset")
    model_path = os.path.join(base_path, "models")

    extract_embeddings(dataset_path, model_path)
