import datetime
import pickle

import tensorflow as tf
from PIL import Image
from matplotlib import pyplot
from mtcnn import MTCNN
from numpy import asarray, savez_compressed, load
from sklearn.metrics import classification_report
from tensorflow import keras

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras import layers

from tensorflow.python.keras.applications.inception_v3 import InceptionV3

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



def extract_face(filename, required_size=(224, 224)):
    # load image from file
    pixels = pyplot.imread(filename)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    if len(results) > 1:
        return None

    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

def load_images(folder_path: str):
    imagePaths = list(paths.list_images(folder_path))
    data = []
    labels = []
    # loop over the image paths
    for k, imagePath in enumerate(imagePaths):
        print(f"image {k + 1} / {len(imagePaths)}")
        try:
            # extract the class label from the filename
            label = imagePath.split(os.path.sep)[-2]
            # load the input image (224x224) and preprocess it
            #image2 = load_img(imagePath, target_size=(224, 224))
            #image = img_to_array(image2)

            image = extract_face(imagePath)
            if image is None:
                print("Error many faces!")
                continue

            image = preprocess_input(image)
            # update the data and labels lists, respectively
            data.append(image)
            labels.append(label)
        except Exception as e:
            print(e)
    # convert the data and labels to NumPy arrays
    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    return data, labels


def model_new():
    shape = (224, 224, 3)
    inputs = Input(shape)
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(512, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    pooledOutput = GlobalAveragePooling2D()(x)
    pooledOutput = Dense(1024, activation='relu')(pooledOutput)
    outputs = Dense(128)(pooledOutput)

    model = Model(inputs, outputs)
    return model


def get_model():
    baseModel = InceptionV3(weights="imagenet", include_top=False,
                            input_shape=(224, 224, 3))
    # construct the head of the model that will be placed on top of the
    # the base model
    headModel = baseModel.output


    # headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    # headModel = Flatten(name="flatten")(headModel)
    headModel = layers.GlobalAveragePooling2D()(headModel)

    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(7, activation="softmax")(headModel)

    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)
    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in baseModel.layers:
        layer.trainable = False

    return model


if __name__ == '__main__':

    # data, labels = load_images()
    # savez_compressed('test-dataset2.npz', data, labels)

    data = load('test-dataset2.npz')
    data, labels = data['arr_0'], data['arr_1']

    fasfasf=2345
    afasf=2623


    #
    lb = LabelBinarizer()
    # lb.fit(labels)
    # train_y = lb.transform(labels)
    labels = lb.fit_transform(labels)
    #labels3 = to_categorical(labels)
    # # partition the data into training and testing splits using 80% of
    # # the data for training and the remaining 20% for testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                      test_size=0.25, stratify=labels, random_state=42)

    # model = get_model()
    #
    # # construct the training image generator for data augmentation
    # aug = ImageDataGenerator(
    #     rotation_range=20,
    #     zoom_range=0.15,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     shear_range=0.15,
    #     horizontal_flip=True,
    #     fill_mode="nearest")
    #
    # INIT_LR = 1e-4
    # EPOCHS = 150
    # BS = 32
    # print("[INFO] compiling model...")
    # opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    # # model.compile(loss="binary_crossentropy", optimizer=opt,
    # #               metrics=["accuracy"])
    #
    # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    #
    # model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    # # train the head of the network
    # print("[INFO] training head...")
    # # H = model.fit(
    # #     aug.flow(trainX, trainY, batch_size=BS),
    # #     steps_per_epoch=len(trainX) // BS,
    # #     validation_data=(testX, testY),
    # #     validation_steps=len(testX) // BS,
    # #     epochs=EPOCHS,
    # #     callbacks=[tensorboard_callback])
    #
    # H = model.fit(x=trainX,
    #           y=trainY,
    #           batch_size=BS,
    #           epochs=EPOCHS,
    #           validation_data=(testX, testY),
    #           callbacks=[tensorboard_callback])
    #
    #
    # N = EPOCHS
    # plt.style.use("ggplot")
    # plt.figure()
    # plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    # plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    # plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    # plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    # plt.title("Training Loss and Accuracy")
    # plt.xlabel("Epoch #")
    # plt.ylabel("Loss/Accuracy")
    # plt.legend(loc="lower left")
    #
    # # To save the trained model
    # model.save('prod_model5.h5')


    model = keras.models.load_model('prod_model5.h5')


    y_pred = model.predict(testX)
    y_trn = model.predict(trainX)

    y_pred = np.argmax(y_pred, axis=1)
    testY = np.argmax(testY, axis=1)
    y_trn = np.argmax(y_trn, axis=1)
    trainY = np.argmax(trainY, axis=1)

    from sklearn import metrics
    score_train = metrics.accuracy_score(trainY, y_trn)
    score_test = metrics.accuracy_score(testY, y_pred)

    afsfa=2352
    print(classification_report(testY, y_pred, target_names=lb.classes_))

    # print(classification_report(testY, predIdxs,
    #                             target_names=lb.classes_))