import os
import pickle
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
from sklearn import metrics


BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_PATH, "models")

classifier_model_path = os.path.join(model_path, "classifier_model.pickle")
label_encoder_path = os.path.join(model_path, "label_encoder.pickle")


def create_model():

    dataset = np.load(f"{model_path}/embeddings-dataset.npz")
    embeddings_x, embeddings_y = dataset['arr_0'], dataset['arr_1']

    train_x, test_x, train_y, test_y = train_test_split(embeddings_x, embeddings_y,
                                                        test_size=0.25, random_state=42,
                                                        stratify=embeddings_y)  # , stratify=embeddings_y)
    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    train_x = in_encoder.transform(train_x)
    test_x = in_encoder.transform(test_x)

    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(train_y)
    train_y = out_encoder.transform(train_y)
    test_y = out_encoder.transform(test_y)

    # search hyperparams
    # svc = SVC(probability=True)#, class_weight='balanced')
    # param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 50, 100, 500, 1000], # [0.1, 1, 10, 100],
    #       #'gamma': [5000, 1000, 100, 10, 1, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
    #       'gamma':['scale', 'auto'],
    #       "kernel": ["linear", "rbf", "poly"]} #
    #
    # self.svc_model = GridSearchCV(svc, param_grid)
    #
    # # Fit the model
    # clf = self.svc_model.fit(train_x, train_y)
    # temp = self.svc_model.best_estimator_
    # temp2 = self.svc_model.best_params_
    # print(temp, temp2)

    # fit model
    svc_model = SVC(C=0.1, gamma='scale', kernel='linear', probability=True)
    clf = svc_model.fit(train_x, train_y)

    # predict
    yhat_train = svc_model.predict(train_x)
    yhat_test = svc_model.predict(test_x)
    # score
    score_train = metrics.accuracy_score(train_y, yhat_train)
    score_test = metrics.accuracy_score(test_y, yhat_test)

    with open(classifier_model_path, 'wb') as handle:
        pickle.dump(svc_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(label_encoder_path, 'wb') as handle:
        pickle.dump(out_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Score train: {score_train} /// Score test: {score_test}")

if __name__ == '__main__':
    create_model()