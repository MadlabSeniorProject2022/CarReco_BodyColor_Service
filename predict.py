from preprocess import normalize
import pickle
import cv2

NORM_KNN_MODEL = pickle.load(open('normKNN.sav', 'rb'))

def knn_predict (image: cv2.Mat):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    norm = normalize(lab)
    result = NORM_KNN_MODEL.predict([norm])
    return result[0]