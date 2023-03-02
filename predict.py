from preprocess import normalize, raw_palette
import pickle
import cv2

NORM_KNN_MODEL = pickle.load(open('normKNN.sav', 'rb'))
RAW_RF_MODEL = pickle.load(open('rawModel.sav', 'rb'))

def knn_predict (image: cv2.Mat):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    norm = normalize(lab)
    result = NORM_KNN_MODEL.predict([norm])
    return result[0]

def raw_predict (image: cv2.Mat):
    par = raw_palette(image)
    result = RAW_RF_MODEL.predict([par])
    return result[0]
