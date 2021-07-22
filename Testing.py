import pickle
import numpy as np
from skimage.feature import hog


def ExtractFeatures(Xn):
    return hog(Xn.reshape([28, 28]), orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1))


Xts = np.loadtxt("TestData.csv")
Xts_features = np.array([ExtractFeatures(x) for x in Xts])

SVM = pickle.load(open("SVM.pkl", 'rb'))
KNN = pickle.load(open("KNN.pkl", 'rb'))

Yts = SVM.predict(Xts_features)
np.savetxt("SVMPredictions.csv", Yts)

Yts = KNN.predict(Xts_features)
np.savetxt("KNNPredictions.csv", Yts)
