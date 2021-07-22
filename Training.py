import pickle
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import hog

X = np.loadtxt("TrainData.csv")
y = np.loadtxt("TrainLabels.csv")


def ExtractFeatures(Xn):
    return hog(Xn.reshape([28, 28]), orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1))


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(np.array([ExtractFeatures(x) for x in X]), y, test_size=0.3,
                                                    random_state=937)

################################## SVM ####################################
# defining parameter range
param_grid_SVM = {
    'C': [0.1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.001, 0.0001],
    'kernel': ['poly', 'sigmoid']
}

grid_SVM = GridSearchCV(
    SVC(),
    param_grid_SVM,
    refit=True,
    verbose=3,
    cv=5,
    n_jobs=-1)

# fitting the model for grid search
grid_SVM.fit(X_train, y_train)

# print best parameter after tuning
print(grid_SVM.best_params_)

# print how our model looks after hyper-parameter tuning
print(grid_SVM.best_estimator_)

grid_predictions = grid_SVM.predict(X_test)

# print classification report
print(classification_report(y_test, grid_predictions))

pickle.dump(grid_SVM.best_estimator_, open("SVM.pkl", 'wb'))

################################## KNN ####################################
param_grid_KNN = {
    'n_neighbors': list(range(1, 30, 2)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_KNN = GridSearchCV(
    KNeighborsClassifier(),
    param_grid_KNN,
    refit=True,
    verbose=3,
    cv=5,
    n_jobs=-1)

grid_KNN.fit(X_train, y_train)

# Dictionary containing the parameters (k) used to generate that score
print(grid_KNN.best_params_)

# Shows the best parameters after search
print(grid_KNN.best_estimator_)

grid_predictions = grid_KNN.predict(X_test)

# print classification report
print(classification_report(y_test, grid_predictions))

pickle.dump(grid_KNN.best_estimator_, open("KNN.pkl", 'wb'))
