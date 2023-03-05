import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from dataset import scan_dataset
from feature import GenerateFeature
import warnings

warnings.filterwarnings("ignore")


data_L, label_L = scan_dataset("./dataset/train_large/")
data_S, label_S = scan_dataset("./dataset/train_small/")
data_test, label_test = scan_dataset("./dataset/test_all/")
data_train = data_L
data_train.extend(data_S)
label_train = np.append(label_L, label_S)

feature_train = np.asarray(GenerateFeature(data_train))
feature_test = np.asarray(GenerateFeature(data_test))
normalizer = StandardScaler()
feature_train = normalizer.fit_transform(feature_train)
feature_test = normalizer.transform(feature_test)

p_hidden_size_1 = [10, 20, 30, 40, 50, 60]
p_hidden_size_2 = [10, 20, 30, 40, 50, 60]

p_hidden_size = list(zip(p_hidden_size_2,p_hidden_size_1))
activation = ['logistic', 'tanh', 'relu']
solver = ['lbfgs', 'sgd', 'adam']
param_grid = {"hidden_layer_sizes": p_hidden_size}

model = GridSearchCV(estimator=MLPClassifier(activation='relu',solver='lbfgs'), param_grid=param_grid, cv=5)
model.fit(feature_train, label_train)
print(model.best_params_)

predict_model = model.best_estimator_
label_pre = predict_model.predict(feature_test)
correct = np.count_nonzero((label_pre == label_test) == True)
print("Predict precision is: ", correct / len(label_test))
result = model.cv_results_
np.save('mlp.npy',result)