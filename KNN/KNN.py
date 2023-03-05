import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from dataset import scan_dataset
from feature import GenerateFeature

data_L,label_L = scan_dataset("../dataset/train_large/")
data_S,label_S = scan_dataset("../dataset/train_small/")
data_test,label_test =scan_dataset("../dataset/test_all/")
data_train = data_L
data_train.extend(data_S)
label_train = np.append(label_L,label_S)

feature_train = np.asarray(GenerateFeature(data_train))
feature_test = np.asarray(GenerateFeature(data_test))
# normalizer = StandardScaler()
# feature_train = normalizer.fit_transform(feature_train)
# feature_test = normalizer.transform(feature_test)
# print(feature_train.shape)
params_k = list(range(1,17))  # 可以选择的K值

results = []
results_t = []
for k in params_k:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(feature_train,label_train)
    label_pre = model.predict(feature_test)
    label_pre_t = model.predict(feature_train)
    correct = np.count_nonzero((label_pre == label_test) == True)/len(label_test)
    correct_t = np.count_nonzero((label_pre_t == label_train) == True)/len(label_train)
    results_t.append(correct_t)
    results.append(correct)

data = np.array(results)
D = data[np.argmax(data)]
plt.plot(params_k,results,label='test',marker='o')
plt.plot(params_k,results_t,label='train',marker='o')
print(D)
plt.legend(loc='best')
plt.show()


model = KNeighborsClassifier(n_neighbors=8)
model.fit(feature_train,label_train)
label_pre = model.predict(feature_test)
matrix = np.zeros((18,18))
for i in range(len(feature_test)):
    matrix[label_pre[i]-1][label_test[i]-1]+=1
plt.matshow(matrix,cmap=plt.cm.Blues)
plt.show()
