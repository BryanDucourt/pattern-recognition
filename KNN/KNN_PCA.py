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
params_c = list(range(1,300,10))

results = []
results_t = []
for comp in params_c:
    pca = PCA(n_components=comp)
    x_train = pca.fit_transform(feature_train)
    x_test = pca.transform(feature_test)

    model = KNeighborsClassifier(n_neighbors=8)
    model.fit(x_train,label_train)

    label_pre = model.predict(x_test)
    label_pre_t = model.predict(x_train)
    correct = np.count_nonzero((label_pre==label_test)==True)
    correct_t = np.count_nonzero((label_pre_t == label_train) == True)
    results.append(correct/len(label_test))
    results_t.append(correct_t/len(label_train))

plt.plot(params_c,results,label='test',marker='o')
plt.plot(params_c,results_t,label='train',marker='o')

plt.legend(loc='best')
plt.show()

data = np.array(results)

pca = PCA(n_components=45)
x_train = pca.fit_transform(feature_train)
x_test = pca.transform(feature_test)

model = KNeighborsClassifier(n_neighbors=8)
model.fit(x_train,label_train)
label_pre = model.predict(x_test)
correct = np.count_nonzero((label_pre==label_test)==True)
print(correct/len(label_test))
# K = data[np.argmax(data[:,0])][2]
# p = data[np.argmax(data[:,0])][3]
# Precision = data[np.argmax(data[:,0])][0]
#
# print("Best dimension:%.1f Best K:%.1f Best p:%.1f" % (D, K, p))
#
# print("Best precision:%6.3f" % Precision)
#
# pca_show = PCA(n_components=2)
# pca_ = pca_show.fit_transform(feature_train)
# plt.scatter(pca_.T[0,:],pca_.T[1,:],marker='o',c=label_train)
# plt.show()
#
# plt.figure(figsize=(11, 8))
# for test in results:
#     plt.plot(test[4]['param_n_neighbors'], test[4]['mean_test_score'], label=f'dim={test[1]}')
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# plt.xlabel('n neighbors')
# plt.ylabel('mean precision')
# plt.show()
# plt.figure(figsize=(11, 8))
# for test in results:
#     plt.scatter(test[4]['param_n_neighbors'], test[4]['mean_fit_time'], label=f'dim={test[1]}')
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# plt.xlabel('n neighbors')
# plt.ylabel('mean fit time')
# plt.show()
# plt.figure(figsize=(11, 8))
# for test in results:
#     plt.scatter(test[4]['param_n_neighbors'], test[4]['mean_score_time'], label=f'dim={test[1]}')
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# plt.xlabel('n neighbors')
# plt.ylabel('mean score time')
# plt.show()