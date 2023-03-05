import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from dataset import scan_dataset
from feature import GenerateFeature

data_L, label_L = scan_dataset("../dataset/train_large/")
data_S, label_S = scan_dataset("../dataset/train_small/")
data_test, label_test = scan_dataset("../dataset/test_all/")
data_train = data_L
data_train.extend(data_S)
label_train = np.append(label_L, label_S)

feature_train = np.asarray(GenerateFeature(data_train))
feature_test = np.asarray(GenerateFeature(data_test))

params_lda = list(range(1, 18))
results = []
results_t = []
for comp in params_lda:
    lda = LinearDiscriminantAnalysis(n_components=comp)
    lda.fit(feature_train, label_train)
    x_train = lda.transform(feature_train)
    x_test = lda.transform(feature_test)

    model = KNeighborsClassifier(n_neighbors=8)
    model.fit(x_train, label_train)

    label_pre = model.predict(x_test)
    label_pre_t = model.predict(x_train)
    correct = np.count_nonzero((label_pre == label_test) == True)
    correct_t = np.count_nonzero((label_pre_t == label_train) == True)
    results.append(correct / len(label_test))
    results_t.append(correct_t / len(label_train))

plt.plot(params_lda, results, label='test', marker='o')
plt.plot(params_lda, results_t, label='train', marker='o')

plt.legend(loc='best')
plt.show()

# data = np.array(results)
# D = data[np.argmax(data[:, 0])][1]
# K = data[np.argmax(data[:, 0])][2]
# p = data[np.argmax(data[:, 0])][3]
# Precision = data[np.argmax(data[:, 0])][0]
#
# print("LDA Best dimension:%.1f Best K:%.1f Best p:%.1f" % (D, K, p))
# print(Precision)
#
lda_show = LinearDiscriminantAnalysis(n_components=2)
lda_show.fit(feature_train, label_train)
lda_dim_2 = lda_show.transform(feature_train)
plt.scatter(lda_dim_2[:, 0], lda_dim_2[:, 1], marker='o', c=label_train)
plt.show()
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
