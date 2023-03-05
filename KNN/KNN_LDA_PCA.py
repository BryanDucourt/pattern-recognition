import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from dataset import scan_dataset
from feature import GenerateFeature

data_L, label_L = scan_dataset("/home/bryandu/Develop/Python/patternRec/dataset/train_large/")
data_S, label_S = scan_dataset("/home/bryandu/Develop/Python/patternRec/dataset/train_small/")
data_test, label_test = scan_dataset("/home/bryandu/Develop/Python/patternRec/dataset/test_all/")
data_train = data_L
data_train.extend(data_S)
label_train = np.append(label_L, label_S)

feature_train = np.asarray(GenerateFeature(data_train))
feature_test = np.asarray(GenerateFeature(data_test))


params_c = list(range(17,300,20))
results = []
results_t = []
for comp in params_c:
    pca = PCA(n_components=comp)
    x_train = pca.fit_transform(feature_train)
    x_test = pca.transform(feature_test)
    lda = LinearDiscriminantAnalysis(n_components=17)
    lda.fit(x_train, label_train)
    x_train_lda = lda.transform(x_train)
    x_test_lda = lda.transform(x_test)
    model = KNeighborsClassifier(n_neighbors=8)
    model.fit(x_train_lda, label_train)

    label_pre = model.predict(x_test_lda)
    label_pre_t = model.predict(x_train_lda)
    correct = np.count_nonzero((label_pre==label_test)==True)
    correct_t = np.count_nonzero((label_pre_t == label_train) == True)
    results.append(correct/len(label_test))
    results_t.append(correct_t/len(label_train))

data = np.array(results)
D = data[np.argmax(data)]
print(D)
plt.plot(params_c,results,label='test',marker='o')
plt.plot(params_c,results_t,label='train',marker='o')

plt.legend(loc='best')
plt.show()
pca_show = PCA(n_components=160)
pca_ = pca_show.fit_transform(feature_train)
lda_show = LinearDiscriminantAnalysis(n_components=2)
lda_show.fit(pca_,label_train)
x_t = lda_show.transform(pca_)
plt.scatter(x_t.T[0,:],x_t.T[1,:],marker='o',c=label_train)
plt.show()
# lda_2 = LinearDiscriminantAnalysis(n_components=2)
# lda_2.fit(x_train_pca,label_train)
# x_train = lda_2.transform(x_train_pca)
# plt.scatter(x_train[:,0],x_train[:,1],c=label_train,marker='o')
# plt.show()