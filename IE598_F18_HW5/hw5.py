import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC
import seaborn as sns

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

df_wine.shape
df_wine.info()
df_wine.head()
pd.set_option('precision',2)
df_wine.describe()


sns.boxplot(x='Class label', y ='Alcohol',data=df_wine)
plt.xlabel('Class label')
plt.ylabel('Alcohol')
plt.show()

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
pd.plotting.scatter_matrix(df_wine,c = y,figsize=[25,25], s=15,marker='D')
plt.savefig('5_01.png', dpi=300)
plt.show()


cm = np.corrcoef(df_wine[df_wine.columns].values.T)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 5},
                 yticklabels=df_wine.columns,
                 xticklabels=df_wine.columns)
plt.tight_layout()
plt.savefig('10_04.png', dpi=300)
plt.show()

X = df_wine[['Alcohol']].values
y = df_wine['Proline'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=20)
    plt.plot(X, model.predict(X), color='black', lw=2)    
    return 
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y,random_state=42)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


log = LogisticRegression()
log.fit(X_train_std, y_train)
print('LR_train accuracy score: %.5f' % (log.score(X_train_std, y_train)))
print('LR_test accuracy score: %.5f' % (log.score(X_test_std, y_test)))


svm = SVC()
svm.fit(X_train_std, y_train)
print('SVM_train accuracy score: %.5f' % (svm.score(X_train_std, y_train)))
print('SVM_test accuracy score: %.5f' % (svm.score(X_test_std, y_test)))


#PCA
#LogisticRegression
log1 = LogisticRegression()
pca= PCA(n_components=2)
X_train_pca=pca.fit_transform(X_train_std)
X_test_pca=pca.transform(X_test_std)
log1.fit(X_train_pca, y_train)
print('LR_pca_train accuracy score: %.5f' % (log1.score(X_train_pca, y_train)))
print('LR_pca_test accuracy score: %.5f' % (log1.score(X_test_pca, y_test)))

#SVM
svm1 = SVC()
pca= PCA(n_components=2)
X_train_pca=pca.fit_transform(X_train_std)
X_test_pca=pca.transform(X_test_std)
svm1.fit(X_train_pca, y_train)
print('SVM_pca_train accuracy score: %.5f' % (svm1.score(X_train_pca, y_train)))
print('SVM_pca_test accuracy score: %.5f' % (svm1.score(X_test_pca, y_test)))

#LDA
#LogisticRegression
log2 = LogisticRegression()
lda= LDA(n_components=2)
X_train_lda=lda.fit_transform(X_train_std, y_train)
X_test_lda=lda.transform(X_test_std)
log2.fit(X_train_lda, y_train)
print('LR_lda_train accuracy score: %.5f' % (log2.score(X_train_lda, y_train)))
print('LR_lda_test accuracy score: %.5f' % (log2.score(X_test_lda, y_test)))

#SVM
svm2 = SVC()
lda= LDA(n_components=2)
X_train_lda=lda.fit_transform(X_train_std, y_train)
X_test_lda=lda.transform(X_test_std)
svm2.fit(X_train_lda, y_train)
print('SVM_lda_train accuracy score: %.5f' % (svm2.score(X_train_lda, y_train)))
print('SVM_lda_test accuracy score: %.5f' % (svm2.score(X_test_lda, y_test)))

#KPCA
#LogisticRegression
gamma_space = np.logspace(-2,0,4)
scikit_kpca =KernelPCA(n_components=13,kernel='rbf')
for gamma in gamma_space:
    scikit_kpca.gamma = gamma
    X_train_kpca= scikit_kpca.fit_transform(X_train_std)
    X_test_kpca= scikit_kpca.transform(X_test_std)
    log3 = LogisticRegression()
    log3.fit(X_train_kpca, y_train)
    print('(gamma:'+str(gamma)+') LR_kpca_train accuracy score: %.5f' % (log3.score(X_train_kpca, y_train)))
    print('(gamma:'+str(gamma)+') LR_kpca_test accuracy score: %.5f' % (log3.score(X_test_kpca, y_test)))

#SVM
gamma_space = np.logspace(-2, 0, 4)
scikit_kpca =KernelPCA(n_components=13,kernel='rbf')
for gamma in gamma_space:
    scikit_kpca.gamma = gamma
    X_train_kpca= scikit_kpca.fit_transform(X_train_std)
    X_test_kpca= scikit_kpca.transform(X_test_std)
    svm3 = SVC()
    svm3.fit(X_train_kpca, y_train)
    print('(gamma:'+str(gamma)+') SVM_kpca_train accuracy score: %.5f' % (svm3.score(X_train_kpca, y_train)))
    print('(gamma:'+str(gamma)+') SVM_kpca_test accuracy score: %.5f' % (svm3.score(X_test_kpca, y_test)))
print("My name is {Fengkai Xu}")
print("My NetID is: {fengkai4}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")