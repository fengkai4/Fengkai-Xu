import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split
import scipy as sp
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import os
from sklearn import linear_model
from sklearn.model_selection import cross_val_score


os.getcwd()

df=pd.read_csv('concrete.csv')

df.shape
df.info()
df.head()
cols = ['cement', 'slag', 'ash', 'water', 'superplastic','coarseagg','fineagg','age','strength']

for i in ['cement', 'slag', 'ash', 'water', 'superplastic','coarseagg','fineagg','age','strength']:
    plt.figure()
    sns.boxplot(x=i,data=df)


sns.pairplot(df[cols], size=2.5)
plt.tight_layout()
plt.savefig('10_03.png', dpi=300)
plt.show()


cm = np.corrcoef(df[cols].values.T)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 8},
                 yticklabels=cols,
                 xticklabels=cols)

plt.tight_layout()
plt.savefig('10_04.png', dpi=300)
plt.show()

X = df[['cement']].values
y = df['strength'].values
#

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=20)
    plt.plot(X, model.predict(X), color='black', lw=2)    
    return 

slr = LinearRegression()
slr.fit(X_train, y_train)
y_pred = slr.predict(X_train)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)
# Print R^2 
print('R^2: %.3f' % slr.score(X_train, y_train))



lin_regplot(X_train, y_train, slr)
plt.xlabel('[cement] ')
plt.ylabel('[strength] ')

plt.savefig('10_07.png', dpi=300)
plt.show()



X = df.iloc[:, :-1].values
y = df['strength'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)




slr = LinearRegression()

slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

np.set_printoptions(precision=3)
print('Slope:' , slr.coef_)
print('Intercept: %.3f' % slr.intercept_)
# Print R^2 
print('R^2: %.3f' % slr.score(X_train, y_train))




ary = np.array(range(100000))




plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=90, color='black', lw=2)
plt.xlim([-10, 90])
plt.tight_layout()

plt.savefig('10_09.png', dpi=300)
plt.show()





print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

# # Using regularized methods for regression



def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()



print('ridge regression')
# Ridge regression:


alpha_space = np.logspace(-2, 0, 4)
ridge_scores = []
ridge_scores_std = []
# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

for alpha in alpha_space:

    ridge.alpha = alpha
    ridge.fit(X_train, y_train)
    y_train_pred = ridge.predict(X_train)
    y_test_pred = ridge.predict(X_test)
#    plt.plot(range(len(df.columns)-1), ridge.coef_)
#    plt.xticks(range(len(df.columns)-1), df.columns.values, rotation=60)
#    plt.margins(0.02)
#    plt.show()
    print('Slope:' , ridge.coef_)
    print('intercept:' , ridge.intercept_)
    print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

    ary = np.array(range(100000))




    plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=90, color='black', lw=2)
    plt.xlim([-10, 90])
    plt.tight_layout()
    plt.title('alpha='+str(alpha))
    plt.savefig('ridge alpha='+str(alpha)+'10_09.png', dpi=300)
    plt.show()







print('lasso regression')
# LASSO regression:


alpha_space = np.logspace(-2, 0, 4)
lasso_scores = []
lasso_scores_std = []

# Create a ridge regressor: lasso
lasso = Lasso(normalize=True)


for alpha in alpha_space:
# Specify the alpha value to use: ridge.alpha
    lasso.alpha = alpha
    lasso.fit(X_train, y_train)
    y_train_pred = lasso.predict(X_train)
    y_test_pred = lasso.predict(X_test)
#    plt.plot(range(len(df.columns)-1), ridge.coef_)
#    plt.xticks(range(len(df.columns)-1), df.columns.values, rotation=60)
#    plt.margins(0.02)
#    plt.show()
    print('Slope:' , lasso.coef_)
    print('intercept:' , lasso.intercept_)

    print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

    ary = np.array(range(100000))




    plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=90, color='black', lw=2)
    plt.xlim([-10, 90])
    plt.tight_layout()
    plt.title('alpha='+str(alpha))
    plt.savefig('lasso alpha='+str(alpha)+'10_09.png', dpi=300)
    plt.show()







print('Elastic Net regression')
# Elastic Net regression:


ratio_space = np.logspace(-2, 0, 4)
elanet_scores = []
elanet_scores_std = []

# Create a ridge regressor: lasso
elanet = ElasticNet(alpha=1.0)


for ratio in ratio_space:
# Specify the alpha value to use: ridge.alpha
    elanet.l1_ratio = ratio
    elanet.fit(X_train, y_train)
    y_train_pred = elanet.predict(X_train)
    y_test_pred = elanet.predict(X_test)
    print('Slope:' , elanet.coef_)
    print('intercept:' , elanet.intercept_)

    print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

    ary = np.array(range(100000))




    plt.title('ratio='+str(ratio))
    plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=90, color='black', lw=2)
    plt.xlim([-10, 90])
    plt.tight_layout()

    plt.savefig('10_09.png', dpi=300)
    plt.show()
print("My name is Fengkai Xu")
print("My NetID is: fengkai4")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

