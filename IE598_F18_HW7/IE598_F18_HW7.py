import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import numpy as np
df_wine= pd.read_csv('wine.csv')
X,y = df_wine.iloc[:,:-1].values, df_wine['Class'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1, random_state=42) 
count=[i for i in range(1,51)]
wine_train_score=[]
wine_test_score=[]
for i in count:
    rf = RandomForestClassifier(n_estimators = i)
    rf.fit(X_train, y_train)
    a=rf.score(X_train,y_train)
    b=rf.score(X_test,y_test)
    wine_train_score.append(a)
    wine_test_score.append(b)
parameters_rf = {'n_estimators':range(1,100)}


result=pd.DataFrame([count,wine_train_score,wine_test_score,],
                        columns=None,index=['estimators','in_sample_accuracy','out_of_sample_accuracy'])
result
rf = RandomForestClassifier(random_state=42)
grid_rf = GridSearchCV(estimator= rf,param_grid= parameters_rf,scoring="accuracy",cv=10)
grid_rf.fit(X,y)
print(grid_rf.best_params_)
best_model = grid_rf.best_estimator_
feat_labels = df_wine.columns[:-1] 
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
print(indices)
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]],importances[indices[f]]))
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),importances[indices],align='center')
plt.xticks(range(X_train.shape[1]),feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()
print("My name is Fengkai Xu")
print("My NetID is: fengkai4")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")