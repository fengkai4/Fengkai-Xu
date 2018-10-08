
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
iris_dataset=load_iris()
X=iris_dataset['data']
y=iris_dataset['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,stratify=y,random_state=42)
model=[]
for c in range(1,11):
    tree=DecisionTreeClassifier(max_depth=4,random_state=c)
    model.append(tree.fit(X_train,y_train))
in_sample_accuracy=[]
out_of_sample_accuracy=[]
for a in model:
    in_sample_accuracy.append(a.score(X_train,y_train))
    out_of_sample_accuracy.append(a.score(X_test,y_test))

a=list(range(1,11))
a.append('mean')
a.append('standard')
in_sample_accuracy.append(np.mean(in_sample_accuracy))
in_sample_accuracy.append(np.std(in_sample_accuracy[:-1]))
out_of_sample_accuracy.append(np.mean(out_of_sample_accuracy))
out_of_sample_accuracy.append(np.std(out_of_sample_accuracy[:-1]))
b=pd.DataFrame([in_sample_accuracy,out_of_sample_accuracy,],
                        columns=a,index=['in_sample_accuracy','out_of_sample_accuracy'])
pd.set_option('precision',3)
b
#cross validation
CVS=[]
score=cross_val_score(DecisionTreeClassifier(max_depth=4),X_train,y_train,cv=10)
CVS.append(score)
pd.set_option('precision',3)
c=pd.DataFrame(CVS,columns=['result1','result2','result3','result4','result5','result6','result7','result8','result9','result 10'],)
c['mean']=c.mean(1)
c['standard']=c.std(1)
dt=DecisionTreeClassifier(max_depth=4)
dt.fit(X_train,y_train)
c['Out-of-sample-accuracy']=dt.score(X_test,y_test)
c
print("My name is Fengkai Xu")
print("My NetID is: fengkai4")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")