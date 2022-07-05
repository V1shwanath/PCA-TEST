import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


class FunctionInfo:
	
	def __init__(self):
		
		# Function information dictionary - { function name: [ list of arguments, return type, funtion description ] }
		
		self.func_dict =   {	
								'TKPY_NAIVE_BAYES_accuracy': [['float'], ['float'], ['Accuracy of the model']],
								'TKPY_NAIVE_BAYES_predprobabilities' : [['array'],['float'],['Prediciton probabilities for if a person survived or not']],
								'TKPY_NAIVE_BAYES_CrossVal' :[['float'],['array'],['Crossvalidation test for the training and testing model']],
								'TKPY_NAIVE_BAYES_pred': [['float'],['array'],['predicted values for any value']],
								'TKPY_NAIVE_BAYES_ConfusionMatrix' : [['float','float'],['matrix'],["confusion_matrix"]]
							}
	def get_func_dict(self):
		return self.func_dict



df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis = 1, inplace = True)
df_target = df.Survived
inputs = df.drop("Survived",axis = 1 , inplace = True)
dummies = pd.get_dummies(df.Sex)
pd.concat([df,dummies],axis=1)
df.Age  = df.Age.fillna(df.Age.mean())
df.drop(["Sex"],axis = 1,inplace=True)
X_train, X_test, y_train, y_test = train_test_split(df,df_target,test_size=0.2)
model = GaussianNB()
model.fit(X_train,y_train)


def TKPY_NAIVE_BAYES_accuracy(dummy):
    return float(model.score(X_test,y_test))

def TKPY_NAIVE_BAYES_predprobabilities(No_of_pred):
    output = model.predict_proba(X_test[:No_of_pred])
    return np.array(output)

def TKPY_NAIVE_BAYES_CrossVal(No_of_crossVal):
    valscore = cross_val_score(GaussianNB(),X_train, y_train, cv=No_of_crossVal)
    return print(np.array(valscore))

def TKPY_NAIVE_BAYES_pred(No_of_pred):
    output = model.predict(X_test[0:No_of_pred])
    return np.array(output)

def TKPY_NAIVE_BAYES_ConfusionMatrix(y_test,y_pred):
    conf_mat = confusion_matrix(y_test, y_pred,)
    return np.array(conf_mat)

print(TKPY_NAIVE_BAYES_pred(10))