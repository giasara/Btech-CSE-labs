#training----------
#1. load the data(our own data)
import pandas as pd
file=pd.read_csv("/content/irisexcel.csv")
x=file[["sepal_length","sepal_width","petal_length","petal_width"]]
y=file["species"]
#cross validation
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split (x,y,test_size=0.2)

#2.load the mode(percepton)
from sklearn.linear_model import Perceptron
a=Perceptron()
#.fit data into a
a.fit(x,y)
#validation
op=a.predict(xtest)
from sklearn.metrics import accuracy_score
score=accuracy_score(ytest,op)
print("accuracy",score*100)
#testing-------predict
op2=a.predict([[1.4,1.5,1.8,1.0]])
print(op2)
