#training
#i.load the DATA(excel sheet)
import pandas as pd
file=pd.read_csv(r"C:\Users\Gouri Krishna\Downloads\13 MLP winequality-red.csv")
x=file[["total sulfur dioxide","density","pH","sulphates","alcohol" ]]
y=file[["quality"]]
#cross validation
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)

#2.load the model(perceptron)
from sklearn.linear_model import Perceptron
r=Perceptron()
#fit data into model
r.fit(xtrain,ytrain)
#validation
op=r.predict(xtest)
from sklearn.metrics import accuracy_score
score=accuracy_score(ytest,op)
print("Accuracy=",score*100)

#testing
op1=r.predict([[00.5,0.5,00.33,00.555,0.999]])
print(op1)

from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
cm=confusion_matrix(ytest,op)
print(cm)
cmd=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=r.classes_)
cmd.plot()
