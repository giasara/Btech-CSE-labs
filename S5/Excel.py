#training----------
#1. load the data(our own data)
import pandas as pd
file=pd.read_csv("/content/irisexcel.csv")
x=file[["sepal_length","sepal_width","petal_length","petal_width"]]
y=file["species"]
#2.load the mode(percepton)
from sklearn.linear_model import Perceptron
a=Perceptron()
#.fit data into a
a.fit(x,y)
#testing-------predict
op=a.predict([[1.4,1.5,1.8,1.0]])
print(op)
