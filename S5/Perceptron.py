#training----------
#1. load the data(our own data)
x=[[1,2],[6,5],[7,8],[3,7]]
y=["blue","black","blue","black"]
#2.load the mode(percepton)
from sklearn.linear_model import Perceptron
a=Perceptron()
#.fit data into a
a.fit(x,y)
#testing-------predict
op=a.predict([[9,7]])
print(op)
