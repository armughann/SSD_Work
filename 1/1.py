import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score

data = pd.read_csv("titanic.csv")

#
# def checkAdult(age):
#     if age >= 18:
#         return "Adult"
#     else:
#         return "Child"

arr = np.array(data["Fare"])
normarr = preprocessing.normalize(arr)
data["NormFare"] = arr
trainingData = data[["Pclass", "NormFare", "Gender", "Survived"]]


def cattoNum(series):
    series = series.astype('category')
    return series.cat.codes


catData = trainingData[["Pclass", "NormFare", "Gender", "Survived"]].apply(cattoNum)
trainingData[["Pclass", "NormFare", "Gender", "Survived"]] = catData
# print(trainingData.head())
trainingData = trainingData.dropna()

x = trainingData[["Survived"]]
y = trainingData[["Pclass", "NormFare", "Gender"]]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
dec = DecisionTreeClassifier(criterion="entropy")
ndec = dec.fit(X_train, y_train)
result = ndec.predict(X_test)

print(accuracy_score(result, y_test))
