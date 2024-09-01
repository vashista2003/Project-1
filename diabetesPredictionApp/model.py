import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pickle

diabetes_data = pd.read_csv('diabetes.csv')
x = diabetes_data.drop(columns='Outcome',axis=1)
y = diabetes_data['Outcome']
scaler = StandardScaler()
scaler.fit(x)
standardized_data = scaler.transform(x)
x = standardized_data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)

classifier = svm.SVC(kernel='linear')
classifier.fit(x_train,y_train)
pred = classifier.predict(x_test)
training_data_accuracy = accuracy_score(pred,y_test)
print(training_data_accuracy)
pickle.dump(classifier,open("model.pkl",'wb'))
model = pickle.load(open('model.pkl','rb'))
#print(x_train)
a = np.asanyarray([[4,110,92,0,0,37.6,0.191,30]])
std_data = scaler.transform(a)
print(std_data)
print(model.predict(std_data))
