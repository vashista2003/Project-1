import numpy as np
import pandas as pd
from flask import Flask,request,jsonify,render_template
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


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    d1 = request.form.get("a")
    d2 = request.form.get("b")
    d3 = request.form.get("c")
    d4 = request.form.get("d")
    d5 = request.form.get("e")
    d6 = request.form.get("f")
    d7 = request.form.get("g")
    d8 = request.form.get("h")
    a = np.asarray([[d1,d2,d3,d4,d5,d6,d7,d8]])
    std_data = scaler.transform(a)
    prediction=model.predict(std_data)
    if prediction[0] == 0:
        return render_template('index.html',Prediction_text="The person is not suffering with diabetes")
    else:
        return render_template('index.html',Prediction_text="The person is suffering with diabetes")

if __name__=='__main__':
    app.run(debug=True)

