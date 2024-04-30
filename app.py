import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('heart_disease_data.csv')

# Preprocess the data
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Create a Streamlit app
st.title("Heart Disease Detection")
st.write("This app uses a random forest classifier to predict the likelihood of heart disease based on various risk factors.")

# Create input fields for the user to enter their data
age = st.slider("Age", 18, 100, 50)
sex = st.selectbox("Sex", ["1", "0"])
cp = st.slider("Chest Pain Type", 100,300,50)
trestbps = st.slider("Resting Blood Pressure", 90, 200, 120)
chol = st.slider("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar", ["1", "0"])
restecg = st.selectbox("Resting ECG", ["1", "0"])
thalach = st.slider("Maximum Heart Rate Achieved", 60, 200, 120)
exang = st.selectbox("Exercise Induced Angina", ["1", "0"])
oldpeak = st.slider("Oldpeak", 0, 10, 5)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", ["2", "1", "0"])
ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 4, 2)
thal = st.selectbox("Thalassemia", ["1", "2", "3"])

# Create a button to submit the input data
submit = st.button("Submit")

# If the button is clicked, make a prediction using the input data
if submit:
    input_data = pd.DataFrame({'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps], 'chol': [chol], 'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach], 'exang': [exang], 'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca], 'thal': [thal]})
    prediction = rfc.predict(input_data)
    st.write("The predicted likelihood of heart disease is:", prediction[0])
    st.write("The accuracy of the model is:", accuracy_score(y_test, rfc.predict(X_test)))
    st.write("The classification report is:")
    st.write(classification_report(y_test, rfc.predict(X_test)))
    st.write("The confusion matrix is:")
    st.write(confusion_matrix(y_test, rfc.predict(X_test)))