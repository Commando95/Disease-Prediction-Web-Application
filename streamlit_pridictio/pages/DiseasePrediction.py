import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

import joblib

import pickle
from pathlib import Path
import streamlit_authenticator as stauth

names = ["Piyush Chittauria", "Shivam Gupta"]
usernames = ["piyushp", "shivam12"]

# load hashed passwords
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)
    
    
authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
    "prediction_1231", "abcdef", cookie_expiry_days=0)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")




if authentication_status:
    

    st.sidebar.title(f"Welcome {name}")
    def main():
        menu_selection = st.sidebar.radio("Navigation", ["Heart Disease", "Diabetes", "Model Performance", "Confusion Matrics", "DataSet"])
        
        
        if menu_selection == "Heart Disease":
            # DESCRIPTION OF THIS PAGE
            st.write("""
            # HEART DISEASE PREDICTION PAGE

            ON THIS PAGE YOU CAN PREDICT THAT A PERSON HAS HEART DISEASE OR NOT.

            Dataset used in this model, is obtained from the [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
            """)



            # SIDE MENU INFO

            st.sidebar.header('User Input Features')

            st.sidebar.markdown("""
            [Example CSV input file]()
            """)

            # Collects user input features into dataframe
            uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
            if uploaded_file is not None:
                input_df = pd.read_csv(uploaded_file)
            else:
                classifier_name = st.sidebar.selectbox('Select Classifier',("LogisticRegression","Randome Forest", "KNN"))#selecting Classifier
                def user_input_features():
                    age = st.sidebar.slider('Age', 30,80,40)
                    sex = st.sidebar.selectbox('Sex',('male','female'))
                    cp = st.sidebar.slider('ChestPain', 0,3,1)
                    trestbps = st.sidebar.slider('BloodPressure', 100,150,125)
                    chol = st.sidebar.slider('Serum Cholesterol', 120,350,200)
                    fbs = st.sidebar.slider('Fasting Blood Sugar', 0,1,1)
                    restecg = st.sidebar.slider('Electrocardiographic', 0,2,1)
                    thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 100,190,145)
                    exang = st.sidebar.slider('Exercise Induced Angina', 0,1,1)
                    oldpeak = st.sidebar.slider('ST Depression', 0.0,7.0,3.5)
                    slope = st.sidebar.slider('Slope of the Peak Exercise', 0,2,1)
                    ca = st.sidebar.slider('Major Vessels Colored by Fluoroscopy', 0,3,1)
                    thal = st.sidebar.slider('Thalassemia', 0,3,2)
                    
                    
                    if sex == "male":
                        sex=1
                    else:
                        sex=0
                    
                    data = {
                            'age': age,
                            'sex': sex,
                            'cp': cp,
                            'trestbps': trestbps,
                            'chol': chol,
                            'fbs': fbs,
                            'restecg': restecg,
                            'thalach': thalach,
                            'exang': exang,
                            'oldpeak': oldpeak,
                            'slope': slope,
                            'ca': ca,
                            'thal': thal}
                    features = pd.DataFrame(data, index=[0])
                    return features
                input_df = user_input_features()

            

            diabetes_raw = pd.read_csv('heart.csv')
            diabetes = diabetes_raw.drop(columns=['target'])
            df = pd.concat([input_df,diabetes],axis=0)


            df = df[:1] # Selects only the first row (the user input data)
            
            
            def get_classifier(clf_name):
                if clf_name == "Randome Forest":
                    clf = joblib.load('HeartRFC.pk1')
                elif clf_name == "KNN":
                    clf = joblib.load('heartKNN.pkl')
                elif clf_name == "LogisticRegression":
                    clf = joblib.load('Heart_D_model.pk1')
                return clf
            
            
            clf = get_classifier(classifier_name)
            
            
            
            
            

            # Displays the user input features
            st.subheader('User Input features')

            if uploaded_file is not None:
                st.write(df)
            else:
                st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
                st.write(df)

            # Reads in saved classification model

            prediction = clf.predict(df)
            prediction_proba = clf.predict_proba(df)


            st.subheader('Prediction')
            st.write(prediction)
            if prediction == 0:
                st.write("The Person does not have a Heart Disease")
            else:
                st.write("The Person has Heart Disease")

            st.subheader('Prediction Probability')
            st.write(prediction_proba)
            
        
        
        #Diabetes============================================================
        elif menu_selection == "Diabetes":
            st.write("""
            # DIABETES PREDICTION PAGE

            ON THIS PAGE YOU CAN PREDICT THAT A PERSON HAS DIABETES OR NOT.


            Dataset used in this model, is obtained from the [kaggle](https://www.kaggle.com/datasets/saurabh00007/diabetescsv)
            """)
            
            st.sidebar.header('User Input Features')

            st.sidebar.markdown("""
            [Example CSV input file]
            """)

            # Collects user input features into dataframe
            uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
            if uploaded_file is not None:
                input_df = pd.read_csv(uploaded_file)
            else:
                classifier_name = st.sidebar.selectbox('Select Classifier',("LogisticRegression","Randome Forest", "KNN"))#selecting Classifier
                def user_input_features():
                    Pregnancies = st.sidebar.slider('Pregnancies', 0,15,8)
                    Glucose = st.sidebar.slider('Glucose', 60,200,130)
                    BloodPressure = st.sidebar.slider('BloodPressure', 0,130,65)
                    SkinThickness = st.sidebar.slider('SkinThickness', 0,50,25)
                    Insulin = st.sidebar.slider('Insulin', 0,900,850)
                    BMI = st.sidebar.slider('BMI', 0.0,50.0,25.0)
                    DiabetesPedigreeFunction = st.sidebar.slider('DiabetesPedigreeFunction', 0.40,1.999,0.900)
                    Age = st.sidebar.slider('Age', 20,80,40)
                    
                    data = {
                            'Pregnancies': Pregnancies,
                            'Glucose': Glucose,
                            'BloodPressure': BloodPressure,
                            'SkinThickness': SkinThickness,
                            'Insulin': Insulin,
                            'BMI': BMI,
                            'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
                            'Age': Age}
                    features = pd.DataFrame(data, index=[0])
                    return features
                input_df = user_input_features()

            
            diabetes_raw = pd.read_csv('diabetes2.csv')
            diabetes = diabetes_raw.drop(columns=['Outcome'])
            df = pd.concat([input_df,diabetes],axis=0)


            df = df[:1] # Selects only the first row (the user input data)
            
            
            
            def get_classifier(clf_name):
                if clf_name == "Randome Forest":
                    clf = joblib.load('diaRFC.pkl')
                elif clf_name == "KNN":
                    clf = joblib.load('diaKNN.pk1')
                elif clf_name == "LogisticRegression":
                    clf = joblib.load('trained_model.pk1')
                return clf
            
            
            clf = get_classifier(classifier_name)
            
            
            

            # Displays the user input features
            st.subheader('User Input features')

            if uploaded_file is not None:
                st.write(df)
            else:
                st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
                st.write(df)

            prediction = clf.predict(df)
            prediction_proba = clf.predict_proba(df)

            st.subheader('Prediction')
            
            st.write(prediction)
            if prediction == 0:
                st.write("Person is not diabetic")
            else:
                st.write("Person is diabetic")

            st.subheader('Prediction Probability')
            st.write(prediction_proba)
        
        
        # Performance result
        elif menu_selection == "Model Performance":
            st.write("# Model Performace Result")
            st.subheader("Diabetes")
            st.write("Model: LogisticRegression")
            data = pd.read_csv('diabetes2.csv')
            X = data.drop('Outcome', axis=1)
            y = data['Outcome']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model = LogisticRegression()
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            precision = precision_score(y_test, predictions)
            recall = recall_score(y_test, predictions)
            roc_auc = roc_auc_score(y_test, predictions)
            st.write("-    Accuracy:", accuracy)
            st.write("-    f1_score:", f1)
            st.write("-    precision:", precision)
            st.write("-    recall:", recall)
            st.write("-    roc_auc_score:", roc_auc)
            
            st.write("_____________________________________________________________________________")
            st.write("Model: RandomForestClassifier")
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train_scaled, y_train)
            
            y_pred = rf.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)
            st.write("-    Accuracy:", accuracy)
            st.write("-    f1_score:", f1)
            st.write("-    precision:", precision)
            st.write("-    recall:", recall)
            st.write("-    roc_auc_score:", roc_auc)
            
            st.write("_____________________________________________________________________________")
            st.write("Model: KNeighborsClassifier")
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_train_scaled, y_train)


            y_pred = knn.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)
            st.write("-    Accuracy:", accuracy)
            st.write("-    f1_score:", f1)
            st.write("-    precision:", precision)
            st.write("-    recall:", recall)
            st.write("-    roc_auc_score:", roc_auc)
            
            
            st.write("_____________________________________________________________________________")
            st.subheader("Heart Disease")
            st.write("Model: LogisticRegression")
            heart_data = pd.read_csv("heart.csv")
            X = heart_data.drop(columns='target', axis=1)
            y = heart_data['target']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model = LogisticRegression()

            model.fit(X_train_scaled, y_train)
            x_train_prediction = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, x_train_prediction)
            f1 = f1_score(y_test, x_train_prediction)
            precision = precision_score(y_test, x_train_prediction)
            recall = recall_score(y_test, x_train_prediction)
            roc_auc = roc_auc_score(y_test, x_train_prediction)
            st.write("-    Accuracy:", accuracy)
            st.write("-    f1_score:", f1)
            st.write("-    precision:", precision)
            st.write("-    recall:", recall)
            st.write("-    roc_auc_score:", roc_auc)
            
            
            st.write("_____________________________________________________________________________")
            st.write("Model: RandomForestClassifier")
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train_scaled, y_train)
            y_pred = rf.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)
            st.write("-    Accuracy:", accuracy)
            st.write("-    f1_score:", f1)
            st.write("-    precision:", precision)
            st.write("-    recall:", recall)
            st.write("-    roc_auc_score:", roc_auc)
            
            st.write("_____________________________________________________________________________")
            st.write("Model: KNeighborsClassifier")
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_train_scaled, y_train)
            y_predi = knn.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_predi)
            f1 = f1_score(y_test, y_predi)
            precision = precision_score(y_test, y_predi)
            recall = recall_score(y_test, y_predi)
            roc_auc = roc_auc_score(y_test, y_predi)
            st.write("-    Accuracy:", accuracy)
            st.write("-    f1_score:", f1)
            st.write("-   precision:", precision)
            st.write("-    recall:", recall)
            st.write("-    roc_auc_score:", roc_auc)
            
        elif menu_selection == "Confusion Matrics":
            st.write("# CONFUSION MATRIX")
            st.subheader("Diabetes")
            data = pd.read_csv('diabetes2.csv')
            X = data.drop('Outcome', axis=1)
            y = data['Outcome']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            rf = LogisticRegression()
            rf.fit(X_train_scaled, y_train)
            
            y_pred = rf.predict(X_test_scaled)
            
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'], ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
            
            st.write("_____________________________________________________________________________")
            #HD confusion Matrics
            st.subheader("Heart Disease")
            heart_data = pd.read_csv("heart.csv")
            X = heart_data.drop(columns='target', axis=1)
            y = heart_data['target']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            rf = LogisticRegression()
            rf.fit(X_train_scaled, y_train)
            
            y_pred = rf.predict(X_test_scaled)
            
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'], ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)

        elif menu_selection == "DataSet":
            dataset_selection = st.sidebar.radio("Navigation", ["Heart DataSet", "Diabetes DataSet"])
            if dataset_selection== "Heart DataSet":
                st.write("# Heart Disease Dataset")
                st.subheader("Dataset")
                dfh = pd.read_csv("heart.csv")
                st.write(dfh)
                
                st.subheader('Dataset Discription')
                st.write(dfh.describe())
                
                
                # pairplot
                
                #st.subheader('Pairplot of the Dataset')
                #pairplot_fig = sns.pairplot(dfh, hue='target', diag_kind='kde')
                #st.pyplot(pairplot_fig)
                
                # countPlot section
                st.subheader('Countplot of the Target Variable')
                fig_count, ax_count = plt.subplots()
                sns.countplot(x='target', data=dfh, ax=ax_count, palette=['#3498db', '#e74c3c'])
                ax_count.set_title('Count of Heart Disease vs No Heart Disease')
                ax_count.set_xlabel('Target')
                ax_count.set_ylabel('Count')
                st.pyplot(fig_count)
                
                
                
                
                st.subheader('DataSet Visualizations')
                fig, axs = plt.subplots(4, 3, figsize=(20, 20))
                # Age and Heart Disease
                fig, ax = plt.subplots()
                sns.histplot(data=dfh, x='age', hue='target', multiple='stack', palette=['#3498db', '#e74c3c'])
                plt.title('Age Distribution by Heart Disease')
                st.pyplot(fig)
                
                fig, ax = plt.subplots()
                sns.countplot(data=dfh, x='sex', hue='target', palette=['#3498db', '#e74c3c'])
                plt.title('Sex Distribution by Heart Disease')
                plt.xticks([0, 1], ['Female', 'Male'])
                st.pyplot(fig)
                
                # Chest Pain Type and Heart Disease
                fig, ax = plt.subplots()
                sns.countplot(data=dfh, x='cp', hue='target', palette=['#3498db', '#e74c3c'])
                plt.title('Chest Pain Type by Heart Disease')
                st.pyplot(fig)

                # Resting Blood Pressure and Heart Disease
                fig, ax = plt.subplots()
                sns.histplot(data=dfh, x='trestbps', hue='target', multiple='stack', palette=['#3498db', '#e74c3c'])
                plt.title('Resting Blood Pressure by Heart Disease')
                st.pyplot(fig)

                # Cholesterol Levels and Heart Disease
                fig, ax = plt.subplots()
                sns.histplot(data=dfh, x='chol', hue='target', multiple='stack', palette=['#3498db', '#e74c3c'])
                plt.title('Cholesterol Levels by Heart Disease')
                st.pyplot(fig)

                # Fasting Blood Sugar and Heart Disease
                fig, ax = plt.subplots()
                sns.countplot(data=dfh, x='fbs', hue='target', palette=['#3498db', '#e74c3c'])
                plt.title('Fasting Blood Sugar by Heart Disease')
                st.pyplot(fig)

                # Exercise Induced Angina and Heart Disease
                fig, ax = plt.subplots()
                sns.countplot(data=dfh, x='exang', hue='target', palette=['#3498db', '#e74c3c'])
                plt.title('Exercise Induced Angina by Heart Disease')
                st.pyplot(fig)

                # ST Depression and Heart Disease
                fig, ax = plt.subplots()
                sns.histplot(data=dfh, x='oldpeak', hue='target', multiple='stack', palette=['#3498db', '#e74c3c'])
                plt.title('ST Depression by Heart Disease')
                st.pyplot(fig)

                # Number of Major Vessels and Heart Disease
                pfig, ax = plt.subplots()
                sns.countplot(data=dfh, x='ca', hue='target', palette=['#3498db', '#e74c3c'])
                plt.title('Number of Major Vessels by Heart Disease')
                st.pyplot(fig)

                # Thalassemia and Heart Disease
                fig, ax = plt.subplots()
                sns.countplot(data=dfh, x='thal', hue='target', palette=['#3498db', '#e74c3c'])
                plt.title('Thalassemia by Heart Disease')
                st.pyplot(fig)
                
                fig, ax = plt.subplots()
                sns.countplot(data=dfh, x='restecg', hue='target', palette=['#3498db', '#e74c3c'])
                plt.title('Electrocardiographic by Heart Disease')
                st.pyplot(fig)
                
                
            
            elif dataset_selection == "Diabetes DataSet":
                st.write("# Diabetes DataSet")
                st.subheader("Dataset")
                data = pd.read_csv("diabetes2.csv")
                st.write(data)
                
                st.subheader('Dataset Discription')
                st.write(data.describe())
                
                #Countplot
                st.subheader('Countplot of the Target Variable')
                fig_count, ax_count = plt.subplots()
                sns.countplot(x='Outcome', data=data, ax=ax_count, palette=['#3498db', '#e74c3c'])
                ax_count.set_title('Count of Diabetes vs No Diabetes')
                ax_count.set_xlabel('Outcome')
                ax_count.set_ylabel('Count')
                st.pyplot(fig_count)
                
                
                
                
                st.subheader('DataSet Visualizations')
                # Age and Outcome
                fig, ax = plt.subplots()
                sns.histplot(data=data, x='Age', hue='Outcome', multiple='stack', palette=['#3498db', '#e74c3c'], ax=ax)
                ax.set_title('Age Distribution by Outcome')
                st.pyplot(fig)

                # Glucose and Outcome
                fig, ax = plt.subplots()
                sns.histplot(data=data, x='Glucose', hue='Outcome', multiple='stack', palette=['#3498db', '#e74c3c'], ax=ax)
                ax.set_title('Glucose Distribution by Outcome')
                st.pyplot(fig)

                # BloodPressure and Outcome
                fig, ax = plt.subplots()
                sns.histplot(data=data, x='BloodPressure', hue='Outcome', multiple='stack', palette=['#3498db', '#e74c3c'], ax=ax)
                ax.set_title('BloodPressure Distribution by Outcome')
                st.pyplot(fig)

                # SkinThickness and Outcome
                fig, ax = plt.subplots()
                sns.histplot(data=data, x='SkinThickness', hue='Outcome', multiple='stack', palette=['#3498db', '#e74c3c'], ax=ax)
                ax.set_title('SkinThickness Distribution by Outcome')
                st.pyplot(fig)

                # Insulin and Outcome
                fig, ax = plt.subplots()
                sns.histplot(data=data, x='Insulin', hue='Outcome', multiple='stack', palette=['#3498db', '#e74c3c'], ax=ax)
                ax.set_title('Insulin Distribution by Outcome')
                st.pyplot(fig)

                # BMI and Outcome
                fig, ax = plt.subplots()
                sns.histplot(data=data, x='BMI', hue='Outcome', multiple='stack', palette=['#3498db', '#e74c3c'], ax=ax)
                ax.set_title('BMI Distribution by Outcome')
                st.pyplot(fig)

                # DiabetesPedigreeFunction and Outcome
                fig, ax = plt.subplots()
                sns.histplot(data=data, x='DiabetesPedigreeFunction', hue='Outcome', multiple='stack', palette=['#3498db', '#e74c3c'], ax=ax)
                ax.set_title('DiabetesPedigreeFunction Distribution by Outcome')
                st.pyplot(fig)
            
            
            
    if __name__ == "__main__":
        main()
