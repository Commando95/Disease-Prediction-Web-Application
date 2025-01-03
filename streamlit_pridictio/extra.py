# Displays the user input features
            st.subheader('User Input features')

            if uploaded_file is not None:
                st.write(df)
            else:
                st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
                st.write(df)

            # Reads in saved classification model

            model = joblib.load('Heart_D_model.pk1')
            # Apply model to make predictions
            prediction = model.predict(df)
            prediction_proba = model.predict_proba(df)


            st.subheader('Prediction')
            st.write(prediction)
            if prediction == 0:
                st.write("The Person does not have a Heart Disease")
            else:
                st.write("The Person has Heart Disease")

            st.subheader('Prediction Probability')
            st.write(prediction_proba)