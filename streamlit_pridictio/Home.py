import streamlit as st

def main():
    st.title("DISEASE PREDICTION WEB APP")
    
    

    # Sidebar menu
    menu_selection = st.sidebar.radio("Go to", ["Home", "Contact"])

    # Conditional content based on menu selection
    if menu_selection == "Home":
        st.write('''In recent years, the prevalence of chronic diseases such as diabetes and heart disease has increased significantly, posing substantial public health challenges worldwide.
                Early detection and management of these conditions are crucial in reducing morbidity and mortality rates. Machine learning offers promising solutions for predictive analytics
                in healthcare, enabling the development of models that can accurately predict the likelihood of a disease based on patient data.
                This project focuses on building and deploying machine learning models to predict diabetes and heart disease. By leveraging historical patient data and advanced
                machine learning techniques, the project aims to provide a reliable and accessible tool for early disease prediction. This can assist healthcare providers in making
                informed decisions and improving patient outcomes.''')
        st.write("**Objectives:**")
        st.markdown('''- **Data Collection and Preparation:** Gather and preprocess datasets for diabetes and heart disease, ensuring the data is clean and suitable for training machine learning models.''')
        st.markdown('''- **Model Training and Evaluation:** Develop predictive models using various algorithms such as Logistic Regression, K-Nearest Neighbors (KNN), Random Forest, and Decision Trees. Evaluate these models using metrics such as accuracy, precision, recall, F1 score,
                    and ROC AUC score to identify the most effective model.''')
        st.markdown('''- **Web Application Development:** Create a user-friendly web application using Flask to allow users to input their health parameters and receive predictions on their risk of diabetes and heart disease.''')
        st.markdown('''- **Model Deployment:** Deploy the trained models within the web application to provide real-time predictions based on user input.''')
        
    elif menu_selection == "Contact":
        st.header(":mailbox: Get In Touch With Me!")


        contact_form = """
        <form action="https://formsubmit.co/chittauriapiyush@gmail.com" method="POST">
            <input type="hidden" name="_captcha" value="false">
            <input type="text" name="name" placeholder="Your name" required>
            <input type="email" name="email" placeholder="Your email" required>
            <textarea name="message" placeholder="Your message here"></textarea>
            <button type="submit">Send</button>
        </form>
        """

        st.markdown(contact_form, unsafe_allow_html=True)

        # Use Local CSS File
        def local_css(file_name):
            with open(file_name) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


        local_css("style/style.css")
        
        
        
        
        
if __name__ == "__main__":
    main()
