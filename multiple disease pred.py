# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 02:33:34 2023
@author: manoj
"""


import pickle
import numpy as np
import streamlit as st
from PIL import Image
import random
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier





# loading the saved models

diabetes_model = pickle.load(open('C:/Users/manoj/Desktop/Multiple Disease Prediction System/Saved Model/diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open('C:/Users/manoj/Desktop/Multiple Disease Prediction System/Saved Model/heart_disease_model.sav','rb'))

parkinsons_model = pickle.load(open('C:/Users/manoj/Desktop/Multiple Disease Prediction System/Saved Model/parkinsons_model.sav', 'rb'))

#Covid-19 Prediction
#df1=pd.read_csv("Covid-19 Predictions.csv")
df1=pd.read_csv("C:/Users/manoj/Desktop/Multiple Disease Prediction System/all datasets/Covid-19 Predictions.csv")

x1=df1.drop("Infected with Covid19",axis=1)
x1=np.array(x1)
y1=pd.DataFrame(df1["Infected with Covid19"])
y1=np.array(y1)
x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.2,random_state=0)
model1=RandomForestClassifier()
model1.fit(x1_train,y1_train)




# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Multiple Disease Prediction System ',
                          
                          [
                           'Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Parkinsons Prediction',
                           'Covid-19',
                            'Doctor consultation'],
                          icons=['','activity','heart','person'],
                          default_index=0)

#Home



            
# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    
    # page title
    st.title('Diabetes Prediction using ML')  
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    
    with col2:
        Insulin = st.text_input('Insulin Level')
    
    with col3:
        BMI = st.text_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'The person is diabetic'
        else:
          diab_diagnosis = 'The person is not diabetic'
        
    st.success(diab_diagnosis)
    
    
# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age') 

    with col2:
        sex = st.text_input('Sex')
        
    with col3:
        cp = st.text_input('Chest Pain types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
     #'age_int','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):

        age_float = float(age)
        sex_float = float(sex)
        cp_float = float(cp)
        trestbps_float = float(trestbps)
        chol_float = float(chol)
        fbs_float = float(fbs)
        restecg_float = float(restecg)
        thalach_float = float(thalach)
        exang_float = float(exang)
        oldpeak_float = float(oldpeak)
        slope_float = float(slope)
        ca_float = float(ca)
        thal_float = float(thal)

        heart_prediction = heart_disease_model.predict([[ age_float, sex_float, cp_float, trestbps_float, chol_float, fbs_float, restecg_float, thalach_float, exang_float, oldpeak_float, slope_float, ca_float, thal_float]])                          


      
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)
        
    
    

# Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):
    
    # page title
    st.title("Parkinson's Disease Prediction using ML")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[float(fo), float(fhi), float(flo), float(Jitter_percent), float(Jitter_Abs), float(RAP), float(PPQ),float(DDP),float(Shimmer),float(Shimmer_dB),float(APQ3),float(APQ5),float(APQ),float(DDA),float(NHR),float(HNR),float(RPDE),float(DFA),float(spread1),float(spread2),float(D2),float(PPE)]])                          
        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)
    

#Covid-19 Page

if (selected == 'Covid-19'):
    st.header("Know If You Are Affected By Covid-19")
    st.write("All The Values Should Be In Range Mentioned")
    drycough=st.number_input("Rate Of Dry Cough (0-20)",min_value=0,max_value=20,step=1)
    fever=st.number_input("Rate Of Fever (0-20)",min_value=0,max_value=20,step=1)
    sorethroat=st.number_input("Rate Of Sore Throat (0-20)",min_value=0,max_value=20,step=1)
    breathingprob=st.number_input("Rate Of Breathing Problem (0-20)",min_value=0,max_value=20,step=1)
    prediction1=model1.predict([[drycough,fever,sorethroat,breathingprob]])[0]

    if st.button("Predict"):
        if prediction1=="Yes":
            st.warning("You Might Be Affected By Covid-19")
        elif prediction1=="No":
            st.success("You Are Safe")    

    


    
def random_doctor_response():
    """Generate a random response from a simulated doctor."""
    responses = [
        "It's always a good idea to drink plenty of water and get enough rest, no matter what's going on with your body.",
        "Make sure you're eating a balanced diet and getting enough exercise to keep your body healthy.",
        "Remember that stress and anxiety can affect your physical health too, so don't forget to take care of your mental health as well.",
        "If you're not feeling better in a few days, or if your symptoms get worse, consider seeing a doctor in person for a more thorough evaluation.",
        "You know your body best, so if something doesn't feel right, trust your instincts and seek medical advice if needed.",
        "It sounds like you might be experiencing a common cold. Get plenty of rest, stay hydrated, and take over-the-counter cold medicine if needed.",
        "Based on your symptoms, it's possible that you have a sinus infection. Make an appointment with your doctor to get a proper diagnosis and treatment.",
        "Your symptoms could be caused by allergies. Try taking an antihistamine and avoiding any triggers that you know of.",
        "It's important to get enough sleep and manage your stress levels to keep your immune system healthy and prevent illness.",
        "If you're experiencing pain or discomfort, consider taking an over-the-counter pain reliever like acetaminophen or ibuprofen.",
    ]
    return random.choice(responses)

if selected == "Doctor consultation":
    st.title("Free Doctor Consultation")
    image = Image.open('C:/Users/manoj/Desktop/Multiple Disease Prediction System/doctor_image.jpg')

    st.image(image, caption='Lets connect')    

    st.write("Please describe your symptoms below:")
    symptoms = st.text_input("Symptoms", "")

    if st.button("Submit"):
        st.write("You said:", symptoms)
        response = random_doctor_response()
        st.write("Here's what the doctor says:")
        st.write(response)
        
        st.markdown("To connect with a doctor for a virtual appointment, please use the following Google Meet link:")
        st.markdown("[Google Meet Link](https://meet.google.com/abc-def-ghi)")

    st.write("Here are some general health guidelines to keep in mind:")
    st.write("- Drink plenty of water and get enough rest")
    st.write("- Eat a balanced diet and get enough exercise")
    st.write("- Manage stress and take care of your mental health")
    st.write("- If your symptoms persist or worsen, consider seeing a doctor in person")


    image = Image.open("C:/Users/manoj/Desktop/Multiple Disease Prediction System/healthy.jpg")
    st.image(image, caption="Stay healthy!", width=600, use_column_width=False)
    st.markdown("<h3 style='text-align: center'>Stay healthy!</h3>", unsafe_allow_html=True)       
        
    