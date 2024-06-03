import streamlit as st
import pandas as pd
st.header('Registration Form')
col1,col2,col3=st.columns(3)
a=col1.selectbox('',('mr','ms'))
firstname=col2.text_input('firstname')
lastname=col3.text_input('lastname')
st.markdown('---')
designation=st.selectbox('select your designation',('hr','software devloper','teacher','student'))
dateofbirth=st.date_input('enter your birth od date')
gender=st.radio('gender',('male','female'))
#age=st.slider('your age is in '0,1000,1000,100)
button=st.button('submit')
if button:
    info={'mr''ms':a,'firstname':firstname,'lastname':lastname,'designation':designation,'gender':gender,'button':button}

    st.json(info)
    st.success("submitedd sucessfully")
    st.balloons()


