import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
st.title('visulation')
df=pd.read_csv(r'C:\Users\Star\Desktop\Stuffs(IMP)\sreamlitProjects(stuff)\3_visualization\tips.csv')
st.dataframe(df)




