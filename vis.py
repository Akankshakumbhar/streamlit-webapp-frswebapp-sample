import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
st.title('visulation')
df=pd.read_csv(r'C:\Users\Star\Desktop\Stuffs(IMP)\sreamlitProjects(stuff)\3_visualization\tips.csv')
st.dataframe(df)
value_counts=(df['sex'].value_counts())

with st.container():

            st.write('boxplot')
            fig,ax=plt.subplots()
            sns.boxplot(x='sex',y='total_bill',data=df)
            st.pyplot(fig)

fig,ax=plt.subplots()

st.pyplot(fig)


with st.container():
                    chart=('bar','hist','boxplot','volinplot')
                    chart_sel=st.selectbox('select the type of chart',chart)
                    fig,ax=plt.subplots()
                    if chart_sel=='bar':
                                              sns.barplot(x='sex', y='total_bill', data=df)
                    elif chart_sel=='boxplot':
                                               sns.boxplot(x='sex',y='total_bill',data=df)
                    elif chart_sel=='volinplot':
                                                sns.violinplot(x='sex',y='total_bill',data=df)
                    else:
                          sns.histplot(x='total_bill',hue='sex',data=df)
st.pyplot(fig)

with st.container():
                     features=('sex','total_bills','day','time','size')
                     selectfeat=st.selectbox('select features',features)
                     if selectfeat=='sex':
                                           st.write(" i am  gender")
                     elif selectfeat=='total_bills':
                                                    st.write("bills")
                     elif selectfeat=='time':
                                              st.write("time i am")
                     else:
                           st.write("nmcbc")


fig,ax=plt.subplots()

hue_type=('time','day','size','sex')
select=st.selectbox('select hue',hue_type)

chart=('scatterplot','boxplot')
selectchart=st.selectbox('select chart',chart)
# plt.xticks('x',rotation=vertical)
if selectchart=='scatterplot':
                              sns.scatterplot(x='total_bill',y='tip',hue=select,data=df)
else:
    sns.boxplot(x='total_bill', y='tip', hue=select, data=df)


st.pyplot(fig)

import numpy as np
st.write('linechart,area,line using numpy')
data=pd.DataFrame(np.random.randint(low=10,high=20,size=(5,3)),columns=['A','B','C'])
st.line_chart(data)
st.bar_chart(data)
st.area_chart(data)


#import plotly.express as px
#fig=px.histogram(x='total_bill',data=df)
#st.plotly_chart(fig)
import streamlit as st
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource

# Sample data
x = [1, 3, 4]
y = [3, 4, 5]

# Create a Bokeh figure
p = figure(title='Line Chart', x_axis_label='X axis', y_axis_label='Y axis')

# Plot a line
p.line(x, y)

# Display the Bokeh figure in Streamlit
st.bokeh_chart(p)


