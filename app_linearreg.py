import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
st.set_page_config("Simple Linear Regression ", layout="centered")
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("styles.css")
#Title 
st.markdown(""" 
     <div class="card">
         <h1>Linear Regression</h1>
         <p>Predict <b>Tip Amount </b> from <b>Total Bill</b> using Linear Regeression... </p></div>""", unsafe_allow_html=True)  
# Load dataset
@st.cache_data     
def load_data():
    return sns.load_dataset("tips")
df=load_data()
#Dataset Preview
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>',unsafe_allow_html=True)
#Data Preparation 
x,y=df[["total_bill"]],df["tip"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
#Train model
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
#Metrics
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)
adj_r2=1-(1-r2)*(len(y_test)-1)/(len(y_test)-2)
#Visualization
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Model Performance")
fig,ax=plt.subplots()
ax.scatter(df["total_bill"],df["tip"],alpha=0.6,color="lightblue")
ax.plot(df["total_bill"],model.predict(scaler.transform(x)),color="lightpink")
ax.set_xlabel("Total Bill ($) ")
ax.set_ylabel("Tip ($)")
st.pyplot(fig)
st.markdown('</div>',unsafe_allow_html=True)
#Performance
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Model Performance Metrics")
c1,c2=st.columns(2)
c1.metric("Mean Absolute Error (MAE)",f"{mae:.2f}")
c2.metric("Root Mean Squared Error (RMSE)",f"{rmse:.2f}")
c3,c4=st.columns(2)
c3.metric("R-squared",f"{r2:.2f}")
c4.metric("Adjusted R-squared",f"{adj_r2:.2f}")
st.markdown('</div>',unsafe_allow_html=True)
#m & c
st.markdown(f"""
    <div class="card">
        <h3>Model Intercept</h3>
        <p><b>Coefficient:</b> {model.coef_[0]:.3f}</p>
        <p><b>Intercept:</b> {model.intercept_:.3f}</p>
    </div>
""", unsafe_allow_html=True)                                    
#Prediction
st.markdown('<div class="card">', unsafe_allow_html=True)
bill = st.slider("Bill Amount", float(df.total_bill.min()), float(df.total_bill.max()),30.0)
bill_df = pd.DataFrame([[bill]], columns=["total_bill"])
tip = model.predict(scaler.transform(bill_df))[0]
st.markdown(f'<div class="prediction-box">Predicted Tip: ${tip:.2f}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)