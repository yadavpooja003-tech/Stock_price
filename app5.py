import streamlit as st
import pandas as pd 
import numpy as np 
st.title("Streamlit App")
st.subheader("Brewed with Streamlit")
st.text("Welcome to your first interactive app")
st.write("choose your fav.variety of tea")
tea = st.selectbox("your favruite chai :",["Masala tea","Rose Tea","Black tea","Green tea ","Kesar tea"])
st.write(f"you choose {tea}.Excellent choice")
# st.succcess("your tea has been brewed")

st.write("Pick one programming language")
pro_lan = st.selectbox("Programming languag are:",["java","python","c++","java script"])
st.write(f"you choose{pro_lan}.Excellent choice")
st.success("Now start coding in your language")
if st.button("Make chai"):
    st.success("your chai is being brewed")

add_masala = st.checkbox("you want to add masala")
if add_masala:
    st.write("masala has been added to your tea")
    
quantity = st.slider("selcect quantity of tea",1,10,3)    
st.write(f"you have selected{quantity} cups of tea")
   
